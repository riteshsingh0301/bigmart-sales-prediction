# BigMart Sales Prediction
# Analytics Vidhya BigMart Sales III
# Final Score: 1144.67 RMSE  |  Rank 160  |  Top 0.3%
#
# This script reproduces the complete solution end to end.
# It covers data loading, exploratory analysis, feature engineering,
# model training, stacking, and final submission generation.
#
# Stack: CatBoost RMSE + CatBoost MAE + ALS Matrix Factorisation
#        + Entity Embedding Neural Network, blended via Ridge regression.
#
# Requirements: catboost, torch, scipy, scikit-learn, pandas, numpy, matplotlib


# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

# All random seeds are fixed so the notebook produces identical results
# on every run. The neural network uses three seeds which are averaged
# to reduce variance from random weight initialisation.
SEED     = 42
N_FOLDS  = 10
ALS_K    = 5
NN_SEEDS = [42, 123, 456]

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# =============================================================================
# 2. DATA LOADING
# =============================================================================

TRAIN_URL = "https://datahack-prod.s3.amazonaws.com/train_file/train_v9rqX0R.csv"
TEST_URL  = "https://datahack-prod.s3.amazonaws.com/test_file/test_AbJTz2l.csv"

train_raw = pd.read_csv(TRAIN_URL)
test_raw  = pd.read_csv(TEST_URL)
test_raw["Item_Outlet_Sales"] = np.nan

print(f"Train: {train_raw.shape}  |  Test: {test_raw.shape}")
print(f"Train sales mean: {train_raw['Item_Outlet_Sales'].mean():.2f}")


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

# The target variable Item_Outlet_Sales has a strong right skew.
# We examine it before deciding on the transformation strategy.
print("\nTarget distribution summary:")
print(train_raw["Item_Outlet_Sales"].describe().round(2))
print(f"Skewness (raw): {train_raw['Item_Outlet_Sales'].skew():.3f}")
print(f"Skewness (log): {np.log1p(train_raw['Item_Outlet_Sales']).skew():.3f}")

# Outlet Type explains roughly 46 percent of total sales variance.
# This is measured by eta squared, the ratio of between-group sum of
# squares to total sum of squares. It is the single most important
# predictor in the dataset.
grand_mean = train_raw["Item_Outlet_Sales"].mean()
groups = [
    train_raw[train_raw["Outlet_Type"] == t]["Item_Outlet_Sales"].values
    for t in train_raw["Outlet_Type"].unique()
]
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total   = ((train_raw["Item_Outlet_Sales"] - grand_mean) ** 2).sum()
print(f"\nOutlet Type eta squared: {ss_between / ss_total:.4f}")

# Mean sales by outlet type gives a sense of how different the store
# formats are from each other.
print("\nMean sales by outlet type:")
print(train_raw.groupby("Outlet_Type")["Item_Outlet_Sales"].mean().round(0).sort_values())

# Item MRP correlation with sales improves after log transformation
# because the relationship between price and sales is multiplicative
# rather than additive at high price points.
log_mrp   = np.log(train_raw["Item_MRP"])
log_sales = np.log1p(train_raw["Item_Outlet_Sales"])
print(f"\nItem MRP vs sales correlation (raw):      {train_raw['Item_MRP'].corr(train_raw['Item_Outlet_Sales']):.4f}")
print(f"log(MRP) vs log(sales) correlation:        {log_mrp.corr(log_sales):.4f}")

# The most important feature engineering insight in this project.
# When visibility is normalised by the item type mean it correlates
# 0.21 with log sales. When normalised by each item's own mean across
# all outlets it correlates 0.64. The per-item ratio captures whether
# an item is receiving more or less shelf space than it normally does,
# which is a proxy for promotional display activity.
combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
item_vis_mean = (
    combined_raw[combined_raw["Item_Visibility"] > 0]
    .groupby("Item_Identifier")["Item_Visibility"].mean()
)
type_vis_mean = (
    combined_raw[combined_raw["Item_Visibility"] > 0]
    .groupby("Item_Type")["Item_Visibility"].mean()
)
tr_vis = train_raw[train_raw["Item_Visibility"] > 0].copy()
tr_vis["vis_per_item"] = tr_vis["Item_Visibility"] / tr_vis["Item_Identifier"].map(item_vis_mean)
tr_vis["vis_per_type"] = tr_vis["Item_Visibility"] / tr_vis["Item_Type"].map(type_vis_mean)
log_s = np.log1p(tr_vis["Item_Outlet_Sales"])
print(f"\nVisibility per-type  correlation: {tr_vis['vis_per_type'].corr(log_s):.4f}")
print(f"Visibility per-item  correlation: {tr_vis['vis_per_item'].corr(log_s):.4f}")

# Test set coverage check. All 1559 item identifiers in the test set
# also appear in training, and all 10 outlets match. This confirms
# there are no cold-start scenarios that would require special handling.
test_items   = set(test_raw["Item_Identifier"].unique())
train_items  = set(train_raw["Item_Identifier"].unique())
test_outlets = set(test_raw["Outlet_Identifier"].unique())
train_outlets= set(train_raw["Outlet_Identifier"].unique())
print(f"\nTest item coverage   : {len(test_items & train_items)}/{len(test_items)} = 100%")
print(f"Test outlet coverage : {len(test_outlets & train_outlets)}/{len(test_outlets)} = 100%")

# Exactly one row exists per (item, outlet) pair in the training data.
# This structural constraint is crucial: it means any item-level target
# encoding with smoothing collapses to approximately 95 percent of the
# global mean, providing almost no signal. Six separate attempts at
# item-level target encodings all degraded model performance.
rows_per_pair = train_raw.groupby(["Item_Identifier", "Outlet_Identifier"]).size()
print(f"\nRows per (item, outlet) pair: {rows_per_pair.value_counts().to_dict()}")


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

# We combine train and test into a single dataframe for feature
# computation. Every feature here is derived from non-target columns
# and computed on the full combined dataset. Features that use the
# target variable are computed later inside the cross-validation loop.

combined = pd.concat([train_raw, test_raw], ignore_index=True)
combined["is_train"] = [True] * len(train_raw) + [False] * len(test_raw)

# Fat content labels in the raw data use six different strings for three
# actual categories. LF and low fat both mean Low Fat; reg means Regular.
# Items whose identifier starts with NC are non-consumable and cannot
# meaningfully be labelled by fat content.
fat_map = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
combined["Item_Fat_Content"] = combined["Item_Fat_Content"].replace(fat_map)
combined.loc[combined["Item_Identifier"].str[:2] == "NC", "Item_Fat_Content"] = "Non-Edible"

# Item Weight is missing for roughly 17 percent of rows. Because weight
# is a physical property of the product and does not vary across stores,
# the per-item identifier mean achieves 100 percent coverage.
item_wt = (
    combined.dropna(subset=["Item_Weight"])
    .groupby("Item_Identifier")["Item_Weight"].first()
)
mask = combined["Item_Weight"].isna()
combined.loc[mask, "Item_Weight"] = combined.loc[mask, "Item_Identifier"].map(item_wt)

# Zero visibility is physically impossible for a sold product. These
# 879 rows represent data collection errors and are replaced with the
# per-item mean of non-zero visibility values.
item_vis_m = (
    combined[combined["Item_Visibility"] > 0]
    .groupby("Item_Identifier")["Item_Visibility"].mean()
)
mask = combined["Item_Visibility"] == 0
combined.loc[mask, "Item_Visibility"] = combined.loc[mask, "Item_Identifier"].map(item_vis_m)

# Three outlets have missing Outlet Size. Cross-referencing with their
# Outlet Type and Outlet Location Type confirms all three are Small
# grocery stores in Tier 2 cities.
for outlet in ["OUT010", "OUT017", "OUT045"]:
    combined.loc[combined["Outlet_Identifier"] == outlet, "Outlet_Size"] = "Small"

# Item category derived from the first two and three characters of the
# Item Identifier. These characters encode the product category and
# sub-category respectively.
combined["Item_Category"] = combined["Item_Identifier"].str[:2]
combined["Item_Cat3"]     = combined["Item_Identifier"].str[:3]

# Outlet age captures the maturity of each store. The data collection
# year is 2013 so we compute 2013 minus the establishment year.
combined["Outlet_Age"]      = 2013 - combined["Outlet_Establishment_Year"]
combined["Outlet_Type_Num"] = combined["Outlet_Type"].map(
    {"Grocery Store": 0, "Supermarket Type1": 1,
     "Supermarket Type2": 2, "Supermarket Type3": 3}
)

# Price features. Log MRP improves the linear correlation from 0.51
# to 0.53. MRP squared captures non-linear effects at the high end
# of the price distribution. The MRP bucket feature encodes the four
# natural price tiers visible as gaps in a scatter plot of MRP values.
combined["log_MRP"]          = np.log(combined["Item_MRP"])
combined["Item_MRP_sq"]      = combined["Item_MRP"] ** 2
combined["MRP_x_OutletType"] = combined["Item_MRP"] * combined["Outlet_Type_Num"]
combined["MRP_Rank_in_Type"] = combined.groupby("Item_Type")["Item_MRP"].rank(pct=True)
combined["MRP_Bucket"]       = pd.cut(
    combined["Item_MRP"],
    bins=[0, 50, 100, 150, 200, 300],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# Visibility features. The per-item ratio is the most important
# engineered feature in the entire solution with a correlation of 0.64
# compared to 0.21 for the per-type version.
item_vis_avg = combined.groupby("Item_Identifier")["Item_Visibility"].mean()
type_vis_avg = combined.groupby("Item_Type")["Item_Visibility"].mean()
combined["Vis_ratio_per_item"]        = combined["Item_Visibility"] / combined["Item_Identifier"].map(item_vis_avg)
combined["Item_Visibility_MeanRatio"] = combined["Item_Visibility"] / combined["Item_Type"].map(type_vis_avg)

# Interaction features between item and outlet properties.
combined["Price_per_Weight"] = combined["Item_MRP"] / combined["Item_Weight"]
combined["Weight_x_MRP"]     = combined["Item_Weight"] * combined["Item_MRP"]

# Cross categorical features encode combinations of product category
# and outlet type that may behave differently from their individual parts.
combined["Outlet_ItemCat"]    = combined["Outlet_Type"] + "_" + combined["Item_Category"]
combined["Item_Outlet_Type"]  = combined["Item_Type"].astype(str) + "_" + combined["Outlet_Type"].astype(str)
combined["ItemCat_x_OutType"] = combined["Item_Category"] + "_" + combined["Outlet_Type"]
combined["Outlet_x_Type"]     = combined["Outlet_Identifier"] + "_" + combined["Outlet_Type"]

# Aggregate features computed on the full combined dataset capture
# the baseline visibility profile and MRP level for each item.
# No target variable is used here so these are safe to compute globally.
item_vis_agg = combined.groupby("Item_Identifier").agg(
    Item_Vis_mean_full=("Item_Visibility", "mean"),
    Item_Vis_std_full=("Item_Visibility", "std")
).reset_index()
item_vis_agg["Item_Vis_std_full"] = item_vis_agg["Item_Vis_std_full"].fillna(0)
combined = combined.merge(item_vis_agg, on="Item_Identifier", how="left")
combined = combined.merge(
    combined.groupby("Item_Identifier")["Item_MRP"].mean().rename("Item_MRP_mean_full"),
    on="Item_Identifier", how="left"
)
combined["Outlet_Item_Count_full"] = combined.groupby(
    "Outlet_Identifier")["Item_Identifier"].transform("count")

print(f"\nFeature engineering complete. Combined shape: {combined.shape}")


# =============================================================================
# 5. ENCODING FOR TREE MODELS AND NEURAL NETWORK
# =============================================================================

# CatBoost handles string categorical features natively without encoding.
# We only need to prepare the feature list and identify which columns
# should be treated as categorical inside CatBoost.
CAT_FEATURES = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "log_MRP",
    "Vis_ratio_per_item", "Item_Visibility_MeanRatio",
    "Price_per_Weight", "Weight_x_MRP",
    "MRP_x_OutletType", "MRP_Rank_in_Type", "Outlet_Age",
    "Item_MRP_mean_full", "Item_Vis_mean_full", "Item_Vis_std_full",
    "Outlet_Item_Count_full",
    "Item_Fat_Content", "Item_Type", "Item_Category", "Item_Cat3",
    "Outlet_Type", "Outlet_Size", "Outlet_Location_Type",
    "Outlet_ItemCat", "Item_Outlet_Type",
    "Outlet_Identifier_enc", "ItemCat_x_OutType", "Outlet_x_Type",
    "Item_MRP_sq", "MRP_Bucket",
]
CAT_FEATURE_NAMES = [
    "Item_Fat_Content", "Item_Type", "Item_Category", "Item_Cat3",
    "Outlet_Type", "Outlet_Size", "Outlet_Location_Type",
    "Outlet_ItemCat", "Item_Outlet_Type",
    "Outlet_Identifier_enc", "ItemCat_x_OutType", "Outlet_x_Type",
]

# The target encoding columns below are the six group columns for which
# we will compute smoothed target encodings inside each CV fold.
# These are the only features derived from the target variable.
TE_COLS = [
    "Outlet_Identifier_enc", "Item_Type", "Outlet_ItemCat",
    "Item_Outlet_Type", "ItemCat_x_OutType", "Outlet_x_Type",
]

data_cat = combined.copy()
data_cat["Outlet_Identifier_enc"] = data_cat["Outlet_Identifier"]

train_df = data_cat[data_cat["is_train"]].reset_index(drop=True)
test_df  = data_cat[~data_cat["is_train"]].reset_index(drop=True)

# All models predict log1p(sales) and predictions are inverted
# with expm1 before computing RMSE or creating submissions.
y         = np.log1p(train_df["Item_Outlet_Sales"].values)
strat_col = train_df["Outlet_Type_Num"].astype(int)
cat_cols  = [c for c in CAT_FEATURE_NAMES if c in CAT_FEATURES]

# Neural network encoding. Each categorical feature needs to be converted
# to integer indices for the embedding lookup layers. The embedding
# dimension for each feature is set to min(50, (n_categories + 1) // 2),
# which is a common rule of thumb that prevents overly large embeddings
# for high cardinality features like Item Identifier.
EMB_COLS = [
    "Item_Identifier", "Outlet_Identifier", "Item_Type",
    "Outlet_Type", "Item_Category", "Item_Fat_Content", "Outlet_Size"
]
for col in EMB_COLS:
    le = LabelEncoder()
    combined[col + "_idx"] = le.fit_transform(combined[col].astype(str))

EMB_SIZES = {
    col: (combined[col + "_idx"].nunique(),
          min(50, max(2, (combined[col + "_idx"].nunique() + 1) // 2)))
    for col in EMB_COLS
}
emb_sizes_list = [EMB_SIZES[c] for c in EMB_COLS]

# Continuous features for the neural network. These are standardised
# using statistics from the training set only.
CONT_COLS = [
    "Item_Weight", "Item_Visibility", "Item_MRP", "log_MRP",
    "Vis_ratio_per_item", "Item_Visibility_MeanRatio", "Price_per_Weight",
    "Weight_x_MRP", "MRP_x_OutletType", "MRP_Rank_in_Type", "Outlet_Age",
    "Item_MRP_mean_full", "Item_Vis_mean_full", "Item_Vis_std_full",
    "Outlet_Item_Count_full",
]

nn_tr_raw = combined[combined["is_train"]].reset_index(drop=True)
nn_te_raw = combined[~combined["is_train"]].reset_index(drop=True)
scaler    = StandardScaler()
nn_tr_cont = scaler.fit_transform(nn_tr_raw[CONT_COLS].values)
nn_te_cont  = scaler.transform(nn_te_raw[CONT_COLS].values)
nn_tr_emb  = nn_tr_raw[[c + "_idx" for c in EMB_COLS]].values
nn_te_emb  = nn_te_raw[[c + "_idx" for c in EMB_COLS]].values
n_cont     = len(CONT_COLS)

# ALS matrix factorisation requires integer index maps from item
# and outlet identifiers to matrix row and column positions.
item_ids   = sorted(combined["Item_Identifier"].unique())
outlet_ids = sorted(combined["Outlet_Identifier"].unique())
item_idx   = {v: i for i, v in enumerate(item_ids)}
outlet_idx = {v: j for j, v in enumerate(outlet_ids)}
n_items, n_outlets = len(item_ids), len(outlet_ids)

# Stratified 10-fold cross-validation with stratification on Outlet Type
# ensures every fold has proportional representation of all four outlet
# formats. Without stratification, some folds could lack a particular
# outlet type entirely, making fold-level metrics unreliable.
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(train_df, strat_col))

print("Encoding complete.")
print(f"Tree features : {len(CAT_FEATURES)} base + {len(TE_COLS)} target encodings")
print(f"Neural network: {len(EMB_COLS)} embedding cols + {n_cont} continuous features")


# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def smoothed_te(train_fold_df, apply_df, group_col,
                target_col="Item_Outlet_Sales", k=20):
    """
    Compute a smoothed target encoding for a single group column.

    The smoothing formula is:
        encoded = (n x group_mean + k x global_mean) / (n + k)

    With k=20, a group needs at least 20 observations before its mean
    is trusted over the global mean. Groups with fewer observations are
    pulled toward the global mean, which prevents overfitting on rare
    categories like small outlets with few products.
    """
    global_mean = train_fold_df[target_col].mean()
    stats = train_fold_df.groupby(group_col)[target_col].agg(["mean", "count"])
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + k * global_mean) /
        (stats["count"] + k)
    )
    return apply_df[group_col].map(stats["smoothed"]).fillna(global_mean)


def add_te_features(X_tr, X_val, X_te, fold_tr, train_df, test_df, cols, k=20):
    """
    Add smoothed target encoding features for all specified group columns.

    The training fold encodings are computed from fold training rows only.
    Validation encodings use the same fold training statistics applied to
    validation rows, so no validation target information is used.
    Test encodings use the full training dataset statistics.

    This three-way split ensures there is no leakage between training
    rows, validation rows, and test rows at any stage.
    """
    for col in cols:
        te        = col + "_TE"
        X_tr[te]  = smoothed_te(fold_tr, fold_tr,  col, k=k).values
        X_val[te] = smoothed_te(fold_tr, X_val,    col, k=k).values
        X_te[te]  = smoothed_te(train_df, test_df, col, k=k).values
    return X_tr, X_val, X_te


def als_oof(train_df, test_df, tr_idx, val_idx, k=5):
    """
    Compute ALS matrix factorisation predictions using proper
    out-of-fold implementation.

    The 1559 x 10 sales matrix is built exclusively from fold training
    entries. Item and outlet bias terms are subtracted before SVD so
    the decomposition captures only the interaction signal rather than
    the main effects. The latent factors are then used to predict both
    validation rows and test rows.

    Using k=5 latent factors. With only 10 outlets, higher k quickly
    leads to overfitting because there are insufficient outlet columns
    to support a meaningful high-rank decomposition.
    """
    fold_tr = train_df.iloc[tr_idx]
    rows = np.array([item_idx[i]   for i in fold_tr["Item_Identifier"].values])
    cols = np.array([outlet_idx[j] for j in fold_tr["Outlet_Identifier"].values])
    vals = np.log1p(fold_tr["Item_Outlet_Sales"].values)

    item_means   = np.zeros(n_items)
    outlet_means = np.zeros(n_outlets)
    for i in range(n_items):
        mask = rows == i
        if mask.sum() > 0:
            item_means[i] = vals[mask].mean()
    for j in range(n_outlets):
        mask = cols == j
        if mask.sum() > 0:
            outlet_means[j] = vals[mask].mean()

    global_mean = vals.mean()
    centered    = vals - item_means[rows] - outlet_means[cols] + global_mean
    M           = coo_matrix((centered, (rows, cols)), shape=(n_items, n_outlets)).tocsr()
    U, s, Vt    = svds(M, k=min(k, min(n_items, n_outlets) - 1))
    M_pred      = U @ np.diag(s) @ Vt + item_means[:, None] + outlet_means[None, :] - global_mean

    fold_val = train_df.iloc[val_idx]
    vr = np.array([item_idx[i]   for i in fold_val["Item_Identifier"].values])
    vc = np.array([outlet_idx[j] for j in fold_val["Outlet_Identifier"].values])
    tr2 = np.array([item_idx[i]  for i in test_df["Item_Identifier"].values])
    tc2 = np.array([outlet_idx[j] for j in test_df["Outlet_Identifier"].values])

    return M_pred[vr, vc], M_pred[tr2, tc2]


print("Helper functions defined.")


# =============================================================================
# 7. ENTITY EMBEDDING NEURAL NETWORK
# =============================================================================

# The neural network is the model that separates this solution from a
# pure gradient boosting approach. Trees cannot learn item-level demand
# signal because there is exactly one row per (item, outlet) combination,
# which makes any target encoding collapse to near-constant noise. The
# neural network learns a 50-dimensional embedding for each item identity
# directly from the prediction gradient without needing a target encoding.

class SalesDataset(Dataset):
    """
    A PyTorch Dataset that holds embedding indices for categorical
    features and standardised continuous features together.
    """
    def __init__(self, emb, cont, targets=None):
        self.emb     = torch.LongTensor(emb)
        self.cont    = torch.FloatTensor(cont)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        if self.targets is not None:
            return self.emb[i], self.cont[i], self.targets[i]
        return self.emb[i], self.cont[i]


class EntityEmbeddingNet(nn.Module):
    """
    A multi-layer perceptron with entity embeddings for categorical features.

    Each categorical feature gets its own embedding layer. The embeddings
    are concatenated with the standardised continuous features and passed
    through a three-layer MLP with batch normalisation and dropout.

    Item Identifier receives a 50-dimensional embedding. This embedding
    encodes each item's demand profile across outlet types and is the
    primary source of item-level signal in the model. Outlet Identifier
    receives a 5-dimensional embedding encoding each store's sales
    multiplier and product mix characteristics.
    """
    def __init__(self, emb_sizes, n_cont, hidden=[256, 128, 64], dropout=0.4):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, n_dims) for n_cats, n_dims in emb_sizes
        ])
        n_emb = sum(d for _, d in emb_sizes)
        n_in  = n_emb + n_cont
        layers = []
        in_sz  = n_in
        for h in hidden:
            layers += [
                nn.Linear(in_sz, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_sz = h
        layers += [nn.Linear(in_sz, 1)]
        self.mlp = nn.Sequential(*layers)
        # Small random initialisation keeps embeddings near zero at the
        # start of training so the network can explore from a neutral
        # starting point without any single item dominating early updates.
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, 0, 0.01)

    def forward(self, x_emb, x_cont):
        embs = [e(x_emb[:, i]) for i, e in enumerate(self.embeddings)]
        x    = torch.cat(embs + [x_cont], dim=1)
        return self.mlp(x).squeeze(1)


def train_nn_fold(tr_emb, tr_cont, tr_y, val_emb, val_cont, val_y,
                  nn_seed, epochs=500, patience=40, batch_size=256):
    """
    Train one instance of the neural network for a single CV fold.

    A key finding from experimentation: using learning rate 3e-4 instead
    of the typical 1e-3 produces worse training OOF (1168 vs 1144) but
    better leaderboard score (1144.67 vs 1145.06). The slower training
    creates smoother, more regularised predictions that generalise better
    to the test set distribution.

    The learning rate scheduler reduces the learning rate by half whenever
    the validation RMSE stops improving for 10 consecutive epochs. The
    minimum learning rate is set to 1e-5 to prevent complete stagnation.
    """
    torch.manual_seed(nn_seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(nn_seed)

    model = EntityEmbeddingNet(emb_sizes_list, n_cont).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5, min_lr=1e-5
    )

    tr_dl  = DataLoader(
        SalesDataset(tr_emb, tr_cont, tr_y), batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(SalesDataset(val_emb, val_cont, val_y), batch_size=512)

    best_val, best_ep, best_state = np.inf, 0, None

    for ep in range(epochs):
        model.train()
        for x_emb, x_cont, yt in tr_dl:
            x_emb = x_emb.to(DEVICE)
            x_cont = x_cont.to(DEVICE)
            yt     = yt.to(DEVICE)
            opt.zero_grad()
            pred = model(x_emb, x_cont)
            loss = torch.sqrt(torch.mean((pred - yt) ** 2))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.cat([
                model(xe.to(DEVICE), xc.to(DEVICE)).cpu()
                for xe, xc, _ in val_dl
            ]).numpy()

        val_rmse = np.sqrt(mean_squared_error(
            np.expm1(val_y), np.expm1(val_preds)
        ))
        sched.step(val_rmse)

        if val_rmse < best_val:
            best_val, best_ep = val_rmse, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep - best_ep >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_val, best_ep


def predict_nn(model, emb, cont):
    """
    Generate predictions from a trained neural network for a given
    set of embedding indices and continuous features.
    """
    model.eval()
    dl = DataLoader(SalesDataset(emb, cont), batch_size=512)
    with torch.no_grad():
        return torch.cat([
            model(xe.to(DEVICE), xc.to(DEVICE)).cpu()
            for xe, xc in dl
        ]).numpy()


print("Neural network defined.")
print(f"Architecture: {sum(d for _, d in emb_sizes_list)} embedding dims + {n_cont} continuous -> 256 -> 128 -> 64 -> 1")


# =============================================================================
# 8. MODEL PARAMETERS
# =============================================================================

# CatBoost RMSE parameters were tuned using Optuna with 10-fold
# cross-validation as the objective function. Depth 4 is deliberately
# shallow to prevent overfitting given only 8,523 training rows.
CAT_RMSE_PARAMS = dict(
    iterations=5000,
    early_stopping_rounds=200,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=3.127,
    bagging_temperature=0.034,
    random_strength=1.819,
    min_data_in_leaf=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    verbose=False,
)

# CatBoost MAE requires deeper trees because the MAE gradient is weaker
# than the RMSE gradient and needs more splits to converge. The
# leaf_estimation_iterations parameter improves convergence speed for
# non-squared losses by performing multiple Newton steps per leaf.
# This model predicts the conditional median rather than the conditional
# mean, making it robust to the high-value outlier rows in the right tail.
CAT_MAE_PARAMS = dict(
    iterations=5000,
    early_stopping_rounds=200,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=5.0,
    bagging_temperature=0.5,
    random_strength=2.0,
    min_data_in_leaf=8,
    leaf_estimation_iterations=10,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=SEED,
    verbose=False,
)

print("Model parameters defined.")


# =============================================================================
# 9. TRAINING LOOP
# =============================================================================

# All four models are trained inside the same cross-validation loop so
# their out-of-fold predictions are always computed on held-out data.
# Test predictions are accumulated as the average across all ten folds.

oof_cr  = np.zeros(len(train_df)); test_cr  = np.zeros(len(test_df))
oof_cm  = np.zeros(len(train_df)); test_cm  = np.zeros(len(test_df))
oof_als = np.zeros(len(train_df)); test_als = np.zeros(len(test_df))
oof_nn  = np.zeros(len(train_df)); test_nn  = np.zeros(len(test_df))

print("\nStarting 10-fold training. Expected runtime: 50 minutes on GPU.\n")

for fold, (tr_idx, val_idx) in enumerate(FOLDS, 1):
    fold_tr  = train_df.iloc[tr_idx]
    fold_val = train_df.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Tree model features. Target encodings are computed from fold
    # training rows only so validation rows never influence their own
    # encoding statistics.
    X_tr  = fold_tr[CAT_FEATURES].copy()
    X_val = fold_val[CAT_FEATURES].copy()
    X_te  = test_df[CAT_FEATURES].copy()
    X_tr, X_val, X_te = add_te_features(
        X_tr, X_val, X_te, fold_tr, train_df, test_df, TE_COLS
    )

    for oof_arr, test_arr, params in [
        (oof_cr, test_cr, CAT_RMSE_PARAMS),
        (oof_cm, test_cm, CAT_MAE_PARAMS),
    ]:
        m = CatBoostRegressor(**params)
        m.fit(
            Pool(X_tr, y_tr, cat_features=cat_cols),
            eval_set=Pool(X_val, y_val, cat_features=cat_cols)
        )
        oof_arr[val_idx] = m.predict(X_val)
        test_arr        += m.predict(X_te) / N_FOLDS

    # ALS matrix factorisation is computed per fold using only the
    # training entries for that fold.
    val_preds, te_preds = als_oof(train_df, test_df, tr_idx, val_idx, k=ALS_K)
    oof_als[val_idx] = val_preds
    test_als        += te_preds / N_FOLDS

    # Three neural network seeds are trained per fold and their
    # predictions are averaged. This reduces variance from random
    # weight initialisation. With one seed, the fold RMSE ranged from
    # 1091 to 1190 across folds. Averaging three seeds reduces the
    # effective standard deviation by roughly a factor of root three.
    tr_emb  = nn_tr_emb[tr_idx]; val_emb  = nn_tr_emb[val_idx]
    tr_cont = nn_tr_cont[tr_idx]; val_cont = nn_tr_cont[val_idx]
    vp_nn   = []; tp_nn = []
    for nn_seed in NN_SEEDS:
        model, best_rmse, best_ep = train_nn_fold(
            tr_emb, tr_cont, y_tr, val_emb, val_cont, y_val, nn_seed
        )
        vp_nn.append(predict_nn(model, val_emb, val_cont))
        tp_nn.append(predict_nn(model, nn_te_emb, nn_te_cont))
    oof_nn[val_idx] = np.mean(vp_nn, axis=0)
    test_nn        += np.mean(tp_nn, axis=0) / N_FOLDS

    cr_f = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_cr[val_idx])))
    cm_f = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_cm[val_idx])))
    nn_f = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_nn[val_idx])))
    print(f"Fold {fold:2d}  CatRMSE={cr_f:.0f}  CatMAE={cm_f:.0f}  EE-NN={nn_f:.0f}")

print("\nIndividual model OOF RMSE:")
for name, oof in [("CatRMSE", oof_cr), ("CatMAE", oof_cm),
                   ("ALS",    oof_als), ("EE-NN", oof_nn)]:
    r = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))
    print(f"  {name:10s}: {r:.4f}")


# =============================================================================
# 10. RIDGE META-LEARNER
# =============================================================================

# Ridge regression is fitted on the four sets of out-of-fold predictions
# from the base models. The alpha hyperparameter is tuned via nested
# 5-fold cross-validation on this OOF matrix. All four base model weights
# should be positive, confirming that each model adds orthogonal signal
# that the others do not capture.

meta_tr = np.column_stack([oof_cr, oof_cm, oof_als, oof_nn])
meta_te = np.column_stack([test_cr, test_cm, test_als, test_nn])

best_alpha, best_rv = None, np.inf
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\nTuning Ridge alpha:")
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    rmses = []
    for ti, vi in skf_meta.split(meta_tr, strat_col):
        m = Ridge(alpha=alpha)
        m.fit(meta_tr[ti], y[ti])
        preds = m.predict(meta_tr[vi])
        rmses.append(np.sqrt(mean_squared_error(np.expm1(y[vi]), np.expm1(preds))))
    mr = np.mean(rmses)
    print(f"  alpha={alpha:7.3f}  meta RMSE={mr:.4f}")
    if mr < best_rv:
        best_rv, best_alpha = mr, alpha

ridge = Ridge(alpha=best_alpha)
ridge.fit(meta_tr, y)

oof_stack  = ridge.predict(meta_tr)
stack_rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof_stack)))

names = ["CatRMSE", "CatMAE", "ALS", "EE-NN"]
print(f"\nBest alpha   : {best_alpha}")
print(f"Stack OOF    : {stack_rmse:.4f}")
print("Ridge weights:")
for name, coef in zip(names, ridge.coef_):
    print(f"  {name:10s}: {coef:+.4f}")


# =============================================================================
# 11. FINAL PREDICTIONS AND POST-PROCESSING
# =============================================================================

# The Ridge meta-learner is applied to the test set stacking features.
# Predictions are in log space and must be converted back with expm1.
# Negative predictions are clipped to zero as sales cannot be negative.
stack_test_log = ridge.predict(meta_te)
final_preds    = np.clip(np.expm1(stack_test_log), 0, None)

# Mean scaling anchors the prediction mean to the known training mean.
# This corrects for any systematic shift introduced by the log transform
# and model calibration differences across folds.
#
# An MRP-bucket correction was tested in an earlier version of this
# solution. It applied bias adjustments derived from OOF residuals
# grouped by (outlet, MRP range) across 50 cells. It appeared to
# reduce training OOF error but compressed prediction standard deviation
# from 1340 to 1330 and hurt leaderboard performance by approximately
# 1 point. The correction was overfitting to training fold residual
# patterns that did not hold on the test set distribution. It was
# permanently removed from version 20 onward.
scale       = train_raw["Item_Outlet_Sales"].mean() / final_preds.mean()
final_preds = np.clip(final_preds * scale, 0, None)

print(f"\nPrediction mean : {final_preds.mean():.1f}  (train mean: {train_raw['Item_Outlet_Sales'].mean():.1f})")
print(f"Prediction std  : {final_preds.std():.1f}")
print(f"Scale factor    : {scale:.6f}")

submission = pd.DataFrame({
    "Item_Identifier":   test_raw["Item_Identifier"],
    "Outlet_Identifier": test_raw["Outlet_Identifier"],
    "Item_Outlet_Sales": final_preds,
})
submission.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")
print(submission.describe().round(1))


# =============================================================================
# 12. RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  Stack OOF RMSE     : {stack_rmse:.4f}")
print(f"  Leaderboard RMSE   : 1144.67")
print(f"  Rank               : 160 of 53,720")
print(f"  Top percentile     : 0.3%")
print()
print("  Ridge weights:")
for name, coef in zip(names, ridge.coef_):
    print(f"    {name:12s}: {coef:+.4f}")
print()
print("  Key findings from this solution:")
print("  1. Outlet Type explains 46% of variance. It is the most important predictor.")
print("  2. CatBoost MAE added approx 5 LB points over a pure-RMSE stack.")
print("  3. MRP correction removed in v20. It was compressing prediction variance.")
print("  4. Item-level target encodings always failed. One row per pair is the reason.")
print("  5. Entity embeddings solved item identity without target encoding.")
print("  6. Lower NN learning rate (3e-4) generalised better despite worse OOF.")
print("=" * 60)
