# BigMart Sales Prediction

**Competition:** Analytics Vidhya BigMart Sales III  
**Final Score:** 1144.67 RMSE  
**Rank:** 160 of 53,720  
**Percentile:** Top 0.3%

---

## Problem Statement

BigMart collected 2013 sales data for 1559 products across 10 retail outlets. Each row represents a unique combination of one product and one store. The objective is to predict Item Outlet Sales for 5,681 product-store combinations in the test set. The evaluation metric is Root Mean Squared Error.

---

## Solution Overview

The solution is a two-layer stacking ensemble. Four base models produce out-of-fold predictions via 10-fold stratified cross-validation. A Ridge regression meta-learner then finds the optimal linear combination of those predictions.

**Layer 1 Base Models**

| Model | OOF RMSE | Ridge Weight | Role |
|---|---|---|---|
| CatBoost RMSE | 1106.26 | 0.585 | Backbone, predicts conditional mean |
| CatBoost MAE | 1076.20 | 0.193 | Predicts conditional median, robust to outliers |
| ALS Matrix Factorisation | 1402.50 | 0.031 | Item and outlet latent factors |
| Entity Embedding NN | 1168.88 | 0.135 | Item identity signal via learned embeddings |

**Layer 2**

Ridge regression with alpha tuned via nested 5-fold CV. Stack OOF RMSE: 1103.44.

---

## Repository Structure

```
bigmart-sales-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── bigmart_solution.py      <- Complete solution script
└── outputs/
    └── submission.csv       <- Best submission file
```

---

## Key Findings

**Outlet Type is the most decisive variable.** Eta squared of 0.46 means it explains 46 percent of total sales variance on its own. Supermarket Type 3 averages 3,694 units per product versus 340 for Grocery stores. No amount of item-level feature engineering compensates for anchoring outlet identity in every model.

**The target distribution requires careful handling.** Raw sales skewness is 1.72. A log1p transform reduces this to 0.23 and prevents high-value rows from dominating the squared loss. This single preprocessing decision has more downstream impact than most feature additions.

**Per-item visibility normalisation is three times more informative than per-type.** Dividing visibility by the item's own historical mean gives a correlation of 0.64 with log sales. The same ratio at the category level gives 0.21. The per-item version captures promotional shelf space changes; the per-type version blurs this signal across many different items.

**Combining RMSE and MAE loss functions was the largest single gain.** RMSE optimises for the conditional mean and is pulled toward outlier high-sale rows. MAE optimises for the conditional median and is robust to those same outliers. The two losses capture complementary aspects of the right-skewed sales distribution and adding CatBoost MAE to a pure-RMSE stack delivered approximately 5 leaderboard points.

**Item-level target encodings always fail on this dataset.** The training data has exactly one row per (item, outlet) combination. With smoothing parameter k=20 and one observation per group, any item-level encoding returns approximately 95 percent of the global mean. Six separate attempts with different formulations all degraded performance. The Entity Embedding Neural Network is the correct way to exploit item identity because it learns embeddings directly from the prediction gradient without any target encoding step.

**Post-processing corrections require empirical leaderboard validation.** An MRP-bucket correction that applied bias adjustments from OOF residuals appeared to reduce training error but compressed prediction standard deviation from 1340 to 1330 and hurt leaderboard performance by 1 point. Corrections derived from training fold patterns do not reliably transfer to the test distribution.

**Lower neural network learning rate generalises better.** Using lr=3e-4 instead of the typical 1e-3 produced worse training OOF (1168 vs 1144) but better leaderboard score (1144.67 vs 1145.06). Slower, smoother training creates predictions that generalise better to the test set distribution.

---

## Feature Engineering

The final feature set contains 33 features: 29 base features derived from item and outlet properties, and 6 smoothed target encodings computed exclusively inside cross-validation folds.

**Price features (6):** log MRP, MRP squared, MRP bucket, MRP rank within item type, MRP multiplied by outlet type numeric, price per weight.

**Visibility features (4):** per-item visibility ratio (primary signal, corr 0.64), per-type visibility ratio, per-item mean visibility, per-item visibility standard deviation.

**Outlet and category features (17):** outlet age, outlet type numeric, outlet item count, item category prefix, item sub-category prefix, and 12 cross-features combining outlet and item category identifiers.

**Target encodings (6):** smoothed target encodings with k=20 for outlet identifier, item type, outlet item category, item outlet type, item category outlet type, and outlet type. Computed from fold training rows only and applied separately to validation and test rows.

---

## How to Run

```bash
pip install -r requirements.txt
python bigmart_solution.py
```

The script downloads the data directly from the competition URLs, runs the full pipeline, and saves `submission.csv` in the working directory. Training requires approximately 50 minutes on a GPU (T4 or better) or 3 to 4 hours on CPU.

---

## Requirements

See `requirements.txt`. Core dependencies are catboost, torch, scikit-learn, scipy, pandas, and numpy.****
