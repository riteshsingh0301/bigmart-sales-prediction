BigMart Sales Prediction — Analytics Vidhya Hackathon

Problem Statement
Predict unit sales for 1,559 products across 10 BigMart retail outlets. The key structural challenge: no item-outlet pair appears in both train and test sets, making this a matrix completion problem rather than a standard regression.
Approach
Feature Engineering

32 features derived from EDA — MRP price tiers, outlet age, item category prefixes, visibility ratios, and cross-interaction features
Smoothed target encodings computed per fold to prevent leakage

Model Architecture

Layer 1: CatBoost (RMSE + MAE objectives) + Bilinear LOO + MRP Lookup, blended via Ridge meta-learner
Layer 2: Two independently configured CatBoost models retrained on an expanded 14,204-row dataset, blended 70/30
