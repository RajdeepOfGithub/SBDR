# 📓 Notebook 01 — What I Did & What I Found

> **Date:** Feb 13, 2026 **Author:** Rajdeep **File:** `notebooks/01_data_loading.ipynb` **Dataset:** UCI Credit Card Default (30K customers, 6 months of payment history)

---

## What This Notebook Covers

I loaded and explored the **UCI Credit Card dataset** — this is our anchor dataset. Everything else gets mapped onto these 30,000 customers. I wanted to understand the data before we start building models, so this is pure exploration and feature engineering.

---

## Key Findings

### 1. No Cleaning Needed

Zero null values. All columns are `int64`. The dataset is surprisingly clean out of the box.

### 2. Class Imbalance

- 78% did NOT default
- 22% defaulted

This is imbalanced. We'll need to handle this later with SMOTE or class weighting when training XGBoost. Otherwise the model will just predict "no default" every time and still be 78% accurate.

### 3. Payment History Is Our Strongest Signal

`PAY_0` through `PAY_6` (repayment status over 6 months) are the **top correlated features** with default. `PAY_0` alone has a correlation of 0.325. This validates our entire BiLSTM approach — payment sequences ARE the most predictive thing in this dataset.

### 4. The Financial Spiral Is Real

I plotted average bill amounts vs average payments over 6 months. Bills climb steadily from ~39K to ~51K TWD, but payments stay flat at ~5K. The gap widens every month. Customers are spending more but paying the same — classic debt spiral. This is exactly the pattern our LSTM will learn to detect.

### 5. Demographics Are Weak Predictors

`AGE`, `SEX`, `EDUCATION`, `MARRIAGE` all have near-zero correlation with default on their own. They won't drive predictions, but we still keep them for:

- XGBoost (might find combinations that matter)
- Fairness audit in Phase C (making sure the model isn't biased)

### 6. Quick Note on PAY_1

There is no `PAY_1` column — it's a known quirk of this dataset. The columns go `PAY_0, PAY_2, PAY_3...PAY_6`. All 6 months of data are present, just oddly named.

---

## Features I Engineered

I created 3 new features from the raw data:

|Feature|What It Is|Correlation w/ Default|Status|
|---|---|---|---|
|`spending_volatility`|Std dev of bill amounts across 6 months|-0.0798|✅ Kept|
|`pay_ratio_1` to `pay_ratio_6`|Monthly payment ÷ monthly bill|Individual ratios|✅ Kept|
|`pay_ratio_trend`|Slope of payment ratios over time|0.0002|❌ Dropped|
|`delinq_accel`|Rate of change in payment delay status|-0.1297|✅ Kept|

**Why I dropped `pay_ratio_trend`:** Correlation was basically zero (0.0002). The extreme outliers in individual month ratios made the trend line meaningless. The individual `pay_ratio_1–6` columns are more useful — the BiLSTM can learn the trend directly from them.

**Best engineered feature:** `delinq_accel` at -0.1297. It captures how fast a customer is spiraling — the faster the deterioration, the higher the default risk.

---

## What I Saved

Processed dataset saved to: `data/processed/uci_credit_processed.csv`

- **Shape:** 30,000 rows × 32 columns
- Original 25 cols - `ID` + `spending_volatility` + 6 `pay_ratio` cols + `delinq_accel`

---

## What's Next

This was **Phase A, Step A1**. Next steps:

- **A2:** Load the other 5 datasets (Lending Club, Sparkov, PhraseBank, Mental Health, Reddit)
- **A3:** Merge everything via synthetic Customer IDs
- **A4:** Full feature engineering on merged data
- **A5:** Generate synthetic customer chats using GPT-4o

---

> [!note] For Neel & Tanya You don't need to run this notebook to continue your work. The processed CSV in `data/processed/` has everything. But I'd recommend skimming the notebook to understand the data — especially the payment history patterns and the bill vs payment chart. It'll help when you're building the BiLSTM and FinBERT branches.