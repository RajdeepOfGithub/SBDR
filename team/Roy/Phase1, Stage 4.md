# Phase A, Step A4 — Cross-Modal Feature Engineering

**Notebook:** `notebooks/04_feature_engineering.ipynb`
**Input:** `data/processed/sbdr_merged_dataset.csv` (30K × 57)
**Output:** `data/processed/sbdr_featured_dataset.csv` (30K × 85)

---

## Objective
Engineer cross-modal features that leverage the merged dataset — combining signals across UCI, Lending Club, and Sparkov that weren't possible before A3. Also prep temporal sequences for the BiLSTM and encode categorical columns for XGBoost.

---

## Feature Groups

### Group 1: Utilization & Capacity Stress (11 new features)
- `util_ratio_1` through `util_ratio_6` — credit utilization per month (BILL_AMT / LIMIT_BAL)
- `avg_util_ratio` — mean utilization across 6 months
- `util_trend` — slope of utilization over time (positive = maxing out more)
- `spend_to_limit` — Sparkov monthly spend vs credit limit
- `bill_to_income` — annualized bill burden vs LC income

**Correlation with default:**
| Feature | Correlation |
|---------|------------|
| avg_util_ratio | 0.1155 |
| spend_to_limit | 0.0698 |
| util_trend | 0.0191 |
| bill_to_income | 0.0061 |

### Group 2: Temporal Trends — BiLSTM prep (7 new features)
- `pay_amt_slope` — payment amount trajectory (paying less over time?)
- `bill_amt_slope` — bill amount trajectory (bills growing?)
- `gap_trend` — payment-to-bill gap widening?
- `months_delinquent` — count of months with PAY > 0
- `worst_month_delay` — peak delinquency severity
- `delay_acceleration` — are delays getting worse? (late months minus early months)
- `recovery_flag` — was customer ever late then recovered to on-time?

**Correlation with default:**
| Feature | Correlation |
|---------|------------|
| months_delinquent | 0.3984 |
| worst_month_delay | 0.3310 |
| delay_acceleration | 0.1693 |

### Group 3: Cross-Modal Interactions (6 new features)
- `dti_x_delinquency` — LC debt-to-income × months delinquent
- `rate_x_util` — LC interest rate × avg utilization
- `inquiry_x_delay` — LC recent inquiries × worst month delay
- `volatility_mismatch` — abs(Sparkov volatility - UCI volatility)
- `sp_fraud_x_delinq` — Sparkov fraud rate × months delinquent
- `income_stress` — bill-to-income × months delinquent

**Correlation with default:**
| Feature | Correlation |
|---------|------------|
| dti_x_delinquency | 0.3876 |
| rate_x_util | 0.1375 |
| inquiry_x_delay | 0.2145 |

### Group 4: Categorical Encoding (4 new features)
- `distress_encoded` — ordinal: low=0, moderate=1, high=2, severe=3
- `pb_label_encoded` — ordinal: positive=0, neutral=1, negative=2
- `mh_status_encoded` — ordinal: Normal=0, Stress=1, Anxiety=2, Depression=3
- `sp_top_category_encoded` — frequency encoding for spending category

**Note:** Text columns (`pb_sentence`, `mh_statement`) left raw — FinBERT handles those in Phase B.

---

## Top 10 Features by Correlation with Default

| Rank | Feature | Correlation | Source |
|------|---------|------------|--------|
| 1 | months_delinquent | 0.3984 | A4 — Group 2 |
| 2 | dti_x_delinquency | 0.3876 | A4 — Group 3 |
| 3 | mh_status_encoded | 0.3552 | A4 — Group 4 |
| 4 | worst_month_delay | 0.3310 | A4 — Group 2 |
| 5 | distress_encoded | 0.3158 | A4 — Group 4 |
| 6 | PAY_0 | 0.3246 | UCI original |
| 7 | pb_label_encoded | 0.2614 | A4 — Group 4 |
| 8 | inquiry_x_delay | 0.2145 | A4 — Group 3 |
| 9 | delay_acceleration | 0.1693 | A4 — Group 2 |
| 10 | avg_util_ratio | 0.1155 | A4 — Group 1 |

**Key insight:** A4 cross-modal features dominate the top 10. The A3 merge was justified — combining LC DTI with UCI payment delays created stronger signal than either dataset alone.

---

## Output

**Shape:** 30,000 × 85 columns (57 from A3 + 28 from A4)
**Nulls:** 0

### Sequence Columns for BiLSTM (Phase B)
- PAY_0, PAY_2–6 (delinquency sequence)
- BILL_AMT1–6 (bill trajectory)
- PAY_AMT1–6 (payment trajectory)
- util_ratio_1–6 (utilization sequence)
- pay_ratio_1–6 (payment ratio sequence)

### Text Columns for FinBERT (Phase B)
- pb_sentence (financial sentiment)
- mh_statement (emotional distress)