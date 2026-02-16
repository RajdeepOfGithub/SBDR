# Phase A, Step A3 — Merge All Datasets via Synthetic Customer ID

**Notebook:** `notebooks/03_dataset_merge.ipynb`
**Output:** `data/processed/sbdr_merged_dataset.csv`

---

## Objective
Merge 5 independent processed datasets into one multi-modal record per customer using UCI Credit Card (30K rows) as the anchor dataset.

---

## Merge Strategy

**Anchor:** UCI Credit Card — each of the 30K rows gets a unique `customer_id` (SBDR_00000 to SBDR_29999)

**Distress Classification:** Customers classified into 4 levels based on payment history (PAY_0 through PAY_6):
- **Low:** On-time or early payments consistently
- **Moderate:** Occasional delays (1 month late)
- **High:** Repeated delays (2+ months late)
- **Severe:** Persistent delinquency (3+ months late, multiple periods)

**Merge Order:**
1. UCI ← Lending Club (risk profile matching by distress tier, 12 aggregate features with noise)
2. UCI ← Sparkov (spending profiles matched by distress-spending similarity, 6 features)
3. UCI ← PhraseBank (financial sentiment text mapped by distress → sentiment alignment)
4. UCI ← Mental Health (emotional text mapped by distress → psychological state alignment)

---

## Merge Logic Details

### Merge 1: Lending Club
- Grouped LC records by risk tier → computed aggregate stats (mean loan amount, DTI, income, etc.)
- Mapped to UCI customers by matching distress level to LC risk tier
- Added Gaussian noise to prevent identical profiles within tiers

### Merge 2: Sparkov
- Computed per-customer spending profiles (total spend, avg monthly, volatility, top category, fraud rate)
- Assigned to UCI customers based on distress-spending similarity matching

### Merge 3: PhraseBank
- Mapping: low → positive/neutral, moderate → neutral, high → neutral/negative, severe → negative
- One sentence assigned per customer from eligible sentiment pool

### Merge 4: Mental Health
- Mapping: low → Normal, moderate → Normal/Stress, high → Stress/Anxiety, severe → Anxiety/Depression
- One statement assigned per customer from eligible status pool

---

## Output

**Shape:** 30,000 × 57 columns
**Nulls:** 0

### Column Groups
| Source | Columns | Count |
|--------|---------|-------|
| UCI Original | LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0–6, BILL_AMT1–6, PAY_AMT1–6, default, spending_volatility, pay_ratio_1–6, delinq_accel | 32 |
| Engineered | customer_id, avg_pay_delay, distress_level | 3 |
| Lending Club | lc_loan_amnt_mean, lc_funded_amnt_mean, lc_annual_inc_mean, lc_dti_mean, lc_revol_util_mean, lc_delinq_2yrs_mean, lc_inq_last_6mths_mean, lc_open_acc_mean, lc_pub_rec_mean, lc_total_acc_mean, lc_installment_mean, lc_int_rate_mean | 12 |
| Sparkov | sp_total_spend, sp_avg_monthly_spend, sp_spend_volatility, sp_num_transactions, sp_top_category, sp_fraud_rate | 6 |
| PhraseBank | pb_sentence, pb_label | 2 |
| Mental Health | mh_statement, mh_status | 2 |

### Distress Distribution
| Level | Count |
|-------|-------|
| Low | 10,253 |
| Moderate | 12,614 |
| High | 4,500 |
| Severe | 2,633 |

### Assigned Sentiment Distributions
**PhraseBank labels:** neutral 23,693 / negative 3,380 / positive 2,927
**Mental Health status:** Normal 21,147 / Stress 3,487 / Anxiety 3,282 / Depression 2,084

---

## Sample Customer Profile
```
Customer: SBDR_00013
Distress Level: severe
Default: 1 | Avg Pay Delay: 1.17
LC Profile: loan=10,505, DTI=19.3%, int_rate=17.0%
Sparkov: total_spend=212,337, volatility=127.0, top_cat=gas_transport
PhraseBank: [negative] Finnair believes the strike will cause daily net losses...
Mental Health: [Depression] No one cares. No one is interested...
```