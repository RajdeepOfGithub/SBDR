# Sentimental-Behavioral Debt Recovery (SBDR)

**A Multi-Modal AI Framework for Compassionate Collections**

---

## The Problem

Traditional debt collection systems are binary and reactive. A customer either pays or gets flagged for recovery. There is no middle ground, no context, and no empathy. The result:

- **Customer churn** — people who hit a temporary rough patch leave the bank permanently after being treated like delinquents.
- **High legal costs** — aggressive recovery pipelines escalate to legal action too early, burning money on customers who would have recovered on their own.
- **Negative social impact** — financial distress is often tied to life events (job loss, medical emergencies, family crises). A "pay or else" system ignores the human behind the account.
- **Lost lifetime value** — acquiring a new customer costs 5-25x more than retaining an existing one. Losing a 20-year banking relationship over a 3-month hardship is bad business.

The core issue is that existing models ask **"Will they pay?"** — a yes/no classification that treats all defaults the same. A single mother who lost her job and a serial fraudster get the same collection call.

---

## Our Solution

SBDR shifts the question from **"Will they pay?"** to **"What intervention will help them pay?"**

We build a proactive, multi-modal AI system that merges two signals banks have never combined before:

1. **What the customer is doing** — 6 months of payment history, spending patterns, and transaction-level behavior.
2. **How the customer feels** — sentiment extracted from customer support chats, communication tone shifts, and distress indicators.

By connecting the numbers to the words, the system detects financial distress *before* a total default and recommends personalized, compassionate interventions.

---

## How It Works

The pipeline has three branches that feed into a single decision engine:

### Branch 1 — NLP (The Words)

A fine-tuned **FinBERT** model processes customer chat logs and support tickets to produce a **Distress Score** (0.0 to 1.0). It is trained on financial language, mental health sentiment data, and real-world debt distress vocabulary from Reddit communities.

### Branch 2 — Deep Learning (The Behavior)

A **Bidirectional LSTM** processes 6 months of sequential payment and spending history to detect hidden stress patterns — things like declining payment ratios, spending volatility spikes, and category shifts that signal life events (e.g., a sudden spike in medical spending).

### Branch 3 — ML (The Decision)

An **XGBoost** classifier combines the Distress Score from Branch 1, the Stress Pattern Vector from Branch 2, and static demographic/credit features to classify the customer into one of four recovery tiers:

| Tier | Name | Trigger | Action |
|------|------|---------|--------|
| 1 | Standard Reminder | Low risk, neutral sentiment | Automated SMS/email |
| 2 | Soft Assistance | High spending volatility, neutral tone | Offer "skip a payment" |
| 3 | Hardship Restructuring | High distress, declining spend | 6-month repayment plan |
| 4 | Legal Recovery | Zero communication, total payment stop | Escalate to collections |

Every prediction is explained using **SHAP** and **LIME**, so a relationship manager can see *why* the AI recommended a specific intervention — no black boxes.

---

## Data Strategy

No single public dataset contains both financial behavior and customer sentiment. We build a multi-modal dataset by merging six sources:

**Financial (the numbers):**
- **Lending Club** — 2.26M loan records with demographics, credit scores, and loan outcomes.
- **UCI Credit Card Default** — 30,000 customers with 6 consecutive months of payment history (our anchor dataset).
- **Sparkov Synthetic Transactions** — 1.3M+ daily transactions with merchant categories for spending profile analysis.

**Sentiment (the words):**
- **Financial PhraseBank** — 4,840 expert-labeled financial sentences for FinBERT fine-tuning.
- **Mental Health Sentiment** — Anxiety, stress, and depression language patterns that mirror real customer distress.
- **Reddit Financial Sentiment** — Raw, unfiltered financial distress vocabulary from communities like r/personalfinance.

The UCI dataset's 30K customers serve as the anchor. Each customer gets a synthetic ID, enriched with Lending Club demographics, Sparkov spending profiles, and sentiment-matched chat logs generated via GPT-4o.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | Pandas, Polars | Financial time-series handling (Polars for large datasets) |
| NLP | HuggingFace FinBERT | Financial sentiment analysis and distress scoring |
| Deep Learning | PyTorch (BiLSTM) | Sequential payment/spending pattern detection |
| ML | XGBoost / CatBoost | Multi-signal classification into recovery tiers |
| Explainability | SHAP, LIME | Per-customer and global model explanations |
| Dashboard | Streamlit | Relationship manager interface |
| Database | SQLite / PostgreSQL | Historical transaction storage |

**Hardware:** Mac Pro M3 Pro | 24GB Unified Memory | PyTorch `mps` backend

---

## Project Phases

**Phase A — Data Preparation**
Collect and clean all six datasets. Merge via synthetic customer IDs. Engineer features: spending volatility, payment ratio trends, delinquency acceleration, sentiment shift scores. Generate synthetic customer chat logs calibrated to distress levels.

**Phase B — Model Training**
Fine-tune FinBERT on financial + mental health sentiment data. Train BiLSTM on 6-month payment sequences. Train XGBoost on combined outputs + demographics. Validate with SHAP/LIME explainability.

**Phase C — Deployment**
Build Streamlit dashboard for relationship managers. Run fairness audits across demographics. Document PII anonymization strategy and regulatory compliance approach.

---

## Why This Matters

- **Mental health awareness in finance** — acknowledges that financial distress is a human problem, not just a numbers problem.
- **Lifetime value over immediate recovery** — a 5% improvement in customer retention can increase profits by 25-95%.
- **Regulatory readiness** — SHAP-based explanations address the "black box" concern that has kept banks from adopting AI in collections.
- **Bias accountability** — fairness audits ensure compassionate interventions are distributed equitably across demographics.

---

## Limitations

| What | Why It Matters | How We Handle It |
|------|---------------|-----------------|
| Synthetic merge | Financial data and chat sentiment are artificially linked | Demonstrates the architecture; real bank data would provide natural linkage |
| UCI data is from 2005 Taiwan | Different economic context | Used for temporal payment patterns (universal); Lending Club adds Western demographics |
| PhraseBank is corporate language | Customers don't talk like earnings reports | Layered with mental health + Reddit data; synthetic chats generated for realism |
| Class imbalance | Defaults are ~22% of UCI data | SMOTE / class weighting in XGBoost + stratified splits |
