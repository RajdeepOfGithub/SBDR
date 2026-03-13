# SBDR — Sentimental-Behavioral Debt Recovery

### A Multi-Modal AI Framework for Compassionate Collections

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?logo=huggingface&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189FDD)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Phase_A-Complete-brightgreen)
![Status](https://img.shields.io/badge/Phase_B-In_Progress-yellow)

**Team:** Roy, Neel, Tanya
**Course:** DATA 606 Capstone — UMBC
**Hardware:** Mac M3 Pro, 24GB Unified RAM

---

## Overview

An AI system that replaces traditional "pay or else" debt collection with compassionate, data-driven interventions. Combines three ML approaches — **FinBERT** for sentiment analysis, **BiLSTM** for spending anomaly detection, and **XGBoost** for risk classification — to assign customers to 4 recovery tiers based on financial distress patterns.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | Pandas, Polars | Financial time-series handling |
| NLP | HuggingFace FinBERT | Financial sentiment → Distress Score |
| Deep Learning | PyTorch BiLSTM (MPS) | Sequential spending pattern detection |
| ML Classification | XGBoost | Multi-signal tier prediction |
| Explainability | SHAP, LIME | Per-customer prediction explanations |
| Synthetic Data | OpenAI GPT-4o | Distress-calibrated chat generation |
| Dashboard | Streamlit | Relationship manager interface |

---

## Dataset

Multi-modal dataset built from 5 sources:

| Source | Records | Role |
|--------|---------|------|
| UCI Credit Card | 30,000 | Anchor — 6-month payment history |
| Lending Club | 2.26M | Demographics + loan outcomes |
| Sparkov Synthetic | 1.85M | Daily transaction profiles |
| Financial PhraseBank | 4,840 | Financial sentiment text |
| Mental Health Sentiment | 27,977 | Emotional distress language |

**Merge strategy:** Synthetic Customer IDs with distress-aligned mapping.
**Final dataset:** 30,000 customers x 88 columns, zero nulls.

---

## The 4 Recovery Tiers

| Tier | Name | Trigger | Action |
|------|------|---------|--------|
| **1** | Standard Reminder | Low risk, neutral sentiment | Automated SMS/email |
| **2** | Soft Assistance | High volatility, neutral sentiment | Offer "skip a payment" |
| **3** | Hardship Restructuring | High distress, declining spend | 6-month repayment plan |
| **4** | Legal Recovery | Zero communication, total payment stop | Escalate to collections |

---

## Project Status

### Phase A — Data Preparation: COMPLETE

- [x] **A1** — UCI exploration + initial features (Notebook 01)
- [x] **A2** — Load remaining datasets: LC, Sparkov, PhraseBank, Mental Health (Notebook 02)
- [x] **A3** — Merge all datasets via synthetic Customer ID — 30K x 57 (Notebook 03)
- [x] **A4** — Cross-modal feature engineering — 30K x 85 (Notebook 04)
- [x] **A5** — Synthetic chat generation via GPT-4o — 30K x 88 (Notebook 05)

### Phase B — Model Training: IN PROGRESS

- [ ] **B1** — Fine-tune FinBERT → Distress Score (0.0–1.0)
- [x] **B2** — Train BiLSTM → Stress Pattern Vector (Notebook 06)
- [ ] **B3** — Train XGBoost → Recovery Tier Prediction
- [ ] **B4** — SHAP + LIME Explainability

### Phase C — Deployment: NOT STARTED

- [ ] **C1** — Streamlit Dashboard
- [ ] **C2** — Fairness Audit
- [ ] **C3** — Final Testing + Documentation

---

## Project Structure

```
SBDR_main/
├── notebooks/
│   ├── Phase 1/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_remaining_datasets.ipynb
│   │   ├── 03_dataset_merge.ipynb
│   │   ├── 04_feature_engineering.ipynb
│   │   └── 05_synthetic_chats.ipynb
│   ├── Phase 2/
│   │   └── DL/
│   │       └── 06_bilstm_stress_patterns.ipynb
│   └── Testing Dataset.ipynb
├── data/
│   ├── raw/          (not tracked)
│   └── processed/    (not tracked)
├── src/
├── dashboard/
├── docs/
├── team/
├── .env              (not tracked)
├── .gitignore
├── requirements.txt
└── README.md
```
