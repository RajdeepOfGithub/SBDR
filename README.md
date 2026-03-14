# SBDR — Sentimental-Behavioral Debt Recovery

### A Multi-Modal AI Framework for Compassionate Collections

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?logo=huggingface&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189FDD)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Phase_A-Complete-brightgreen)
![Status](https://img.shields.io/badge/Phase_B-Complete-brightgreen)
![Status](https://img.shields.io/badge/Phase_C-In_Progress-yellow)

**Team:** Roy, Neel, Tanya
**Course:** DATA 606 Capstone — UMBC

---

## Overview

An AI system that replaces traditional "pay or else" debt collection with compassionate, data-driven interventions. Combines three ML branches — **FinBERT** for chat sentiment scoring, **BiLSTM** for payment behavior anomaly detection, and **XGBoost** for multi-signal risk classification — to assign 30,000 customers to 5 recovery tiers with a rule-based strategic default audit layer.

**Key results:** 93.4% accuracy · 0.990 AUC-ROC · 76.1% macro F1 · 485 strategic defaulters flagged

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | Pandas, NumPy | Financial time-series handling |
| NLP | HuggingFace FinBERT (zero-shot) | Chat sentiment → Distress Score (0–1) |
| Deep Learning | PyTorch BiLSTM | Sequential payment pattern + anomaly detection |
| ML Classification | XGBoost | Multi-signal 5-tier recovery prediction |
| Explainability | SHAP | Per-feature branch contribution analysis |
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

**Final dataset:** 30,000 customers × 88 columns, zero nulls.

> ⚠️ Datasets are not tracked in this repo. See `data/raw/` and `data/processed/` — run notebooks in order to reproduce.

---

## The 5 Recovery Tiers

| Tier | Name | Rule |
|------|------|------|
| **1** | Standard Reminder | Low distress, no default |
| **2** | Soft Assist | Moderate distress, no default |
| **3** | Hardship | High distress, no default |
| **4** | Severe Genuine Distress | Severe distress OR (default + high distress) |
| **5** | Strategic Default | Default + low distress + payment anomaly flag |

---

## Results (Phase B)

| Metric | Value |
|--------|-------|
| Accuracy | 93.4% |
| AUC-ROC (macro OvR) | 0.990 |
| F1 weighted | 93.6% |
| F1 macro | 76.1% |
| Strategic defaults flagged (audit layer) | 485 (+121% vs model alone) |

**SHAP branch contributions:** UCI/Derived 66.4% · BiLSTM 26.1% · FinBERT 5.2% · Sparkov 2.3%

---

## Project Status

### Phase A — Data Preparation: COMPLETE ✅

- [x] **A1** — UCI exploration + initial features (Notebook 01)
- [x] **A2** — Load remaining datasets: LC, Sparkov, PhraseBank, Mental Health (Notebook 02)
- [x] **A3** — Merge all datasets via synthetic Customer ID — 30K × 57 (Notebook 03)
- [x] **A4** — Cross-modal feature engineering — 30K × 85 (Notebook 04)
- [x] **A5** — Synthetic chat generation via GPT-4o — 30K × 88 (Notebook 05)

### Phase B — Model Training: COMPLETE ✅

- [x] **B1** — FinBERT zero-shot distress scoring — 30K × 94 (Notebook 07)
- [x] **B2** — BiLSTM autoencoder stress vectors + anomaly detection — 30K × 122 (Notebook 06)
- [x] **B3** — XGBoost 5-tier classifier — 93.4% acc, 0.990 AUC (Notebook 08)
- [x] **B3.5** — Rule-based strategic default audit layer — 485 flagged (Notebook 09)
- [ ] **B4** — SHAP + LIME explainability (in progress)

### Phase C — Deployment: IN PROGRESS

- [ ] **C1** — Streamlit Dashboard
- [ ] **C2** — Fairness Audit
- [ ] **C3** — Final Testing + Documentation

---

## Project Structure

```
SBDR/
├── notebooks/
│   ├── Phase 1/
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_remaining_datasets.ipynb
│   │   ├── 03_dataset_merge.ipynb
│   │   ├── 04_feature_engineering.ipynb
│   │   └── 05_synthetic_chats.ipynb
│   └── Phase 2/
│       ├── DL/
│       │   └── 06_bilstm_stress_patterns.ipynb
│       ├── NLP/
│       │   └── 07_finbert_distress_score.ipynb
│       └── ML/
│           ├── 08_xgboost_recovery_tier.ipynb
│           └── 09_audit_layer.ipynb
├── models/
│   ├── bilstm/       — manifest, plots, train_val_indices.npz
│   ├── finbert/      — manifest, plots
│   └── xgboost/      — manifest, plots, best_model.json
├── data/
│   ├── raw/          (not tracked — run notebooks to reproduce)
│   └── processed/    (not tracked — run notebooks to reproduce)
├── src/
├── dashboard/
├── docs/
├── team/
├── .gitignore
├── requirements.txt
└── README.md
```

> ⚠️ **Model weights** for BiLSTM (`.pt`) and FinBERT (`.pt`, 418MB) are not tracked.
> Run `06_bilstm_stress_patterns.ipynb` and `07_finbert_distress_score.ipynb` to reproduce.
> The trained XGBoost model (`models/xgboost/best_model.json`) is included.
