# SBDR — Sentimental-Behavioral Debt Recovery

### A Multi-Modal AI Framework for Compassionate Collections

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189FDD)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-38%2F38_Passing-brightgreen)](tests/)
[![Phase A](https://img.shields.io/badge/Phase_A-Complete-brightgreen)]()
[![Phase B](https://img.shields.io/badge/Phase_B-Complete-brightgreen)]()
[![Phase C](https://img.shields.io/badge/Phase_C-Complete-brightgreen)]()

**Team:** Rajdeep Roy · Neel · Tanya
**Course:** DATA 606 — Capstone in Data Science, UMBC
**Date:** Spring 2026

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [The 3 AI Branches](#the-3-ai-branches)
- [The 5 Recovery Tiers](#the-5-recovery-tiers)
- [Results](#results)
- [Dashboard](#dashboard)
- [Getting Started](#getting-started)
- [Running the Project Step by Step](#running-the-project-step-by-step)
- [Running Tests](#running-tests)
- [Key Features](#key-features)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [References](#references)
- [Team Contributions](#team-contributions)

---

## Overview

Traditional debt collection treats every customer the same — threatening calls, mailed letters, credit damage — regardless of whether someone genuinely can't pay or is choosing not to. SBDR replaces that with a data-driven approach that reads customer emotions, detects payment behavior anomalies, and assigns each person to the right recovery tier.

The system combines three machine learning branches:

1. **FinBERT NLP** — reads customer chat messages and scores financial distress (0 to 1)
2. **BiLSTM Autoencoder** — analyzes 6 months of payment history to flag behavioral anomalies
3. **XGBoost Classifier** — fuses all signals into a 5-tier recovery classification

A **rule-based audit layer** sits on top to catch strategic defaulters the model misses.

**Key results:** 93.4% accuracy · 0.990 AUC-ROC · 76.1% macro F1 · 485 strategic defaulters flagged (+121% over model alone)

---

## Research Question

> **Can a multi-modal AI framework that combines NLP, sequential deep learning, and gradient-boosted classification accurately distinguish between customers experiencing genuine financial hardship and those strategically defaulting on debt — and assign each to an appropriate recovery intervention?**

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Processing | Pandas, NumPy | Financial time-series handling |
| NLP | HuggingFace FinBERT (zero-shot) | Chat sentiment → Distress Score (0–1) |
| Deep Learning | PyTorch BiLSTM Autoencoder | Sequential payment pattern + anomaly detection |
| ML Classification | XGBoost 3.2 | Multi-signal 5-tier recovery prediction |
| Explainability | SHAP | Per-feature and per-branch contribution analysis |
| Synthetic Data | OpenAI GPT-4o | Distress-calibrated customer chat generation |
| Dashboard | Streamlit | Relationship manager command center |
| Testing | pytest | 38 unit tests covering full pipeline |

---

## Repository Structure
SBDR/
├── README.md                    # This file
├── LIMITATIONS.md               # Detailed project limitations
├── requirements.txt             # Full dependency list
├── requirements_deploy.txt      # Lightweight deployment dependencies
├── dashboard.py                 # Streamlit dashboard entry point
├── .gitignore
│
├── notebooks/                   # Ordered Jupyter notebooks (run 01→09)
│   ├── 01_uci_cleaning.ipynb           # UCI Credit Card data cleaning
│   ├── 02_lending_club_merge.ipynb     # Lending Club merge + demographics
│   ├── 03_sparkov_aggregation.ipynb    # Sparkov transaction profiles
│   ├── 04_chat_generation.ipynb        # GPT-4o synthetic chat generation
│   ├── 05_feature_engineering.ipynb    # 22 derived features
│   ├── 06_eda.ipynb                    # Exploratory data analysis
│   ├── 07_finbert_sentiment.ipynb      # FinBERT distress scoring
│   ├── 08_bilstm_xgboost.ipynb        # BiLSTM autoencoder + XGBoost training
│   └── 09_audit_layer.ipynb            # Rule-based strategic default detection
│
├── src/                         # Reusable Python modules
│   ├── data_loader.py
│   ├── feature_engine.py
│   ├── finbert_scorer.py
│   ├── bilstm_model.py
│   ├── xgboost_classifier.py
│   └── audit_layer.py
│
├── data/
│   ├── raw/                     # Original datasets (not tracked — see Dataset section)
│   └── processed/               # Cleaned + merged outputs from notebooks
│
├── models/                      # Trained model artifacts
│   └── xgb_model.json           # Final XGBoost model (2.1 MB)
│
├── dashboard/                   # Dashboard components and config
├── tests/                       # Unit tests (38 tests, all passing)
│   └── test_pipeline.py
├── docs/                        # Additional documentation
└── team/                        # Team member info

---

## Dataset

Built from 5 complementary real-world sources:

| Source | Records | Role in SBDR |
|--------|---------|-------------|
| [UCI Credit Card](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) | 30,000 | Anchor — 6-month payment history, demographics |
| [Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | 2.26M | Income, employment, loan outcomes |
| [Sparkov Synthetic](https://github.com/namebrandon/Sparkov_Data_Generation) | 1.85M | Daily transaction profiles |
| [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) | 4,840 | Financial sentiment calibration |
| Mental Health Sentiment | 27,977 | Emotional distress language patterns |

**Final merged dataset:** 30,000 customers × 88 columns × 0 null values

> ⚠️ **Raw datasets are not tracked in this repo** due to size. Download them from the links above and place in `data/raw/`, then run notebooks 01–05 in order to reproduce the processed dataset.

---

## The 3 AI Branches

### Branch 1: FinBERT (NLP)
- Pre-trained transformer model from HuggingFace (`ProsusAI/finbert`)
- Zero-shot classification — no manual labeling needed
- Reads customer chat messages → outputs a **distress score** from 0.0 (calm) to 1.0 (severe)
- SHAP contribution: **5.2%** of final model decision
- Handled by: **Tanya**

### Branch 2: BiLSTM Autoencoder (Deep Learning)
- Bidirectional LSTM reads 6 months of payment data forwards and backwards
- Learns "normal" payment patterns, flags anomalies via high reconstruction error
- Generates **28 stress vector features** per customer (94 → 122 total features)
- SHAP contribution: **26.1%** of final model decision
- Handled by: **Neel**

### Branch 3: XGBoost Classifier (Machine Learning)
- Receives all 122 features from both branches + original data
- Gradient-boosted decision trees with 5-fold cross-validated hyperparameter tuning
- Multi-class classification into 5 recovery tiers
- Trained model size: **2.1 MB**
- Handled by: **Rajdeep**

### Audit Layer (All Three)
- Post-classification rule: Default + Low Distress + Anomaly Flag → Tier 5
- Increased strategic default detection from 219 → **485** (+121%)
- Designed collaboratively by all team members

---

## The 5 Recovery Tiers

| Tier | Name | Rule | Action |
|------|------|------|--------|
| **1** | Standard Reminder | Low distress, no default | Send automated payment reminder |
| **2** | Soft Assist | Moderate distress, no default | Offer flexible payment plan |
| **3** | Hardship Program | High distress, no default | Connect to financial counselor |
| **4** | Severe Genuine Distress | Severe distress OR (default + distress) | Pause collections, offer relief |
| **5** | Strategic Default | Default + low distress + anomaly | Escalate — can pay but won't |

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | **93.4%** |
| AUC-ROC (macro) | **0.990** |
| F1-Score (weighted) | **93.6%** |
| F1-Score (macro) | **76.1%** |
| Strategic Defaults Flagged | **485** (with audit layer) |
| Audit Layer Improvement | **+121%** over model alone |

### Per-Tier F1 Scores

| Tier | Name | F1 Score |
|------|------|----------|
| 1 | Standard Reminder | 96% |
| 2 | Soft Assist | 92% |
| 3 | Hardship Program | 78% |
| 4 | Severe Genuine Distress | 58% |
| 5 | Strategic Default | 80% |

### SHAP Feature Branch Contributions

| Branch | Contribution |
|--------|-------------|
| UCI / Derived Features | 66.4% |
| BiLSTM Stress Vectors | 26.1% |
| FinBERT Distress Score | 5.2% |
| Sparkov Transactions | 2.3% |

### Fairness Audit
Tier distribution was analyzed across **sex**, **age**, and **education** — no statistically significant disparity was found in any dimension. The fairness audit is built into the Streamlit dashboard for real-time monitoring.

---

## Dashboard

The Streamlit dashboard serves as the relationship manager's command center with four panels:

1. **Customer Roster** — Browse all 30K customers, filter by tier, risk level, demographics
2. **Multi-Branch Evidence** — View FinBERT score + BiLSTM flags + XGBoost prediction side by side
3. **Fairness Audit** — Real-time demographic distribution across all 5 tiers
4. **Strategic Default Panel** — Review the 485 flagged customers before escalation

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- GPU recommended for BiLSTM training (CPU works but slower)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/RajdeepOfGithub/SBDR.git
cd SBDR

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Quick Start (Dashboard Only)

If you just want to see the dashboard with pre-computed results:

```bash
pip install -r requirements_deploy.txt
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`.

---

## Running the Project Step by Step

The notebooks are numbered and must be run in order. Each notebook builds on the output of the previous one.
Step 1:  notebooks/01_uci_cleaning.ipynb           → Cleans UCI Credit Card data
Step 2:  notebooks/02_lending_club_merge.ipynb      → Merges Lending Club demographics
Step 3:  notebooks/03_sparkov_aggregation.ipynb     → Aggregates Sparkov transactions
Step 4:  notebooks/04_chat_generation.ipynb         → Generates synthetic chats (needs OpenAI API key)
Step 5:  notebooks/05_feature_engineering.ipynb      → Creates 22 derived features
Step 6:  notebooks/06_eda.ipynb                     → Exploratory data analysis + visualizations
Step 7:  notebooks/07_finbert_sentiment.ipynb       → Runs FinBERT distress scoring
Step 8:  notebooks/08_bilstm_xgboost.ipynb          → Trains BiLSTM + XGBoost
Step 9:  notebooks/09_audit_layer.ipynb             → Applies rule-based audit layer
Final:   streamlit run dashboard.py                 → Launches the dashboard

> **Note:** Notebook 04 requires an OpenAI API key for GPT-4o chat generation. Set it as an environment variable: `export OPENAI_API_KEY=your_key_here`. If you skip this step, pre-generated chats are available in `data/processed/`.

---

## Running Tests

```bash
# Run all 38 unit tests
pytest tests/test_pipeline.py -v
```

Tests cover:
- Data integrity and schema validation
- Feature engineering correctness
- Model prediction consistency
- Tier assignment logic
- Audit layer accuracy
- Edge cases and boundary conditions

---

## Key Features

- **Multi-modal AI** — Combines NLP (text), deep learning (sequences), and ML (tabular) in one pipeline
- **5 granular tiers** — Not just "pay or escalate" but 5 distinct recovery strategies
- **Rule-based audit layer** — Catches 121% more strategic defaulters than the model alone
- **Full SHAP explainability** — Every single decision can be explained feature by feature
- **Fairness built in** — Demographic audit across sex, age, and education, monitored in real time
- **Production-ready** — 2.1 MB model, millisecond inference, Streamlit dashboard for relationship managers
- **Comprehensive testing** — 38 unit tests covering the full pipeline

---

## Limitations

1. **Synthetic chat data** — Customer chats were generated by GPT-4o, not collected from real interactions. Real chat data would improve FinBERT accuracy.
2. **Static dataset** — Model operates on historical data. Real-time deployment would need streaming infrastructure.
3. **English only** — No multi-language support yet.
4. **No live A/B testing** — Results are from held-out test sets, not real-world recovery outcomes.
5. **Tier 4 overlap** — At 58% F1, Tier 4 has the most confusion with neighboring tiers.

See [LIMITATIONS.md](LIMITATIONS.md) for full details.

---

## Future Roadmap

| Phase | Timeline | Description |
|-------|----------|-------------|
| Phase 1 | Q3 2026 | Real-time API, bank CRM integration, live chat processing |
| Phase 2 | Q1 2027 | Voice/call transcript analysis, agent coaching |
| Phase 3 | Q3 2027 | Reinforcement learning — system improves from repayment outcomes |
| Phase 4 | 2028+ | Multi-language, EU/UK compliance, top-50 agency partnerships |

---

## References

### Papers
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD, 785–794
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017

### Tools & Libraries
Python 3.11 · PyTorch · HuggingFace Transformers · XGBoost · SHAP · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · OpenAI GPT-4o API

---

## Team Contributions

| Member | Primary Responsibility |
|--------|----------------------|
| **Tanya** | FinBERT NLP pipeline, chat generation, fairness audit design |
| **Neel** | BiLSTM autoencoder, Streamlit dashboard, data engineering |
| **Rajdeep** | XGBoost classifier, evaluation framework, SHAP analysis |
| **All three** | Audit layer design, system architecture, testing |

---

<p align="center">
  <b>DATA 606 Capstone — University of Maryland, Baltimore County — Spring 2026</b>
</p>
