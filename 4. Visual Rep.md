# SBDR — Visual Project Roadmap

## Sentimental-Behavioral Debt Recovery: A Multi-Modal AI Framework for Compassionate Collections

---

## 1. The Big Picture

```mermaid
flowchart LR
    P1["Traditional Collections\n'Pay or Else'"]
    P2["Customer Churn\nHigh Legal Costs"]
    S1["SBDR: Detect Distress\nBefore Default"]
    S2["Merge Transactions\n+ Sentiment"]
    S3["Personalized\nInterventions"]
    O1["4 Action Tiers"]
    O2["Higher Recovery\nLower Churn"]

    P1 --> P2
    P2 -->|"SBDR Solves This"| S1
    S1 --> S2
    S2 --> S3
    S3 --> O1
    O1 --> O2

    style P1 fill:#ff6b6b,color:#fff
    style P2 fill:#ff6b6b,color:#fff
    style S1 fill:#4ecdc4,color:#fff
    style S2 fill:#4ecdc4,color:#fff
    style S3 fill:#4ecdc4,color:#fff
    style O1 fill:#45b7d1,color:#fff
    style O2 fill:#45b7d1,color:#fff
```

---

## 2. Project Objectives

```mermaid
flowchart TB
    ROOT(("SBDR\nObjectives"))

    ROOT --> OBJ1["Detect Life Events"]
    ROOT --> OBJ2["Quantify Financial Anxiety"]
    ROOT --> OBJ3["Classify Recovery Personas"]
    ROOT --> OBJ4["Explainable AI"]

    OBJ1 --> O1A["Spending Anomalies"]
    OBJ1 --> O1B["Medical Bills Spike"]
    OBJ1 --> O1C["LSTM on 6-Month History"]

    OBJ2 --> O2A["NLP on Chat Logs"]
    OBJ2 --> O2B["FinBERT Distress Score"]
    OBJ2 --> O2C["Sentiment Shift Over Time"]

    OBJ3 --> O3A["Temporary Hardship"]
    OBJ3 --> O3B["Habitual Delinquency"]
    OBJ3 --> O3C["Fraudulent Intent"]

    OBJ4 --> O4A["SHAP Values"]
    OBJ4 --> O4B["LIME Explanations"]
    OBJ4 --> O4C["Regulatory Compliance"]

    style ROOT fill:#6c5ce7,color:#fff
    style OBJ1 fill:#74b9ff,color:#2d3436
    style OBJ2 fill:#a29bfe,color:#fff
    style OBJ3 fill:#fd79a8,color:#fff
    style OBJ4 fill:#ffeaa7,color:#2d3436
```

---

## 3. Tech Stack

```mermaid
flowchart LR
    D1["Pandas / Polars\nData Processing"]
    D2["SQLite / PostgreSQL\nStorage"]
    N1["HuggingFace FinBERT\nSentiment Analysis"]
    DL1["PyTorch BiLSTM\nSequence Modeling"]
    ML1["XGBoost / CatBoost\nClassification"]
    X1["SHAP / LIME\nExplainability"]
    U1["Streamlit\nDashboard"]

    D1 --> N1
    D1 --> DL1
    D2 -.->|"feeds"| D1
    N1 --> ML1
    DL1 --> ML1
    ML1 --> X1
    X1 --> U1

    style D1 fill:#f9ca24,color:#000
    style D2 fill:#f9ca24,color:#000
    style N1 fill:#6c5ce7,color:#fff
    style DL1 fill:#e17055,color:#fff
    style ML1 fill:#00b894,color:#fff
    style X1 fill:#fd79a8,color:#fff
    style U1 fill:#0984e3,color:#fff
```

**Hardware:** Mac Pro M3 Pro | 24GB Unified Memory | PyTorch `mps` device

---

## 4. Phase A — Data Preparation

### 4a. Data Sources

```mermaid
flowchart TB
    SD1["Lending Club\n2M+ rows | Loan status,\nincome, credit scores"]
    SD2["UCI Credit Card\n30K records | 6 months\npayment history"]
    SD3["Sparkov Synthetic\nMillions of transactions\nDaily spending profiles"]

    UD1["Financial PhraseBank\nSentiment-labeled\nfinancial text"]
    UD2["Mental Health Sentiment\nAnxiety & Stress\nkeywords"]
    UD3["Reddit Financial\nr/personalfinance\nReal debt distress text"]

    SD1 --> MERGE["Merge via\nCustomer ID"]
    SD2 --> MERGE
    SD3 --> MERGE
    UD1 --> MERGE
    UD2 --> MERGE
    UD3 --> MERGE

    style SD1 fill:#dfe6e9,color:#2d3436
    style SD2 fill:#dfe6e9,color:#2d3436
    style SD3 fill:#dfe6e9,color:#2d3436
    style UD1 fill:#a29bfe,color:#fff
    style UD2 fill:#a29bfe,color:#fff
    style UD3 fill:#a29bfe,color:#fff
    style MERGE fill:#ffeaa7,color:#2d3436
```

### 4b. Merge Strategy (The Secret Sauce)

```mermaid
flowchart LR
    STEP1["Select Customer ID\nfrom financial dataset"]
    STEP2["Map Sentiment\nLate payment =\nHigh Stress text"]
    STEP3["Multi-Modal Record\nSarah: declining balance\n+ anxious chat log"]
    STEP4["Feature Engineering"]

    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4

    STEP4 --> F1["Spending Volatility\nStdDev of monthly spend"]
    STEP4 --> F2["Sentiment Shift\nTone change over\nlast 3 messages"]

    style STEP1 fill:#ffeaa7,color:#2d3436
    style STEP2 fill:#ffeaa7,color:#2d3436
    style STEP3 fill:#55efc4,color:#2d3436
    style STEP4 fill:#55efc4,color:#2d3436
    style F1 fill:#74b9ff,color:#2d3436
    style F2 fill:#74b9ff,color:#2d3436
```

---

## 5. Phase B — The Multi-Modal Pipeline

```mermaid
flowchart TB
    INPUT["Merged Multi-Modal Dataset"]

    INPUT --> B1A["Chat Logs &\nSupport Tickets"]
    INPUT --> B2A["6-Month Payment\n& Spending History"]

    B1A --> B1B["FinBERT\nFine-tuned"]
    B1B --> B1C["Distress Score\n0.0 to 1.0"]

    B2A --> B2B["Bidirectional LSTM\nPyTorch MPS"]
    B2B --> B2C["Stress Pattern\nVector"]

    B1C --> B3A["XGBoost / CatBoost"]
    B2C --> B3A
    DEMO["Demographics +\nCredit History"] --> B3A

    B3A --> B3C["Recovery Tier\nPrediction"]
    B3C --> XAI["SHAP / LIME"]
    XAI --> DASH["Streamlit Dashboard"]

    style INPUT fill:#636e72,color:#fff
    style B1A fill:#6c5ce7,color:#fff
    style B1B fill:#6c5ce7,color:#fff
    style B1C fill:#6c5ce7,color:#fff
    style B2A fill:#e17055,color:#fff
    style B2B fill:#e17055,color:#fff
    style B2C fill:#e17055,color:#fff
    style DEMO fill:#dfe6e9,color:#2d3436
    style B3A fill:#00b894,color:#fff
    style B3C fill:#00b894,color:#fff
    style XAI fill:#fd79a8,color:#fff
    style DASH fill:#0984e3,color:#fff
```

> **Branch 1 (Purple):** NLP — FinBERT processes chat logs into a Distress Score
> **Branch 2 (Orange):** DL — BiLSTM detects hidden stress patterns in spending sequences
> **Branch 3 (Green):** ML — XGBoost combines both outputs + demographics into a Recovery Tier

---

## 6. Phase C — The 4 Action Tiers

```mermaid
flowchart LR
    MODEL["XGBoost\nOutput"]

    MODEL --> T1
    MODEL --> T2
    MODEL --> T3
    MODEL --> T4

    T1["TIER 1\nStandard Reminder\n---\nLow risk + Neutral tone\n→ Automated SMS/Email"]
    T2["TIER 2\nSoft Assistance\n---\nHigh volatility + Neutral tone\n→ Skip a Payment"]
    T3["TIER 3\nHardship Restructuring\n---\nHigh distress + Declining spend\n→ 6-Month Plan"]
    T4["TIER 4\nLegal Recovery\n---\nZero communication + No payment\n→ Escalate to Collections"]

    style MODEL fill:#636e72,color:#fff
    style T1 fill:#00b894,color:#fff
    style T2 fill:#fdcb6e,color:#2d3436
    style T3 fill:#e17055,color:#fff
    style T4 fill:#d63031,color:#fff
```

---

## 7. Customer Persona Classification

```mermaid
flowchart LR
    INPUT2["Customer\nData"] --> CLASSIFY["Multi-Modal\nClassifier"]

    CLASSIFY --> P1["Temporary Hardship"]
    CLASSIFY --> P2["Habitual Delinquency"]
    CLASSIFY --> P3["Fraudulent Intent"]

    P1 --> D1["Job loss or medical crisis\nHigh distress + recent change"]
    P2 --> D2["Chronic late payer\nLow distress + steady pattern"]
    P3 --> D3["No communication\nAbrupt total stop"]

    D1 --> A1["→ Tier 2 or 3\nCompassionate Intervention"]
    D2 --> A2["→ Tier 1 or 2\nBehavioral Nudges"]
    D3 --> A3["→ Tier 4\nEscalation Protocol"]

    style P1 fill:#74b9ff,color:#2d3436
    style P2 fill:#ffeaa7,color:#2d3436
    style P3 fill:#ff7675,color:#fff
    style A1 fill:#74b9ff,color:#2d3436
    style A2 fill:#ffeaa7,color:#2d3436
    style A3 fill:#ff7675,color:#fff
```

---

## 8. Data Flow — Raw to Decision

```mermaid
flowchart LR
    R["6 Raw\nDatasets"] --> C["Clean\n& Merge"]
    C --> FE["Feature\nEngineering"]
    FE --> FB["FinBERT\nFine-tune"]
    FE --> LS["BiLSTM\nTrain"]
    FB --> XG["XGBoost\nTrain"]
    LS --> XG
    XG --> SH["SHAP\nExplain"]
    SH --> FA["Fairness\nAudit"]
    FA --> ST["Streamlit\nDashboard"]

    style R fill:#dfe6e9,color:#2d3436
    style C fill:#ffeaa7,color:#2d3436
    style FE fill:#ffeaa7,color:#2d3436
    style FB fill:#6c5ce7,color:#fff
    style LS fill:#e17055,color:#fff
    style XG fill:#00b894,color:#fff
    style SH fill:#fd79a8,color:#fff
    style FA fill:#a29bfe,color:#fff
    style ST fill:#0984e3,color:#fff
```

---

## 9. Explainability & Compliance

```mermaid
flowchart TB
    PRED["Model Prediction"]

    PRED --> SHAP2["SHAP\nGlobal Feature Importance"]
    PRED --> LIME2["LIME\nLocal Per-Customer Explanation"]

    SHAP2 --> REPORT2["Explanation Report"]
    LIME2 --> REPORT2

    REPORT2 --> RM2["Relationship Manager View"]
    REPORT2 --> AUDIT2["Regulatory Audit Trail"]
    REPORT2 --> FAIR["Fairness Checks"]

    FAIR --> FC1["Demographic Parity"]
    FAIR --> FC2["Bias in Compassion Distribution"]
    FAIR --> FC3["PII Anonymization"]

    style PRED fill:#636e72,color:#fff
    style SHAP2 fill:#fd79a8,color:#fff
    style LIME2 fill:#fd79a8,color:#fff
    style REPORT2 fill:#ffeaa7,color:#2d3436
    style RM2 fill:#0984e3,color:#fff
    style AUDIT2 fill:#a29bfe,color:#fff
    style FAIR fill:#ff7675,color:#fff
```

> **RM Dashboard Example:** *"This customer was placed in Tier 3 because their Distress Score rose 40% and payments dropped 3 consecutive months."*

---

## 10. Business Impact

```mermaid
flowchart LR
    C1["New Customer Acquisition\n5-25x MORE expensive\nthan retention"]
    C1 -->|"SBDR solves this"| S1B["Help customer through\n3-month rough patch"]
    S1B --> R1B["Retain customer\nfor 20+ years"]
    R1B --> R2B["5% retention boost\n= 25-95% more profit"]
    R2B --> R3B["Shift: Recovery at Any Cost\n→ Lifetime Value Optimization"]

    style C1 fill:#d63031,color:#fff
    style S1B fill:#6c5ce7,color:#fff
    style R1B fill:#00b894,color:#fff
    style R2B fill:#00b894,color:#fff
    style R3B fill:#00b894,color:#fff
```

---

## 11. Project Execution Timeline

```mermaid
gantt
    title SBDR Project Timeline
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Phase A - Data Prep
    Collect financial datasets       :a1, 2026-02-10, 7d
    Collect NLP sentiment datasets   :a2, 2026-02-10, 7d
    Merge datasets via Customer ID   :a3, after a1, 5d
    Feature engineering              :a4, after a3, 4d

    section Phase B - Model Training
    Fine-tune FinBERT                :b1, after a4, 7d
    Train BiLSTM                     :b2, after a4, 7d
    Train XGBoost integration        :b3, after b1, 5d
    SHAP and LIME explainability     :b4, after b3, 4d

    section Phase C - Deployment
    Build Streamlit Dashboard        :c1, after b4, 5d
    Fairness audit and bias testing  :c2, after b4, 3d
    Final testing and documentation  :c3, after c1, 4d
```

---

## 12. Complete Architecture — Single View

```mermaid
flowchart TB
    D1B["Lending Club +\nUCI Credit +\nSparkov"]
    D2B["Financial PhraseBank +\nMental Health +\nReddit"]

    D1B --> MERGE2["Clean & Merge\nvia Customer ID"]
    D2B --> MERGE2

    MERGE2 --> FEAT["Feature Engineering\nSpending Volatility +\nSentiment Shift"]

    FEAT --> FINBERT["FinBERT\nDistress Score"]
    FEAT --> BILSTM["BiLSTM\nStress Pattern"]

    FINBERT --> XGBOOST["XGBoost\nCombine All Signals"]
    BILSTM --> XGBOOST
    DEMOG["Demographics +\nCredit History"] --> XGBOOST

    XGBOOST --> SHAPF["SHAP / LIME\nExplainability"]

    SHAPF --> DASHBOARD2["Streamlit Dashboard"]

    DASHBOARD2 --> TIER1["Tier 1\nReminder"]
    DASHBOARD2 --> TIER2["Tier 2\nAssistance"]
    DASHBOARD2 --> TIER3["Tier 3\nRestructuring"]
    DASHBOARD2 --> TIER4["Tier 4\nLegal"]

    SHAPF --> GUARD2["Fairness Audit +\nPII Anonymization +\nRegulatory Compliance"]

    style D1B fill:#dfe6e9,color:#2d3436
    style D2B fill:#a29bfe,color:#fff
    style MERGE2 fill:#ffeaa7,color:#2d3436
    style FEAT fill:#ffeaa7,color:#2d3436
    style FINBERT fill:#6c5ce7,color:#fff
    style BILSTM fill:#e17055,color:#fff
    style DEMOG fill:#dfe6e9,color:#2d3436
    style XGBOOST fill:#00b894,color:#fff
    style SHAPF fill:#fd79a8,color:#fff
    style DASHBOARD2 fill:#0984e3,color:#fff
    style TIER1 fill:#00b894,color:#fff
    style TIER2 fill:#fdcb6e,color:#2d3436
    style TIER3 fill:#e17055,color:#fff
    style TIER4 fill:#d63031,color:#fff
    style GUARD2 fill:#ff7675,color:#fff
```

---

## Color Key

| Color | Meaning |
|-------|---------|
| Grey | Raw Data / Input |
| Yellow | Data Preparation / Feature Engineering |
| Purple | NLP (FinBERT) |
| Orange | Deep Learning (BiLSTM) |
| Green | ML (XGBoost) / Positive Outcomes |
| Pink | Explainability (SHAP/LIME) |
| Blue | Deployment (Streamlit) |
| Red | Risk / Legal / Guardrails |


