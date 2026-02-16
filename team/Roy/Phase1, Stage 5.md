# Phase A, Step A5 — Synthetic Chat Generation

**Notebook:** `notebooks/05_synthetic_chats.ipynb`
**Input:** `data/processed/sbdr_featured_dataset.csv` (30K × 85)
**Output:** `data/processed/sbdr_final_dataset.csv` (30K × 88)
**Auxiliary outputs:**
- `data/processed/sbdr_chat_logs.csv` (chat-only file for FinBERT convenience)
- `data/processed/chat_pools_raw.json` (raw template pools backup)

---

## Objective
Generate realistic 3-turn customer support conversations for all 30K customers. Each customer gets a chat log that matches their distress tier in tone and references their actual financial numbers.

---

## Strategy: Tiered Pool Generation + Dynamic Number Injection

Why this approach over alternatives:
- **Per-customer generation (30K API calls):** Overkill — diminishing returns after ~200 unique templates per tier. ~$30+ cost.
- **Pure generic pools:** Weak training signal — "I'm struggling" is less useful than "My bill jumped to $38,000"
- **Winner — Pool + injection:** 800 unique conversation structures × personalized numbers = 30K unique-looking chats. ~$2 cost.

---

## Generation Details

**Model:** GPT-4o (via OpenAI API)
**Batching:** 20 conversations per API call × 10 batches per tier × 4 tiers = 40 API calls
**Temperature:** 1.0 (maximize diversity)
**Total templates:** 800 (200 per tier)
**Runtime:** ~5–8 minutes
**Cost:** ~$2

### Conversation Structure (3 turns per customer)
- **Turn 1 (Customer):** Initial contact — why they're reaching out
- **Turn 2 (Bank Agent):** Response/offer
- **Turn 3 (Customer):** Reaction — acceptance, pushback, silence

### Tier-Specific Tone Design

| Tier | Customer Tone | Topics | Length | Placeholders Used |
|------|--------------|--------|--------|-------------------|
| Low | Calm, polite, routine | Balance checks, card replacement, payment confirmation | 1–2 sentences | LIMIT_BAL, BILL_AMT, PAY_AMT, MONTHLY_SPEND |
| Moderate | Slightly anxious, cooperative | Late payment concern, requesting extension, budget pressure | 1–3 sentences | LIMIT_BAL, BILL_AMT, PAY_AMT, MONTHLY_SPEND, DTI |
| High | Anxious, vulnerable, apologetic | Job loss, medical emergency, divorce, inability to pay | 2–4 sentences | LIMIT_BAL, BILL_AMT, PAY_AMT, MONTHLY_SPEND, DTI, ANNUAL_INC |
| Severe | Defeated, withdrawn, minimal | Total payment stop, ignoring calls, hopelessness | 1–5 words | BILL_AMT, PAY_AMT (sparingly) |

### Generation Results
| Tier | Templates | JSON Errors | Retries | Final Count |
|------|-----------|-------------|---------|-------------|
| Low | 200 | 0 | 0 | 200 |
| Moderate | 200 | 0 | 0 | 200 |
| High | 200 | 0 | 0 | 200 |
| Severe | 200 | 1 (batch 7) | 1 (success) | 200 |

---

## Number Injection

Placeholders replaced with real values from each customer's row:

| Placeholder | Source Column | Format |
|-------------|-------------|--------|
| `{{LIMIT_BAL}}` | LIMIT_BAL | $XXX,XXX |
| `{{BILL_AMT}}` | BILL_AMT1 | $XXX,XXX |
| `{{PAY_AMT}}` | PAY_AMT1 | $XXX,XXX |
| `{{MONTHLY_SPEND}}` | sp_avg_monthly_spend | $XXX,XXX |
| `{{DTI}}` | lc_dti_mean | XX.X% |
| `{{ANNUAL_INC}}` | lc_annual_inc_mean | $XXX,XXX |

---

## Sample Outputs by Tier

**LOW — SBDR_00351:**
> Customer: "Hi, I need a replacement of my credit card that expires soon."
> Agent: "No problem, your new card is scheduled to be sent out before expiration."
> Customer: "Thank you, that's reassuring to know."

**MODERATE — SBDR_16504:**
> Customer: "I missed a payment recently and I'm really feeling it. What can we do?"
> Agent: "We can review your account and adjust payment dates or amounts."
> Customer: "Thanks. I'd like to go over these adjustments with you."

**HIGH — SBDR_15218:**
> Customer: "My partner just lost their job... Managing $110,000 amidst the job loss is really distressing."
> Agent: "I'm really sorry. We can help by offering programs that restructure your payment obligations."
> Customer: "Thank you for understanding. I'm overwhelmed by uncertainty."

**SEVERE — SBDR_15505:**
> Customer: "So many calls... can't handle it."
> Agent: "I'm sorry if it feels overwhelming. We could discuss a plan to pause these demands."
> Customer: "Okay."

---

## Output

**Shape:** 30,000 × 88 columns (85 from A4 + 3 chat columns)
**Nulls:** 0
**New columns:** `chat_turn_1`, `chat_turn_2`, `chat_turn_3`

---

## Phase A — COMPLETE ✅

| Step | Notebook | What | Output |
|------|----------|------|--------|
| A1 ✅ | 01_data_loading.ipynb | UCI load + explore + initial features | uci_credit_processed.csv (30K × 32) |
| A2 ✅ | 02_remaining_datasets.ipynb | Load LC, Sparkov, PhraseBank, Mental Health | 4 processed CSVs |
| A3 ✅ | 03_dataset_merge.ipynb | Merge all via synthetic Customer ID | sbdr_merged_dataset.csv (30K × 57) |
| A4 ✅ | 04_feature_engineering.ipynb | Cross-modal feature engineering | sbdr_featured_dataset.csv (30K × 85) |
| A5 ✅ | 05_synthetic_chats.ipynb | GPT-4o chat generation + number injection | sbdr_final_dataset.csv (30K × 88) |

**Next → Phase B: Model Training**
- B1: Fine-tune FinBERT on chat logs → Distress Score
- B2: Train BiLSTM on payment sequences → Stress Pattern Vector
- B3: Train XGBoost combining all signals → Recovery Tier Prediction
- B4: SHAP + LIME explainability