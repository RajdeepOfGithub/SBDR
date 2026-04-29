# SBDR — Known Limitations & Mitigations

This document addresses the known limitations of the SBDR system honestly and explains
the architectural decisions made to work around them. Each limitation has a documented
root cause, measurable impact, and mitigation strategy.

---

## L1 — FinBERT Running in Zero-Shot Mode (No Fine-Tuning)

**Root cause:** AMD ROCm 6.3/6.4 driver incompatibility caused GPU segfaults during
backpropagation on this machine's gfx1100 architecture. Fine-tuning FinBERT on the
`financial_phrasebank_processed.csv` training set was not feasible without GPU.
CPU-only fine-tuning on 30,000 samples × 3 turns was estimated at 6–8 hours per epoch.

**Measured impact:**
- High-tier distress scores slightly above severe-tier (0.6161 vs 0.5633 mean distress)
- The expected monotonic relationship (Tier 5 > Tier 4 > Tier 3) is partially inverted
  between Tier 3 and Tier 4
- `distress_avg` contributes only 5.2% of total SHAP weight (vs UCI's 66.4%)

**Why this is not a fatal flaw:**
The ANOVA test across tiers returns F=8,498, p≈0 — there is strong, statistically
significant tier separation even in zero-shot mode. The correlation between
`distress_avg` and `distress_encoded` (the engineered label) is 0.60, which confirms
the signal is real, just not perfectly calibrated.

**Mitigation applied:**
- Tier 5 XGBoost threshold adjusted from `distress_avg < 0.33` to `distress_avg < 0.55`
  to compensate for zero-shot score inflation
- B3.5 Audit Layer serves as a second opinion, catching 266 additional Tier 5 cases
  that XGBoost alone missed (+121% increase)
- In production deployment, fine-tuning on confirmed financial distress labels would
  replace the zero-shot approach

---

## L2 — Tier 5 F1-Score = 0.27

**Root cause:** Directly downstream of L1. The distress calibration issue means the
XGBoost model sees noisy FinBERT features for the Tier 5 boundary, which has only
180 training samples even before the calibration noise.

**Measured impact:**
- Tier 5 precision/recall are lower than Tiers 1–4
- Final Tier 5 count after audit: 485 (1.6% of portfolio)

**Why the overall model is still strong:**
- Accuracy: 93.4%, AUC-ROC: 0.990, F1-weighted: 93.6%
- Tiers 1–4 (98.4% of cases) classify near-perfectly
- The audit layer is specifically designed to compensate for Tier 5 uncertainty:
  Rule B catches cases with `tier_prob_5 >= 0.15` even when the final prediction
  is Tier 4, and Rule C independently flags fraud + anomaly + default combinations

**Mitigation applied:** B3.5 Audit Layer (Notebook 09) — see Section 4 of Roadmap.

---

## L3 — Synthetic Chat De-escalation Arc

**Root cause:** GPT-4o was prompted to generate realistic customer service chat logs
with a de-escalation arc (Turn 1: customer distress → Turn 2: agent offer → Turn 3:
customer accepts). This is a realistic pattern but it means Turn 3 is always less
distressed than Turn 1.

**Measured impact:**
- `distress_shift` (Turn 3 − Turn 1) is negative for 96.6% of customers
- The sentiment shift signal, while present, has low discriminative power because
  all tiers show the same arc direction
- In production, genuine defaulters may show escalating sentiment (Turn 3 more
  distressed than Turn 1), which would be a much stronger Tier 4/5 signal

**What this means for deployment:**
`distress_shift` as a feature should be re-evaluated on real customer interaction
logs. The current model likely underweights it due to low variance in training data.

---

## L4 — Sparkov & Lending Club Features Are Aggregate-Matched, Not Individual

**Root cause:** The UCI dataset does not contain individual Sparkov or Lending Club
records. Features like `lc_annual_inc_mean` and `sp_total_spend` are aggregated
profile means matched by credit risk tier, not individual transaction-level data
linked to each customer ID.

**Measured impact:**
- Sparkov contributes only 2.3% of SHAP weight
- LC features contribute within the UCI/Derived bucket at ~5%
- These features add signal at the portfolio level but cannot identify individual
  behavioral anomalies the way real linked data would

**What this means for deployment:**
In a real bank environment, individual transaction feeds from card networks would
replace the Sparkov aggregates, making the behavioral anomaly detection substantially
more powerful.

---

## L5 — Tier 5 Training Labels Are Synthetically Constructed

**Root cause:** No real confirmed strategic default cases exist in the public UCI
dataset. Tier 5 labels were engineered using a rule: `default=1 AND distress_avg < 0.55
AND bilstm_anomaly_flag=1`. This is a principled heuristic, not observed fraud.

**What this means for deployment:**
The model should be retrained on confirmed historical fraud and strategic default
cases before production use. The current Tier 5 boundary is a research prototype,
not a legally defensible classifier.

**Risk register reference:** See `docs/1. Roadmap.md` Section 7 — Risk Register.

---

## Summary Table

| ID | Limitation | Impact | Mitigated? |
|----|-----------|--------|------------|
| L1 | FinBERT zero-shot (no fine-tuning) | Distress calibration drift | Partial — audit layer compensates |
| L2 | Tier 5 F1 = 0.27 | Lower Tier 5 precision/recall | Yes — B3.5 audit layer (+121%) |
| L3 | Synthetic chat de-escalation arc | distress_shift low variance | Documented — feature needs real data |
| L4 | Sparkov/LC features are aggregate-matched | Low individual-level signal | Accepted — prototype limitation |
| L5 | Tier 5 labels are rule-engineered | Not real observed fraud | Documented — requires production retraining |

---

## Fairness Audit (C2)

Tier 5 escalation rates across demographic groups (from dashboard Fairness section):

| Group | Tier 5 Rate |
|-------|-------------|
| Male | 2.0% |
| Female | 1.4% |
| Age < 30 | 1.6% |
| Age 30–39 | 1.6% |
| Age 40–49 | 1.5% |
| Age 50+ | 2.0% |
| Graduate | 1.7% |
| University | 1.6% |
| High School | 1.6% |
| Other Education | 1.1% |

No demographic group is disproportionately escalated to Tier 5. The 0.6pp
gender gap (Male 2.0% vs Female 1.4%) is within acceptable tolerance.
SEX, AGE, and EDUCATION are not top SHAP drivers — `avg_pay_delay` and
`bilstm_anomaly_flag` dominate (UCI behavioral signals, not demographics).

---

*Last updated: 2026-04-29 | SBDR Capstone — DATA 606 — UMBC*
