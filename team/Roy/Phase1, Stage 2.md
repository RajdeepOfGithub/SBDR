---
# Step A2 Complete — All Datasets Loaded

| # | Dataset | Raw Size | Processed Size | Key Info |
|---|---------|----------|----------------|----------|
| 1 | UCI Credit Card | 30K × 25 | 30K × 32 | ✅ Done in NB01 |
| 2 | Lending Club | 2.26M × 151 | 2.26M × 19 | Polars loaded, SBDR tiers mapped |
| 3 | Sparkov Synthetic | 1.85M × 23 | 1.85M × 9 | 999 customers, 2 years of transactions |
| 4 | Financial PhraseBank | 2,264 × 2 | 2,264 × 2 | neutral/positive/negative labels |
| 5 | Mental Health Sentiment | 53K × 3 | 38K × 2 | Filtered to Anxiety, Stress, Depression, Normal |

### Saved to `data/processed/`:
- `uci_credit_processed.csv` (NB01)
- `lending_club_processed.csv`
- `sparkov_processed.csv`
- `financial_phrasebank_processed.csv`
- `mental_health_processed.csv`

### Next Steps:
- **A3:** Merge datasets via synthetic Customer IDs
- **A4:** Full feature engineering on merged data
- **A5:** Generate synthetic customer chats (GPT-4o)