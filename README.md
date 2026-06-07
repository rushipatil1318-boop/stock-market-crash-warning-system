# Data Directory

This folder contains the datasets used in the Stock Market Crash Early Warning System.

---

## Files

### `master_dataset.csv` — Main dataset (8,178 rows)
The cleaned, merged dataset used for all modeling. Created by joining the three raw sources on date.

**Columns:**
| Column | Type | Description |
|---|---|---|
| Date | datetime | Trading date |
| sp500_open | float | S&P 500 opening price |
| sp500_high | float | S&P 500 daily high |
| sp500_low | float | S&P 500 daily low |
| sp500_close | float | S&P 500 closing price |
| sp500_volume | int | Daily trading volume |
| vix_close | float | VIX closing value |
| sentiment_score | float | Compound sentiment (-1 to +1) |
| crash_label | int | Target: 1 = crash in next 10 days, 0 = no crash |

### `sp500_raw.csv` — Raw S&P 500 data
Original Kaggle dataset, unmodified.

### `vix_raw.csv` — Raw VIX data
Original Kaggle dataset, unmodified.

### `sentiment_raw.csv` — Raw sentiment data
Financial news headlines with VADER-computed sentiment scores.

---

## Data Sources (Kaggle)

All datasets are publicly available on Kaggle:
- S&P 500: [kaggle.com/datasets/camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500)
- VIX: [kaggle.com/datasets/lp187q/vix-index-from-2004-to-2017](https://www.kaggle.com/datasets/lp187q/vix-index-from-2004-to-2017)
- Sentiment: [kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)

---

## Data Cleaning Steps

1. Parsed date columns to uniform `YYYY-MM-DD` format across all three sources
2. Renamed columns to avoid naming conflicts before joining
3. Inner join on date — only dates present in all three sources kept
4. Removed 340 rows with missing values after join (weekends, holidays with partial coverage)
5. Verified no future data in feature columns (temporal integrity check)

---

## Class Distribution

| Label | Count | Percentage |
|---|---|---|
| 0 — No crash | 7,507 | 91.8% |
| 1 — Crash incoming | 671 | 8.2% |

The class imbalance (roughly 11:1) is handled in XGBoost using `scale_pos_weight=11`.
