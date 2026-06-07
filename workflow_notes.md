# KNIME Workflow Notes — Node by Node

This document explains what each major node in `crash_warning_v3.knwf` does, and why I made certain design choices.

---

## Stage 1 — Data Ingestion

### Excel Reader (×3)
Three separate reader nodes pull in the raw datasets:
- `sp500_raw.csv` — daily S&P 500 open, high, low, close, volume
- `vix_raw.csv` — daily VIX closing values
- `sentiment_raw.csv` — date + compound sentiment score (from NLP processing of financial headlines)

**Design choice:** I kept the three datasets separate until the Joiner stage deliberately — merging too early would make it harder to inspect each source for issues independently.

---

## Stage 2 — Joining & Cleaning

### Joiner Node
Merges all three datasets on the `Date` column using an inner join. This means only dates present in all three sources are kept — which is why the final dataset is 8,178 rows rather than the full ~10,000 from sentiment.

**Problem I ran into:** VIX data uses `DATE` (uppercase) while S&P 500 uses `Date` — KNIME's joiner is case-sensitive by default. Fixed by adding a Column Rename node before the join.

### Missing Value Handler
After joining, about 340 rows had NaN values — mostly from weekends and holidays where sentiment scores existed but market data didn't. Used "Remove Row" strategy rather than imputation, since financial time series imputation can introduce artificial patterns.

---

## Stage 3 — Feature Engineering (Python Script Node)

This is the most important stage. The Python node creates the features the models actually learn from.

```python
import pandas as pd
import numpy as np

def create_features(df):
    # Lag features — yesterday's and last week's values
    df['sp500_lag1'] = df['sp500_close'].shift(1)
    df['sp500_lag5'] = df['sp500_close'].shift(5)
    df['vix_lag1'] = df['vix_close'].shift(1)
    
    # Rolling averages — 5-day and 20-day
    df['sp500_ma5'] = df['sp500_close'].rolling(window=5).mean()
    df['sp500_ma20'] = df['sp500_close'].rolling(window=20).mean()
    df['vix_ma5'] = df['vix_close'].rolling(window=5).mean()
    
    # Volatility — 10-day rolling standard deviation
    df['sp500_vol10'] = df['sp500_close'].rolling(window=10).std()
    
    # Sentiment momentum — 3-day rolling average of sentiment
    df['sentiment_ma3'] = df['sentiment_score'].rolling(window=3).mean()
    df['sentiment_lag1'] = df['sentiment_score'].shift(1)
    
    # Target label — crash = drop > 5% in next 10 days
    df['future_min'] = df['sp500_close'].shift(-10).rolling(window=10).min()
    df['pct_drop'] = (df['future_min'] - df['sp500_close']) / df['sp500_close']
    df['crash_label'] = (df['pct_drop'] < -0.05).astype(int)
    
    # Drop rows with NaN from rolling/shift operations
    df = df.dropna()
    
    return df
```

**Important note on the target label:** The `shift(-10)` looks 10 days into the future — this is intentional for the label only. The features only use past data (lag/rolling backwards), so there is no leakage. This was a mistake in early versions (v1, v2) where I accidentally used future VIX values as features.

---

## Stage 4 — Train/Test Split

### Row Splitter Node
Split is done **chronologically, not randomly:**
- Training set: all data before January 1, 2020
- Test set: 2020 onwards (includes COVID crash — a real-world stress test)

**Why chronological?** Random splitting in time series leaks future information into training. The model would "see" 2020 data patterns during training and then predict 2020 — which is cheating. Chronological split is the correct approach for financial data.

---

## Stage 5 — Normalization

### Normalizer Node
Min-max normalization applied to all numeric features before feeding into both models. The normalization parameters are computed on training data only and then applied to test data — another place where data leakage is easy to accidentally introduce.

---

## Stage 6 — XGBoost Branch

### XGBoost Learner
Settings used:
- Number of trees: 150
- Max depth: 5
- Learning rate: 0.05
- Subsample: 0.8
- Objective: binary:logistic

### XGBoost Predictor
Outputs both predicted class and probability score. The probability column (`P(crash=1)`) is what gets passed to the ensemble combiner.

---

## Stage 7 — Keras LSTM Branch

### Keras Network Creator → LSTM Learner
The LSTM takes a 10-day sequence window as input — so each prediction is based on the last 10 days of feature values, treating it as a sequential pattern problem.

Architecture:
- Input: (10 timesteps × 10 features)
- LSTM layer: 64 units
- Dropout: 0.2
- Dense output: 1 unit, sigmoid activation

Training: 30 epochs, batch size 32, Adam optimizer.

**Honest note:** The LSTM was the trickiest part to get working in KNIME — the DL4J integration requires specific tensor shapes. Spent about 2 sessions debugging the reshape node before it worked correctly.

---

## Stage 8 — Ensemble Combiner (Python Script Node)

Simple weighted average of the two model probabilities:

```python
# Ensemble: 60% XGBoost, 40% LSTM
# XGBoost weighted higher because it performed better on tabular features

df['ensemble_prob'] = (0.6 * df['xgb_prob']) + (0.4 * df['lstm_prob'])
df['ensemble_prediction'] = (df['ensemble_prob'] > 0.5).astype(int)
```

The 60/40 weighting was chosen based on validation set performance — XGBoost consistently outperformed LSTM on the tabular feature set, so it gets higher weight.

---

## Stage 9 — Evaluation

### ROC Curve Node
Plots ROC curve and computes AUC for all three models (XGBoost alone, LSTM alone, ensemble).

### Scorer Node (Confusion Matrix)
Shows true positives, false positives, false negatives for crash detection — for this use case, **false negatives** (missing a real crash) are more costly than false positives, so recall is prioritized over precision.

---

## Known Limitations

1. **Sentiment data quality** — the Kaggle sentiment dataset uses simple VADER scores, not domain-specific financial NLP. A proper deployment would use FinBERT or similar.
2. **Class imbalance** — crashes are rare events (~8% of dataset). SMOTE or class weighting would help, not yet implemented.
3. **No real-time data** — currently works only on historical static data. Live deployment would need API integration.

These are documented honestly because any technical interviewer will ask about limitations — better to show you're aware of them.
