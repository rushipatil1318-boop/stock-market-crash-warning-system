# Stock Market Crash Early Warning System

Predicts whether the S&P 500 will fall more than 5% below today's close within the next 10 trading days, using market prices, the VIX volatility index and financial-news sentiment. Built in KNIME with Python script nodes, an XGBoost model and a Keras LSTM, combined as an ensemble. Coursework for Machine Learning & AI Techniques at ISM Dortmund (June 2026).

## Approach

- **Data:** S&P 500 daily prices, CBOE VIX and VADER news-sentiment scores from Kaggle, merged on date (8,178 rows, about 2002–2022). About 8.2% of days are labelled as crash events.
- **Features:** 24 features from past data only: price lags, rolling returns, 5/20-day moving averages and their crossover, rolling volatility, VIX lags, a VIX-spike flag and sentiment lags and trends. See [src/feature_engineering.py](src/feature_engineering.py).
- **Split:** chronological. Training data is everything before 2020; 2020–2022 (including the COVID crash) is the test set.
- **Models:** XGBoost (150 trees, depth 5, class weighting for the imbalance) and an LSTM over 10-day windows (64 units, dropout 0.2). The ensemble averages their probabilities (60% XGBoost, 40% LSTM).

## Results (test set 2020–2022)

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.81 | 0.61 | 0.68 | 0.64 |
| Keras LSTM | 0.74 | 0.54 | 0.72 | 0.62 |
| Ensemble | 0.78 | 0.63 | 0.71 | 0.67 |

Recall is prioritised, since missing a crash costs more than a false alarm. Earlier versions reached an AUC of about 0.91, which turned out to be data leakage (future VIX values in the features and a random instead of chronological split). The numbers above are after fixing both.

## Repository contents

```
src/feature_engineering.py   feature code from the KNIME Python Script node
docs/project_report.md       full report: data, labels, method, results, limitations
docs/workflow_notes.md       node-by-node notes on the KNIME workflow
docs/model_summary.txt       metrics and feature importance
data/README.md               description of the datasets and cleaning steps
```

The CSV datasets and the KNIME workflow file are not included in this repository.

## Limitations and next steps

VADER is a general sentiment tool, not a financial one (FinBERT would fit better). Class imbalance was handled only with class weights. The model works on historical data only and has no macroeconomic features such as interest rates or the yield curve. Next steps: FinBERT sentiment, macro indicators, threshold tuning and a small dashboard for the daily crash probability.
