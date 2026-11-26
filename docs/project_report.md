# Project Report — Stock Market Crash Early Warning System

**Author:** Rushikesh Patil  
**Course:** Machine Learning & AI Techniques (MLAIT)  
**Institution:** ISM International School of Management, Dortmund  
**Date:** June 2026

---

## 1. Problem Statement

Stock market crashes cause significant economic damage — the 2008 financial crisis wiped out roughly $10 trillion in household wealth in the US alone, and the 2020 COVID crash saw the S&P 500 lose 34% in just 33 days. Despite this, most retail investors and even many institutional ones receive crash signals too late to act meaningfully.

The goal of this project was to build an early warning system that flags crash risk *before* it materializes, using a combination of market data and news sentiment — trained and evaluated on historical data.

---

## 2. Data Sources

Three publicly available Kaggle datasets were used:

**S&P 500 Historical Data**
- Daily OHLCV (Open, High, Low, Close, Volume) prices
- Approximately 5,000 trading days of data
- Used as the primary price signal and for target label creation

**CBOE VIX Index**
- Daily closing values of the CBOE Volatility Index
- VIX is often called the "fear gauge" — spikes in VIX historically precede market downturns
- Provides the volatility dimension of the feature set

**Financial News Sentiment**
- NLP-processed sentiment scores from financial news headlines
- Compound score ranging from -1 (very negative) to +1 (very positive)
- Captures the "market mood" dimension beyond raw price data

After merging on date and removing rows where any dataset had missing values, the final master dataset contained **8,178 rows** spanning approximately 2002 to 2022.

---

## 3. Target Label Definition

A "crash event" was defined as:
> The S&P 500 closing price drops more than **5%** below today's price at any point within the next **10 trading days**.

This definition was chosen because:
- 5% is a meaningful threshold — intraday noise won't trigger it
- 10 days gives enough lookahead to be actionable
- It aligns with academic literature on short-term drawdown prediction

Approximately **8.2%** of the dataset rows were labeled as crash events — a meaningful class imbalance that affects model training.

---

## 4. Feature Engineering

24 features were created from the raw data, grouped into four categories:

**Price momentum features**
Lag values (1-day, 3-day, 5-day), rolling returns, and moving averages (5-day, 20-day). The MA crossover (5-day minus 20-day) captures the golden/death cross signal used widely in technical analysis.

**Volatility features**
Rolling standard deviation of S&P 500 over 5-day and 10-day windows. High recent volatility tends to precede further volatility — this is the "volatility clustering" effect.

**VIX-based features**
Raw VIX lag values, 5-day VIX moving average, and a binary VIX spike indicator (1 if VIX jumped more than 20% in a single day). VIX spikes above 30 have historically been strong crash precursors.

**Sentiment features**
1-day lagged sentiment, 3-day rolling average sentiment, and 5-day sentiment trend (change in sentiment over the past week). Deteriorating sentiment often leads price drops by 1-3 days.

All features use only historical data — no future values appear in the feature set.

---

## 5. Methodology

### 5.1 Train-Test Split

A chronological split was used — not a random split. All data before January 1, 2020 was used for training; 2020 onwards was held out as the test set.

This is critical for time series data. Random splitting causes temporal leakage — the model "sees" patterns from 2020 during training and then predicts 2020 data, which artificially inflates evaluation scores. The chronological split ensures the model is genuinely predicting unseen future events.

### 5.2 XGBoost Classifier

XGBoost was chosen as the primary model because it handles tabular data well and provides feature importance scores, which are useful for interpretability.

Hyperparameters: 150 trees, max depth 5, learning rate 0.05, subsample 0.8. Scale-pos-weight was set to approximately 11 (ratio of negative to positive class) to handle the class imbalance.

### 5.3 Keras LSTM

An LSTM network was added to capture sequential dependencies in the time series. Each input was a 10-day window of feature values, treating crash prediction as a sequence classification problem.

Architecture: LSTM (64 units) → Dropout (0.2) → Dense (1, sigmoid). Trained for 30 epochs with Adam optimizer.

### 5.4 Ensemble Combination

The final prediction combines both models using a weighted average of their output probabilities:

```
ensemble_probability = (0.6 × XGBoost_prob) + (0.4 × LSTM_prob)
```

XGBoost received higher weight because it consistently outperformed the LSTM on the tabular feature set during validation.

---

## 6. Results

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.81 | 0.61 | 0.68 | 0.64 |
| Keras LSTM | 0.74 | 0.54 | 0.72 | 0.62 |
| **Ensemble** | **0.78** | **0.63** | **0.71** | **0.67** |

For crash detection, **recall is prioritized** over precision — it is better to raise a false alarm than to miss a real crash event.

### A note on earlier results

Early versions of this project showed AUC scores of ~0.91. This was traced to data leakage — specifically, VIX values from the future window were accidentally included as features in v1, and v2 used a random split rather than a chronological one. After correcting both issues, the honest AUC settled at 0.78, which is what is reported here.

Documenting this is intentional — understanding *why* a model performs the way it does is more valuable than reporting inflated numbers.

---

## 7. Limitations

**Sentiment data quality.** The Kaggle sentiment dataset uses VADER, a general-purpose sentiment tool not trained on financial language. Domain-specific models like FinBERT would likely improve sentiment features significantly.

**Class imbalance.** Crashes are rare (~8% of data). While scale-pos-weight in XGBoost addresses this partially, techniques like SMOTE or threshold tuning were not fully explored.

**Static dataset.** The system currently works only on historical data. A production deployment would require live data feeds from financial APIs (Alpha Vantage, Polygon.io, etc.) and retraining pipelines.

**No macroeconomic features.** Interest rate changes, Fed announcements, and yield curve inversions are known crash predictors but were not included due to data availability constraints.

---

## 8. Possible Extensions

- Replace VADER with FinBERT for better financial sentiment
- Add macroeconomic indicators (Fed Funds Rate, yield spread)
- Implement online learning to update the model as new data arrives
- Build a simple dashboard (Plotly/Dash) to visualize crash probability in real time
- Explore Transformer-based models for the sequential component

---

## 9. Conclusion

This project demonstrated that combining structured market data (price, volatility) with unstructured sentiment signals can produce a meaningful early warning system for stock market crashes, achieving an AUC of 0.78 on a held-out post-2020 test set that includes the COVID crash.

The most important technical lesson was around temporal data integrity — the difference between a correctly implemented time series pipeline and a leaky one can be the difference between a reported AUC of 0.91 and a realistic one of 0.78. Getting this right matters more than chasing high numbers.

---

*This project was completed as part of the MLAIT coursework at ISM International School of Management, Dortmund, June 2026.*
