"""
Feature Engineering Script
============================
Stock Market Crash Early Warning System
Author: Rushikesh Patil
MSc Marketing Management, ISM Dortmund

This script runs inside the KNIME Python Script Node.
Input: Merged dataframe with sp500, vix, and sentiment columns
Output: Feature-engineered dataframe ready for model training

Note: This is extracted from the KNIME workflow for readability.
In KNIME, this runs as an embedded Python node using the
KNIME Python Integration extension.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the merged raw dataframe and creates all ML features.
    
    Key design decisions:
    - All features use only PAST data (lag/rolling backward)
    - Target label uses future data (shift forward) -- this is intentional
    - No future data leaks into features
    """

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # -------------------------------------------------------
    # 1. Price-based Lag Features
    # -------------------------------------------------------
    df['sp500_lag1'] = df['sp500_close'].shift(1)   # yesterday
    df['sp500_lag3'] = df['sp500_close'].shift(3)   # 3 days ago
    df['sp500_lag5'] = df['sp500_close'].shift(5)   # last week

    # Daily return
    df['sp500_return1d'] = df['sp500_close'].pct_change(1)
    df['sp500_return5d'] = df['sp500_close'].pct_change(5)

    # -------------------------------------------------------
    # 2. VIX (Volatility Index) Features
    # -------------------------------------------------------
    df['vix_lag1'] = df['vix_close'].shift(1)
    df['vix_lag3'] = df['vix_close'].shift(3)
    df['vix_ma5']  = df['vix_close'].rolling(window=5).mean()

    # VIX spike indicator -- sudden jump > 20% in VIX is a known warning signal
    df['vix_spike'] = (df['vix_close'].pct_change(1) > 0.20).astype(int)

    # -------------------------------------------------------
    # 3. Moving Averages
    # -------------------------------------------------------
    df['sp500_ma5']  = df['sp500_close'].rolling(window=5).mean()
    df['sp500_ma20'] = df['sp500_close'].rolling(window=20).mean()

    # Golden cross / death cross signal
    # Negative = short-term MA below long-term MA = bearish signal
    df['ma_crossover'] = df['sp500_ma5'] - df['sp500_ma20']

    # -------------------------------------------------------
    # 4. Volatility Features
    # -------------------------------------------------------
    df['sp500_vol5']  = df['sp500_close'].rolling(window=5).std()
    df['sp500_vol10'] = df['sp500_close'].rolling(window=10).std()

    # -------------------------------------------------------
    # 5. Sentiment Features
    # -------------------------------------------------------
    df['sentiment_lag1'] = df['sentiment_score'].shift(1)
    df['sentiment_ma3']  = df['sentiment_score'].rolling(window=3).mean()

    # Sentiment deterioration -- negative shift over 5 days
    df['sentiment_trend5'] = df['sentiment_score'].diff(5)

    # -------------------------------------------------------
    # 6. Target Label Creation
    # -------------------------------------------------------
    # Crash = S&P 500 drops more than 5% within the next 10 trading days
    # We look at the MINIMUM price in the next 10 days
    df['future_min_10d'] = (
        df['sp500_close']
        .shift(-1)                        # start from tomorrow
        .rolling(window=10, min_periods=1)
        .min()
        .shift(-(10-1))                   # align back
    )

    df['forward_pct_drop'] = (
        (df['future_min_10d'] - df['sp500_close']) / df['sp500_close']
    )

    # Binary label: 1 = crash incoming, 0 = no crash
    df['crash_label'] = (df['forward_pct_drop'] < -0.05).astype(int)

    print(f"Crash events in dataset: {df['crash_label'].sum()} "
          f"({df['crash_label'].mean()*100:.1f}% of rows)")

    # -------------------------------------------------------
    # 7. Drop rows with NaN from rolling/shift operations
    # -------------------------------------------------------
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after} rows with NaN values")
    print(f"Final dataset: {rows_after} rows")

    return df


def temporal_split(df: pd.DataFrame, cutoff_date: str = '2020-01-01'):
    """
    Splits data chronologically -- NOT randomly.
    
    Why: Random splitting in time series causes data leakage.
    Post-2020 patterns would leak into training, inflating scores.
    
    cutoff_date: everything before this = train, after = test
    """
    cutoff = pd.to_datetime(cutoff_date)
    train = df[df['Date'] < cutoff].copy()
    test  = df[df['Date'] >= cutoff].copy()

    print(f"Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"Test:  {len(test)} rows  ({test['Date'].min().date()} to {test['Date'].max().date()})")
    print(f"Train crash rate: {train['crash_label'].mean()*100:.1f}%")
    print(f"Test crash rate:  {test['crash_label'].mean()*100:.1f}%")

    return train, test


if __name__ == "__main__":
    # For testing outside KNIME
    # Load your merged CSV here
    df_raw = pd.read_csv('../data/master_dataset.csv')
    df_features = engineer_features(df_raw)
    train_df, test_df = temporal_split(df_features)

    df_features.to_csv('../data/features_engineered.csv', index=False)
    print("Feature engineering complete. File saved.")
