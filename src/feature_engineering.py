"""
feature_engineering.py
All feature engineering for Fraud_Data.
Kept separate from preprocessor.py so domain-specific logic
is easy to audit and extend without touching cleaning code.
"""

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from purchase_time.
    hour_of_day captures time-of-day fraud patterns (e.g. 2–4am spikes).
    day_of_week captures weekend vs weekday differences.
    """
    df = df.copy()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df


def add_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hours between account creation and first purchase.
    Key fraud signal: fraudsters often purchase within minutes of signing up
    before the account gets flagged. Negative values are data quality issues
    and are clipped to 0.
    """
    df = df.copy()
    diff = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['time_since_signup'] = diff.clip(lower=0)
    return df


def add_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions per user (overall frequency) and a rolling
    24-hour window count.
    High velocity = potential card testing or account takeover.
    """
    df = df.copy().sort_values(['user_id', 'purchase_time'])

    # Overall frequency per user
    df['user_txn_freq'] = df.groupby('user_id')['user_id'].transform('count')

    # Rolling 24h count — number of prior txns by same user in last 24h
    def rolling_24h(group):
        times = group['purchase_time']
        counts = []
        for i, t in enumerate(times):
            window_start = t - pd.Timedelta(hours=24)
            count = ((times[:i] >= window_start) & (times[:i] < t)).sum()
            counts.append(count)
        group['txn_count_24h'] = counts
        return group

    df = df.groupby('user_id', group_keys=False).apply(rolling_24h)
    return df


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = add_time_features(df)
    df = add_time_since_signup(df)
    df = add_transaction_velocity(df)
    return df