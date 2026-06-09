"""
data_loader.py
Handles loading raw CSVs and converting IP addresses to integers
for range-based geolocation merging.
"""

import pandas as pd
import numpy as np


def load_fraud_data(path: str) -> pd.DataFrame:
    """Load and do minimal type-fixing on Fraud_Data.csv."""
    df = pd.read_csv(path)
    df['signup_time']   = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df


def load_ip_data(path: str) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv and convert bounds to integers."""
    df = pd.read_csv(path)
    df['lower_int'] = df['lower_bound_ip_address'].apply(ip_to_int)
    df['upper_int'] = df['upper_bound_ip_address'].apply(ip_to_int)
    return df


def load_creditcard(path: str) -> pd.DataFrame:
    """Load creditcard.csv — already PCA-transformed, minimal processing."""
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def ip_to_int(ip) -> int:
    """
    Convert a dotted-decimal IP string to a 32-bit integer.
    Example: '192.168.1.1' -> 3232235777
    Used so we can do a numeric range lookup instead of string matching.
    """
    try:
        parts = str(ip).split('.')
        return sum(int(p) << (8 * (3 - i)) for i, p in enumerate(parts))
    except Exception:
        return 0


def merge_ip_country(fraud_df: pd.DataFrame,
                     ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based merge: assign country to each transaction by checking
    whether its IP integer falls within [lower_int, upper_int].
    Uses merge_asof (sorted merge) for efficiency — O(n log n) instead
    of a slow row-by-row loop.
    """
    fraud_df = fraud_df.copy()
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

    fraud_sorted = fraud_df.sort_values('ip_int')
    ip_sorted    = ip_df.sort_values('lower_int')

    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted[['lower_int', 'upper_int', 'country']],
        left_on='ip_int',
        right_on='lower_int',
        direction='backward'
    )

    # Invalidate rows where ip_int overshoots the matched upper bound
    merged.loc[merged['ip_int'] > merged['upper_int'], 'country'] = 'Unknown'
    merged['country'] = merged['country'].fillna('Unknown')
    return merged