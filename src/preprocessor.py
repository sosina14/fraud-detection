"""
preprocessor.py
Handles cleaning, encoding, scaling, and SMOTE resampling.
Each function is intentionally small and single-responsibility
so it can be unit-tested independently.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Columns to drop before modeling — identifiers and raw datetimes
# that have been replaced by engineered features
DROP_COLS = [
    'user_id', 'device_id', 'ip_address', 'ip_int',
    'lower_int', 'upper_int',
    'signup_time', 'purchase_time'
]

CATEGORICAL_COLS = ['source', 'browser', 'sex', 'country']


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows. Justified: duplicates bias model training."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df)} duplicate rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.
    Strategy: drop rows where < 1% of data is affected;
    otherwise median-impute numeric, mode-impute categorical.
    """
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.01].index.tolist()
    low_null  = null_pct[(null_pct > 0) & (null_pct <= 0.01)].index.tolist()

    # Median impute columns with >1% missing
    for col in high_null:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop rows with very few nulls
    df.dropna(subset=low_null, inplace=True)

    print(f"  Imputed: {high_null} | Dropped rows with nulls in: {low_null}")
    return df


def encode_categoricals(df: pd.DataFrame,
                        cols: list = CATEGORICAL_COLS) -> pd.DataFrame:
    """One-hot encode categorical columns. drop_first=True avoids multicollinearity."""
    df = pd.get_dummies(df, columns=[c for c in cols if c in df.columns],
                        drop_first=True)
    return df


def split_features_target(df: pd.DataFrame,
                           target_col: str = 'class',
                           drop_cols: list = DROP_COLS):
    """Separate X and y, dropping identifier columns."""
    drop = [c for c in drop_cols + [target_col] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target_col]
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split then StandardScaler.
    Scaler is fit on train only — prevents data leakage into test set.
    Returns scaled arrays and the fitted scaler (needed for inference).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE oversampling on training set only.
    Chosen over undersampling because the fraud class is already small
    (~9% in Fraud_Data, ~0.17% in creditcard) — undersampling would
    discard too much legitimate transaction data.
    """
    before = pd.Series(y_train).value_counts().to_dict()
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    after = pd.Series(y_res).value_counts().to_dict()

    print(f"  SMOTE: {before} → {after}")
    return X_res, y_res