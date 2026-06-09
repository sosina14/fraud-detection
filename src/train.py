"""
train.py
Model training functions — kept separate from evaluation
so each can be imported and tested independently.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

os.makedirs('../models', exist_ok=True)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Logistic Regression baseline.
    class_weight='balanced' adjusts for imbalance without needing SMOTE
    — used here as a sanity check alongside SMOTE results.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train,
                  n_estimators=300,
                  max_depth=6,
                  learning_rate=0.05) -> XGBClassifier:
    """
    XGBoost ensemble model.
    scale_pos_weight = neg/pos ratio tells XGBoost to penalise
    missing a fraud case more heavily than a false alarm.
    """
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    spw = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"  Saved: {path}")


def load_model(path: str):
    return joblib.load(path)