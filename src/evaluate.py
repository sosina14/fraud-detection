"""
evaluate.py
All evaluation metrics and plots in one place.
Import this in both task2 and task3 notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold


def evaluate(model, X_test, y_test, label="Model") -> dict:
    """Return AUC-PR, F1, and save confusion matrix + PR curve plots."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_proba)
    f1     = f1_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay(cm).plot(ax=axes[0])
    axes[0].set_title(f'{label} — confusion matrix')

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(recall, precision, color='#4C9BE8')
    axes[1].fill_between(recall, precision, alpha=0.1, color='#4C9BE8')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{label} — PR curve (AUC-PR={auc_pr:.3f})')

    plt.tight_layout()
    safe = label.lower().replace(" ", "_")
    plt.savefig(f'../data/processed/{safe}_eval.png', dpi=150)
    plt.show()

    print(f"[{label}] AUC-PR={auc_pr:.4f}  F1={f1:.4f}")
    return {"Model": label, "AUC-PR": round(auc_pr, 4), "F1": round(f1, 4)}


def cross_validate(model, X, y, label="Model", k=5) -> dict:
    """Stratified k-fold CV — reports mean ± std for AUC-PR and F1."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    auc_scores, f1_scores = [], []

    for X_tr, X_val, y_tr, y_val in (
        (X[tr], X[va], y[tr], y[va]) for tr, va in skf.split(X, y)
    ):
        model.fit(X_tr, y_tr)
        auc_scores.append(average_precision_score(y_val, model.predict_proba(X_val)[:,1]))
        f1_scores.append(f1_score(y_val, model.predict(X_val)))

    print(f"[{label}] CV AUC-PR: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"[{label}] CV F1    : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    return {
        "Model": label,
        "CV AUC-PR mean": round(np.mean(auc_scores), 4),
        "CV AUC-PR std":  round(np.std(auc_scores), 4),
        "CV F1 mean":     round(np.mean(f1_scores), 4),
        "CV F1 std":      round(np.std(f1_scores), 4),
    }


def comparison_table(results: list) -> pd.DataFrame:
    """Pretty-print a side-by-side comparison DataFrame."""
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    return df