"""Shared evaluation: every model reports the same metrics on the anomalous class.

Owner: shared. Accuracy is deliberately excluded (predicting all-normal scores ~97%).
"""

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, y_score) -> dict[str, float]:
    """Precision, recall, F1 (anomalous class = 1) and PR-AUC from scores."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
    }
