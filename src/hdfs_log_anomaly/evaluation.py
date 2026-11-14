"""Shared evaluation: every model reports the same metrics on the anomalous class.

Accuracy is deliberately excluded (predicting all-normal scores ~97%).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
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


def best_f1_threshold(y_true, y_score) -> float:
    """Return the cutoff on y_score that maximises F1 on the anomalous class.

    Predict anomalous when score >= cutoff. Call this on the validation split only, then
    apply the cutoff once to the test split, so every model is thresholded the same way.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1)])
