"""Supervised: class-weighted XGBoost on event-count vectors.

Input is the (n_sessions, n_events) count matrix from EventCountVectorizer and the 0/1
labels. Counts are used as they are: trees split on thresholds, so the log(1 + x) scaling
that logistic regression needs would not change which sessions end up together.
scale_pos_weight (negatives / positives in the training split) makes the rare anomalies
count more, the same weighting as logistic regression's class_weight="balanced".
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
LEARNING_RATE = 0.1
GRID = tuple(
    {"max_depth": depth, "n_estimators": trees}
    for depth in (3, 6, 9)
    for trees in (100, 300)
)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 6,
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Fit a class-weighted gradient-boosted tree classifier on the training split."""
    positives = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=(len(y_train) - positives) / positives,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    return model.fit(X_train, y_train)


def scores(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return the probability that each session is anomalous (higher = more anomalous)."""
    return model.predict_proba(X)[:, 1]


def tune(X_train, y_train, X_val, y_val, grid=GRID) -> tuple[XGBClassifier, list[dict]]:
    """Train one model per grid entry and keep the one with the best validation PR-AUC.

    PR-AUC is threshold-free, so the tree settings are chosen independently of the F1 cutoff
    picked later. Returns the best model and one row per candidate:
    {"max_depth", "n_estimators", "val_pr_auc"}.
    """
    table, best_model, best_pr_auc = [], None, -1.0
    for config in grid:
        model = train(X_train, y_train, **config)
        pr_auc = average_precision_score(y_val, scores(model, X_val))
        table.append({**config, "val_pr_auc": float(pr_auc)})
        if pr_auc > best_pr_auc:
            best_model, best_pr_auc = model, pr_auc
    return best_model, table


def feature_importances(model: XGBClassifier, feature_names: list[str]) -> pd.Series:
    """Return each event's importance (share of total split gain), largest first."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
