"""Supervised baseline: class-weighted Logistic Regression on log-scaled count vectors.

Input is the (n_sessions, n_events) count matrix from EventCountVectorizer and the 0/1
labels. Counts are compressed with log(1 + x) first so large counts (e.g. 203 reads of one
block) don't dominate. class_weight="balanced" makes the rare anomalies count more.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def train(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> Pipeline:
    """Fit log1p scaling + class-weighted Logistic Regression on the training split."""
    model = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p)),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000)),
        ]
    )
    return model.fit(X_train, y_train)


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return the probability that each session is anomalous (higher = more anomalous)."""
    return model.predict_proba(X)[:, 1]


def tune_C(X_train, y_train, X_val, y_val, Cs=C_GRID) -> tuple[Pipeline, list[dict]]:
    """Train one model per C and keep the one with the best validation PR-AUC.

    PR-AUC is threshold-free, so C is chosen independently of the F1 cutoff picked later.
    Returns the best model and one {"C", "val_pr_auc"} row per candidate.
    """
    table, best_model, best_pr_auc = [], None, -1.0
    for C in Cs:
        model = train(X_train, y_train, C=C)
        pr_auc = average_precision_score(y_val, scores(model, X_val))
        table.append({"C": C, "val_pr_auc": float(pr_auc)})
        if pr_auc > best_pr_auc:
            best_model, best_pr_auc = model, pr_auc
    return best_model, table


def coefficients(model: Pipeline, feature_names: list[str]) -> pd.Series:
    """Return each event's coefficient, largest first (positive = pushes toward anomalous)."""
    weights = pd.Series(model.named_steps["clf"].coef_[0], index=feature_names)
    return weights.sort_values(ascending=False)
