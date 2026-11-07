"""Unsupervised: Isolation Forest on log-scaled count vectors.

Labels are never passed to fit(). They are used only later, on the validation split, to
pick an F1 cutoff (see evaluation.best_f1_threshold). Input is the same (n_sessions,
n_events) count matrix from EventCountVectorizer as the supervised models.

Isolation Forest recursively partitions the feature space. Points that fall into short
paths (easy to isolate) are treated as anomalies. contamination is left at "auto"
because we do not use sklearn's built-in predict() labels; we rank sessions with
scores() and threshold them ourselves.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

RANDOM_STATE = 42
N_ESTIMATORS = 100


def train(X_train: np.ndarray, random_state: int = RANDOM_STATE) -> Pipeline:
    """Fit log1p scaling + Isolation Forest on the training features only (no labels)."""
    model = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p)),
            (
                "iforest",
                IsolationForest(
                    n_estimators=N_ESTIMATORS,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model.fit(X_train)


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return one anomaly score per session. Higher means more anomalous.

    sklearn's IsolationForest.score_samples is an *inlier* score: typical (normal)
    sessions get larger values, and easy-to-isolate (anomalous) sessions get smaller,
    often more negative, values.

    evaluate() and best_f1_threshold() assume the opposite polarity — they flag a
    session as anomalous when score >= cutoff, the same convention as logistic
    regression's P(anomaly). Negating the sklearn scores flips the axis so that
    convention holds:

        sklearn:   high = normal,   low = anomaly
        ours:      high = anomaly,  low = normal

    Do not pass model.predict() (+1 inlier / -1 outlier) into evaluate(); those labels
    use sklearn's own contamination cutoff, not the validation F1 threshold.
    """
    return -model.score_samples(X)
