import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models.logistic_regression import (
    coefficients,
    scores,
    train,
    tune_C,
)

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)

FEATURE_NAMES = ["E2", "E5", "E7"]


def synthetic_counts(n=2000, anomaly_rate=0.1, seed=0):
    """Count rows where anomalies have many E7 events and normals have almost none."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < anomaly_rate).astype(int)
    X = np.column_stack(
        [
            rng.poisson(2, n),  # E2: noise
            rng.poisson(3, n),  # E5: noise
            np.where(y == 1, rng.poisson(5, n) + 1, rng.poisson(0.1, n)),  # E7: the signal
        ]
    )
    return X, y


def test_model_is_class_weighted_and_log_transforms_counts():
    X, y = synthetic_counts()
    model = train(X, y)
    assert model.named_steps["clf"].class_weight == "balanced"
    assert model.named_steps["log1p"].transform(np.array([[0, 3]])).tolist() == [[0.0, np.log(4)]]


def test_scores_are_probabilities_one_per_row():
    X, y = synthetic_counts()
    result = scores(train(X, y), X)
    assert result.shape == (len(X),)
    assert result.min() >= 0 and result.max() <= 1


def test_learns_the_signal_on_held_out_data():
    X, y = synthetic_counts(seed=0)
    X_new, y_new = synthetic_counts(seed=1)
    result = scores(train(X, y), X_new)
    assert average_precision_score(y_new, result) > 0.9


def test_coefficients_rank_the_signal_event_first():
    X, y = synthetic_counts()
    ranked = coefficients(train(X, y), FEATURE_NAMES)
    assert isinstance(ranked, pd.Series)
    assert ranked.index[0] == "E7"
    assert list(ranked.values) == sorted(ranked.values, reverse=True)


def test_tune_c_returns_one_row_per_candidate_and_the_best_model():
    X, y = synthetic_counts(seed=0)
    X_val, y_val = synthetic_counts(seed=1)
    Cs = (0.01, 1.0, 100.0)
    best_model, table = tune_C(X, y, X_val, y_val, Cs=Cs)
    assert [row["C"] for row in table] == list(Cs)
    best_row = max(table, key=lambda row: row["val_pr_auc"])
    assert best_model.named_steps["clf"].C == best_row["C"]


@needs_data
def test_real_data_beats_a_trivial_detector_by_a_wide_margin():
    train_df, _, test_df = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train_df)
    model = train(vectorizer.transform(train_df), train_df.label.to_numpy())
    y_score = scores(model, vectorizer.transform(test_df))
    y_pred = (y_score >= 0.5).astype(int)
    metrics = evaluate(test_df.label.to_numpy(), y_pred, y_score)
    # A detector that guesses scores PR-AUC ~0.03 (the anomaly rate), so 0.5 is a loose sanity floor.
    assert metrics["pr_auc"] > 0.5
