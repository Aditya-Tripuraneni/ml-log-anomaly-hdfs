import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models.xgboost_model import (
    feature_importances,
    scores,
    train,
    tune,
)

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)

FEATURE_NAMES = ["E2", "E5", "E7"]
SMALL = {"n_estimators": 30}  # keep the synthetic tests fast


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


def test_class_weight_is_the_negative_to_positive_ratio():
    X, y = synthetic_counts()
    model = train(X, y, **SMALL)
    expected = (y == 0).sum() / (y == 1).sum()
    assert model.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_scores_are_probabilities_one_per_row():
    X, y = synthetic_counts()
    result = scores(train(X, y, **SMALL), X)
    assert result.shape == (len(X),)
    assert result.min() >= 0 and result.max() <= 1


def test_learns_the_signal_on_held_out_data():
    X, y = synthetic_counts(seed=0)
    X_new, y_new = synthetic_counts(seed=1)
    result = scores(train(X, y, **SMALL), X_new)
    assert average_precision_score(y_new, result) > 0.9


def test_training_is_deterministic_for_a_fixed_seed():
    X, y = synthetic_counts()
    a = scores(train(X, y, **SMALL), X)
    b = scores(train(X, y, **SMALL), X)
    assert np.allclose(a, b)


def test_feature_importances_rank_the_signal_event_first():
    X, y = synthetic_counts()
    ranked = feature_importances(train(X, y, **SMALL), FEATURE_NAMES)
    assert isinstance(ranked, pd.Series)
    assert ranked.index[0] == "E7"
    assert list(ranked.values) == sorted(ranked.values, reverse=True)


def test_tune_returns_one_row_per_config_and_the_best_model():
    X, y = synthetic_counts(seed=0)
    X_val, y_val = synthetic_counts(seed=1)
    grid = [
        {"max_depth": 1, "n_estimators": 5},
        {"max_depth": 3, "n_estimators": 30},
    ]
    best_model, table = tune(X, y, X_val, y_val, grid=grid)
    assert [(row["max_depth"], row["n_estimators"]) for row in table] == [(1, 5), (3, 30)]
    best_row = max(table, key=lambda row: row["val_pr_auc"])
    params = best_model.get_params()
    assert (params["max_depth"], params["n_estimators"]) == (
        best_row["max_depth"],
        best_row["n_estimators"],
    )


@needs_data
def test_real_data_beats_a_trivial_detector_by_a_wide_margin():
    train_df, _, test_df = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train_df)
    model = train(vectorizer.transform(train_df), train_df.label.to_numpy())
    y_score = scores(model, vectorizer.transform(test_df))
    # A detector that guesses scores PR-AUC ~0.03 (the anomaly rate), so 0.9 is a loose sanity floor.
    assert average_precision_score(test_df.label.to_numpy(), y_score) > 0.9
