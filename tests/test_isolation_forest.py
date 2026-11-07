import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models.isolation_forest import scores, train

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)


def synthetic_counts(n=4000, anomaly_rate=0.05, seed=0):
    """Dense Poisson background plus rare one-off spikes (Isolation Forest's intended case)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < anomaly_rate).astype(int)
    X = rng.poisson(2, (n, 6)).astype(float)
    for i in np.flatnonzero(y):
        X[i, rng.integers(0, 6)] += rng.uniform(30, 60)
    return X, y


def test_train_fits_without_labels():
    X, _y = synthetic_counts()
    model = train(X)
    assert model.named_steps["iforest"].random_state == 42
    assert hasattr(model.named_steps["iforest"], "offset_")


def test_scores_are_one_per_row():
    X, _y = synthetic_counts()
    result = scores(train(X), X)
    assert result.shape == (len(X),)


def test_inverted_scores_rank_anomalies_higher_than_sklearn_inlier_scores():
    """Raw score_samples is high for normals; our scores() negate that."""
    X, y = synthetic_counts()
    model = train(X)
    inverted = scores(model, X)
    inlier = model.score_samples(X)
    assert np.allclose(inverted, -inlier)
    assert inverted[y == 1].mean() > inverted[y == 0].mean()
    assert inlier[y == 1].mean() < inlier[y == 0].mean()


def test_learns_the_isolated_cluster_on_held_out_data():
    X, _y = synthetic_counts(seed=0)
    X_new, y_new = synthetic_counts(seed=1)
    result = scores(train(X), X_new)
    assert average_precision_score(y_new, result) > 0.7


@needs_data
def test_real_data_beats_a_rate_only_baseline():
    train_df, _, test_df = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train_df)
    model = train(vectorizer.transform(train_df))
    y_score = scores(model, vectorizer.transform(test_df))
    y_pred = (y_score >= np.median(y_score)).astype(int)
    metrics = evaluate(test_df.label.to_numpy(), y_pred, y_score)
    # Guessing scores PR-AUC ~0.03 (the anomaly rate). 0.1 is a loose sanity floor.
    assert metrics["pr_auc"] > 0.1
