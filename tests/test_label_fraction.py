import numpy as np
import pytest

from hdfs_log_anomaly.label_fraction import subsample_labels


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 900)
    X = rng.integers(0, 5, size=(1000, 3))
    return X, y


def test_full_fraction_keeps_everything(data):
    X, y = data
    _, y_sub = subsample_labels(X, y, 1.0, seed=0)
    assert len(y_sub) == len(y)
    assert np.array_equal(np.sort(y_sub), np.sort(y))


def test_subsample_size_and_anomaly_rate_preserved(data):
    X, y = data
    X_sub, y_sub = subsample_labels(X, y, 0.1, seed=0)
    assert len(y_sub) == 100
    assert y_sub.sum() == 10
    assert X_sub.shape == (100, 3)


def test_rows_stay_aligned_with_labels(data):
    X, y = data
    X = np.column_stack([X, y])  # last column is a copy of the label
    X_sub, y_sub = subsample_labels(X, y, 0.2, seed=3)
    assert np.array_equal(X_sub[:, -1], y_sub)


def test_same_seed_same_sample_different_seed_differs(data):
    X, y = data
    a, _ = subsample_labels(X, y, 0.1, seed=1)
    b, _ = subsample_labels(X, y, 0.1, seed=1)
    c, _ = subsample_labels(X, y, 0.1, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_keeps_at_least_one_of_each_class_for_tiny_fractions(data):
    X, y = data
    _, y_sub = subsample_labels(X, y, 0.001, seed=0)
    assert set(y_sub) == {0, 1}


def test_rejects_bad_fraction(data):
    X, y = data
    with pytest.raises(ValueError):
        subsample_labels(X, y, 0.0, seed=0)
    with pytest.raises(ValueError):
        subsample_labels(X, y, 1.5, seed=0)
