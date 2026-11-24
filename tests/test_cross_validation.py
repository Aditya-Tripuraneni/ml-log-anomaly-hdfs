import numpy as np
import pytest

from hdfs_log_anomaly.cross_validation import cv_splits


@pytest.fixture
def y():
    return np.array([1] * 100 + [0] * 900)


def test_each_session_is_evaluated_exactly_once(y):
    evaluated = np.concatenate(
        [eval_idx for _, _, eval_idx in cv_splits(y, n_splits=5)]
    )
    assert sorted(evaluated) == list(range(len(y)))


def test_fit_threshold_and_eval_rows_never_overlap(y):
    for fit_idx, threshold_idx, eval_idx in cv_splits(y, n_splits=5):
        assert not set(fit_idx) & set(threshold_idx)
        assert not set(fit_idx) & set(eval_idx)
        assert not set(threshold_idx) & set(eval_idx)
        assert len(fit_idx) + len(threshold_idx) + len(eval_idx) == len(y)


def test_every_part_keeps_the_anomaly_rate(y):
    for fit_idx, threshold_idx, eval_idx in cv_splits(y, n_splits=5):
        for part in (fit_idx, threshold_idx, eval_idx):
            assert y[part].mean() == pytest.approx(0.10, abs=0.01)


def test_threshold_part_is_a_small_share_of_the_training_part(y):
    fit_idx, threshold_idx, eval_idx = next(iter(cv_splits(y, n_splits=5)))
    train_total = len(fit_idx) + len(threshold_idx)
    assert len(threshold_idx) / train_total == pytest.approx(1 / 9, abs=0.01)
    assert len(eval_idx) == pytest.approx(len(y) / 5, abs=2)


def test_same_seed_same_folds_different_seed_differs(y):
    a = [e for _, _, e in cv_splits(y, 5, seed=1)]
    b = [e for _, _, e in cv_splits(y, 5, seed=1)]
    c = [e for _, _, e in cv_splits(y, 5, seed=2)]
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_returns_one_split_per_fold(y):
    assert len(list(cv_splits(y, n_splits=4))) == 4


def test_pooled_sessions_are_train_plus_val_and_exclude_test():
    from hdfs_log_anomaly.cross_validation import pooled_sessions
    from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits

    if not (PREPROCESSED_DIR / "Event_traces.csv").exists():
        pytest.skip("HDFS_v1 not downloaded (run scripts/download_data.py)")
    train, val, test = make_splits(load_sessions())
    pooled = pooled_sessions()
    assert len(pooled) == len(train) + len(val)
    assert set(pooled.block_id) == set(train.block_id) | set(val.block_id)
    assert not set(pooled.block_id) & set(test.block_id)
