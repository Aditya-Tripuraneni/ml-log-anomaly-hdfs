import numpy as np
import pandas as pd
import pytest

from hdfs_log_anomaly.data import (
    PREPROCESSED_DIR,
    SEED,
    SPLIT,
    load_sessions,
    make_splits,
    parse_events,
)

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)


def synthetic_sessions(n=10_000, anomaly_rate=0.03, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < anomaly_rate).astype(int)
    return pd.DataFrame(
        {
            "block_id": [f"blk_{i}" for i in range(n)],
            "events": [[5, 22, 11] for _ in range(n)],
            "label": labels,
        }
    )


def test_parse_events_strips_brackets_and_prefix():
    assert parse_events("[E5,E22,E5,E11]") == [5, 22, 5, 11]


def test_splits_are_disjoint_and_cover_everything():
    sessions = synthetic_sessions()
    train, val, test = make_splits(sessions)
    ids = [set(df.block_id) for df in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert sum(len(df) for df in (train, val, test)) == len(sessions)
    assert set().union(*ids) == set(sessions.block_id)


def test_split_sizes_match_ratios():
    sessions = synthetic_sessions()
    splits = make_splits(sessions)
    for df, ratio in zip(splits, SPLIT):
        assert abs(len(df) / len(sessions) - ratio) < 0.005


def test_splits_are_stratified():
    sessions = synthetic_sessions()
    overall = sessions.label.mean()
    for df in make_splits(sessions):
        assert abs(df.label.mean() - overall) < 0.002


def test_split_is_deterministic_and_ignores_row_order():
    sessions = synthetic_sessions()
    a = make_splits(sessions)
    b = make_splits(sessions.sample(frac=1, random_state=123))
    for x, y in zip(a, b):
        assert list(x.block_id) == list(y.block_id)


def test_different_seed_gives_different_split():
    sessions = synthetic_sessions()
    a = make_splits(sessions, seed=SEED)
    b = make_splits(sessions, seed=SEED + 1)
    assert set(a[2].block_id) != set(b[2].block_id)


def test_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        make_splits(synthetic_sessions(), split=(0.5, 0.2, 0.2))


@pytest.fixture(scope="module")
def real_sessions():
    return load_sessions()


@needs_data
def test_real_data_counts_match_proposal(real_sessions):
    assert len(real_sessions) == 575_061
    assert real_sessions.label.sum() == 16_838
    assert not real_sessions.block_id.duplicated().any()


@needs_data
def test_real_data_has_no_missing_values_or_empty_sessions(real_sessions):
    assert not real_sessions.isna().any().any()
    assert real_sessions.events.map(len).min() >= 1


@needs_data
def test_real_data_only_exposes_agreed_columns(real_sessions):
    # Type / Latency / TimeInterval from Event_traces.csv leak the label or are out of scope.
    assert list(real_sessions.columns) == ["block_id", "events", "label"]


@needs_data
def test_real_data_matches_proposal_example(real_sessions):
    row = real_sessions.set_index("block_id").loc["blk_7503483334202473044"]
    assert row.label == 0
    assert row.events == [5, 5, 22, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26, 3, 2, 2, 23, 23, 23, 21, 21, 21]


@needs_data
def test_real_splits_keep_anomaly_rate(real_sessions):
    rate = real_sessions.label.mean()
    for df in make_splits(real_sessions):
        assert abs(df.label.mean() - rate) < 0.001
