import numpy as np
import pandas as pd
import pytest

from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits
from hdfs_log_anomaly.features import EventCountVectorizer

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)


def sessions_from(*event_lists):
    return pd.DataFrame(
        {
            "block_id": [f"blk_{i}" for i in range(len(event_lists))],
            "events": list(event_lists),
            "label": [0] * len(event_lists),
        }
    )


def test_counts_each_event_per_session():
    sessions = sessions_from([5, 5, 22, 5], [22], [7, 5])
    vectorizer = EventCountVectorizer().fit(sessions)
    assert vectorizer.feature_names_ == ["E5", "E7", "E22"]
    assert vectorizer.transform(sessions).tolist() == [
        [3, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
    ]


def test_columns_are_sorted_numerically_not_alphabetically():
    vectorizer = EventCountVectorizer().fit(sessions_from([22, 5, 9, 100, 2]))
    assert vectorizer.feature_names_ == ["E2", "E5", "E9", "E22", "E100"]


def test_vocabulary_comes_only_from_the_fitted_sessions():
    train = sessions_from([5, 22])
    unseen = sessions_from([5, 9, 9, 22])
    vectorizer = EventCountVectorizer().fit(train)
    assert vectorizer.feature_names_ == ["E5", "E22"]
    # E9 never appeared in train, so it gets no column and is ignored.
    assert vectorizer.transform(unseen).tolist() == [[1, 1]]


def test_output_shape_and_dtype():
    sessions = sessions_from([5, 5], [22], [7])
    matrix = EventCountVectorizer().fit(sessions).transform(sessions)
    assert matrix.shape == (3, 3)
    assert np.issubdtype(matrix.dtype, np.integer)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        EventCountVectorizer().transform(sessions_from([5]))


def test_fit_transform_matches_fit_then_transform():
    sessions = sessions_from([5, 5, 22], [7])
    a = EventCountVectorizer().fit_transform(sessions)
    b = EventCountVectorizer().fit(sessions).transform(sessions)
    assert (a == b).all()


@pytest.fixture(scope="module")
def real_sessions():
    return load_sessions()


@needs_data
def test_matches_the_datasets_own_occurrence_matrix(real_sessions):
    vectorizer = EventCountVectorizer().fit(real_sessions)
    assert vectorizer.feature_names_ == [f"E{i}" for i in range(1, 30)]  # E30 never occurs

    reference = pd.read_csv(PREPROCESSED_DIR / "Event_occurrence_matrix.csv").set_index("BlockId")
    reference = reference.loc[real_sessions.block_id, vectorizer.feature_names_]
    assert (vectorizer.transform(real_sessions) == reference.to_numpy()).all()


@needs_data
def test_real_train_split_fit_applies_cleanly_to_val_and_test(real_sessions):
    train, val, test = make_splits(real_sessions)
    vectorizer = EventCountVectorizer().fit(train)
    for split in (train, val, test):
        matrix = vectorizer.transform(split)
        assert matrix.shape == (len(split), len(vectorizer.feature_names_))
        assert matrix.min() >= 0
