import inspect

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from hdfs_log_anomaly.data import PREPROCESSED_DIR, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import evaluate
from hdfs_log_anomaly.features import event_sequences
from hdfs_log_anomaly.models.deeplog_lstm import (
    PAD,
    UNSEEN_SURPRISE,
    make_windows,
    scores,
    train,
)

needs_data = pytest.mark.skipif(
    not (PREPROCESSED_DIR / "Event_traces.csv").exists(),
    reason="HDFS_v1 not downloaded (run scripts/download_data.py)",
)


def synthetic_sequences(n=600, anomaly_rate=0.1, seed=0):
    """Normal sessions repeat a fixed lifecycle; anomalies jump to an out-of-order event.

    This is the case DeepLog is built for: the counts of an anomaly can look normal, only
    the order gives it away.
    """
    rng = np.random.default_rng(seed)
    lifecycle = [22, 5, 5, 5, 11, 9, 26, 26, 26, 23, 21]
    sequences, labels = [], []
    for _ in range(n):
        seq = list(lifecycle)
        anomalous = rng.random() < anomaly_rate
        if anomalous:
            i, j = sorted(rng.choice(len(seq), size=2, replace=False))
            seq[i], seq[j] = seq[j], seq[i]
            anomalous = seq != lifecycle
        sequences.append(np.array(seq))
        labels.append(int(anomalous))
    return sequences, np.array(labels)


@pytest.fixture(scope="module")
def synthetic_model():
    sequences, _ = synthetic_sequences(seed=0)
    return train(sequences, n_sessions=None, epochs=15)


def test_make_windows_left_pads_each_session_separately():
    windows, targets, session_index = make_windows([np.array([1, 2, 3]), np.array([4])], 2)
    assert windows.tolist() == [[PAD, PAD], [PAD, 1], [1, 2], [PAD, PAD]]
    assert targets.tolist() == [1, 2, 3, 4]
    assert session_index.tolist() == [0, 0, 0, 1]


def test_train_takes_no_labels():
    assert "y" not in inspect.signature(train).parameters
    assert "labels" not in inspect.signature(train).parameters


def test_training_loss_goes_down(synthetic_model):
    losses = synthetic_model.train_losses
    assert len(losses) == 15
    assert losses[-1] < losses[0] / 2


def test_scores_are_one_per_session(synthetic_model):
    sequences, _ = synthetic_sequences(n=50, seed=3)
    assert scores(synthetic_model, sequences).shape == (50,)


def test_unseen_event_gets_the_worst_score(synthetic_model):
    normal = [22, 5, 5, 5, 11, 9, 26, 26, 26, 23, 21]
    pair = [np.array(normal), np.array(normal[:4] + [999])]
    rank = scores(synthetic_model, pair, method="rank")
    nll = scores(synthetic_model, pair)
    assert rank[1] == synthetic_model.n_classes
    assert nll[1] == UNSEEN_SURPRISE
    assert rank[0] < rank[1] and nll[0] < nll[1]


def test_rejects_unknown_scoring_method(synthetic_model):
    with pytest.raises(ValueError):
        scores(synthetic_model, [np.array([22, 5])], method="mean")


def test_session_that_stops_early_scores_higher_than_a_complete_one(synthetic_model):
    """Every event of a truncated session is normal; only the missing rest gives it away."""
    normal = [22, 5, 5, 5, 11, 9, 26, 26, 26, 23, 21]
    result = scores(synthetic_model, [np.array(normal), np.array(normal[:3])])
    assert result[1] > result[0]


def test_topk_rank_scores_also_detect_out_of_order_events(synthetic_model):
    sequences, y = synthetic_sequences(seed=2)
    assert average_precision_score(y, scores(synthetic_model, sequences, method="rank")) > 0.9


def test_detects_out_of_order_events_on_held_out_data(synthetic_model):
    sequences, y = synthetic_sequences(seed=1)
    assert average_precision_score(y, scores(synthetic_model, sequences)) > 0.9


@needs_data
def test_real_data_beats_a_rate_only_baseline():
    train_df, _, test_df = make_splits(load_sessions())
    test_df = test_df.sample(20_000, random_state=0)
    model = train(event_sequences(train_df), n_sessions=5_000, epochs=2)
    y_score = scores(model, event_sequences(test_df))
    y_pred = (y_score >= np.median(y_score)).astype(int)
    metrics = evaluate(test_df.label.to_numpy(), y_pred, y_score)
    # Guessing scores PR-AUC ~0.03 (the anomaly rate). 0.1 is a loose sanity floor.
    assert metrics["pr_auc"] > 0.1
