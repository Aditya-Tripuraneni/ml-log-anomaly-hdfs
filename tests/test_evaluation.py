import pytest

from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate


def test_evaluate_perfect_predictions():
    metrics = evaluate([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "pr_auc": 1.0}


def test_best_f1_threshold_picks_the_cutoff_with_highest_f1():
    # Cutoff 0.35 flags {0.35, 0.4, 0.8}: precision 2/3, recall 1, F1 0.8 (the best possible).
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    assert best_f1_threshold(y_true, y_score) == pytest.approx(0.35)


def test_best_f1_threshold_separates_perfectly_separable_scores():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.9]
    threshold = best_f1_threshold(y_true, y_score)
    y_pred = [int(score >= threshold) for score in y_score]
    assert evaluate(y_true, y_pred, y_score)["f1"] == 1.0
