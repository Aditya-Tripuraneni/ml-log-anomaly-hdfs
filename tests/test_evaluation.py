from hdfs_log_anomaly.evaluation import evaluate


def test_evaluate_perfect_predictions():
    metrics = evaluate([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "pr_auc": 1.0}
