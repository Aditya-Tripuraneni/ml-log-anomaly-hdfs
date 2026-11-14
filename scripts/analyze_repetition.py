"""Measure how repetitive HDFS count vectors are and save the numbers the report cites.

Usage: uv run python scripts/analyze_repetition.py

Writes results/metrics/repetition_analysis.json: unique count vectors, how many test
sessions share a count vector with the training split, and how logistic regression and
XGBoost (with the settings chosen on validation) do on the test sessions whose count
vector never appears in training. Also records the L-BFGS iterations logistic regression
needed.
"""

import json
from pathlib import Path

import numpy as np

from hdfs_log_anomaly.data import SEED, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models import logistic_regression as lr
from hdfs_log_anomaly.models import xgboost_model as xgb

RESULTS = Path(__file__).resolve().parents[1] / "results" / "metrics"
OUTPUT = RESULTS / "repetition_analysis.json"


def main() -> None:
    train, val, test = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train)
    X_train, X_val, X_test = (vectorizer.transform(df) for df in (train, val, test))
    y_train, y_val, y_test = (df.label.to_numpy() for df in (train, val, test))

    seen_in_train = {row.tobytes() for row in X_train}
    seen = np.array([row.tobytes() in seen_in_train for row in X_test])
    anomalous = y_test == 1
    all_vectors = {row.tobytes() for X in (X_train, X_val, X_test) for row in X}

    lr_chosen = json.loads((RESULTS / "logistic_regression.json").read_text())[
        "chosen_C"
    ]
    xgb_chosen = json.loads((RESULTS / "xgboost.json").read_text())["chosen"]
    lr_model = lr.train(X_train, y_train, C=lr_chosen)
    candidates = {
        "logistic_regression": (lr, lr_model),
        "xgboost": (xgb, xgb.train(X_train, y_train, **xgb_chosen)),
    }

    unseen_anomalies = anomalous & ~seen
    chosen = {}
    for name, (module, model) in candidates.items():
        threshold = best_f1_threshold(y_val, module.scores(model, X_val))
        pred = module.scores(model, X_test) >= threshold
        wrong = pred != anomalous
        chosen[name] = {
            "anomalies_caught_among_unseen": int(pred[unseen_anomalies].sum()),
            "errors_total": int(wrong.sum()),
            "errors_on_seen_vectors": int((wrong & seen).sum()),
        }
    n_iter = int(np.max(lr_model.named_steps["clf"].n_iter_))

    results = {
        "seed": SEED,
        "unique_count_vectors_all_sessions": len(all_vectors),
        "test_sessions": len(y_test),
        "test_sessions_seen_in_train": int(seen.sum()),
        "test_sessions_seen_fraction": float(seen.mean()),
        "test_anomalies": int(anomalous.sum()),
        "test_anomalies_seen_in_train": int((anomalous & seen).sum()),
        "test_anomalies_seen_fraction": float(
            (anomalous & seen).sum() / anomalous.sum()
        ),
        "test_sessions_unseen": int((~seen).sum()),
        "test_anomalies_unseen": int((anomalous & ~seen).sum()),
        "models": chosen,
        "logistic_regression_lbfgs_iterations": n_iter,
    }
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
