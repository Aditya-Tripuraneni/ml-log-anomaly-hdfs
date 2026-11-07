"""Run Isolation Forest end to end and save its metrics.

Usage: uv run python scripts/run_isolation_forest.py

Steps: load sessions -> split -> count vectors (vocabulary from train only) -> fit
Isolation Forest on train features only -> invert inlier scores -> pick the F1 cutoff
on validation -> apply once to test.
"""

import json
from pathlib import Path

from hdfs_log_anomaly.data import SEED, SPLIT, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models.isolation_forest import N_ESTIMATORS, scores, train
from hdfs_log_anomaly.plots import plot_confusion_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTPUT = RESULTS / "metrics" / "isolation_forest.json"
FIGURES = RESULTS / "figures"


def main() -> None:
    train_df, val, test = make_splits(load_sessions())

    vectorizer = EventCountVectorizer().fit(train_df)
    X_train, X_val, X_test = (vectorizer.transform(df) for df in (train_df, val, test))
    y_val, y_test = val.label.to_numpy(), test.label.to_numpy()

    model = train(X_train)
    val_scores = scores(model, X_val)
    test_scores = scores(model, X_test)

    threshold = best_f1_threshold(y_val, val_scores)
    metrics = {
        split: {k: float(v) for k, v in evaluate(y, (s >= threshold).astype(int), s).items()}
        for split, y, s in (("val", y_val, val_scores), ("test", y_test, test_scores))
    }

    results = {
        "model": "isolation_forest",
        "seed": SEED,
        "split": SPLIT,
        "split_sizes": {"train": len(train_df), "val": len(val), "test": len(test)},
        "n_estimators": N_ESTIMATORS,
        "contamination": "auto",
        "f1_threshold_from_val": threshold,
        "metrics": metrics,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))

    plot_confusion_matrix(
        y_test, (test_scores >= threshold).astype(int),
        "Isolation Forest: confusion matrix (test set)",
        FIGURES / "isolation_forest_confusion_matrix.png",
    )

    print(f"threshold = {threshold:.4f}")
    for split, values in metrics.items():
        print(f"{split:>4}: " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
