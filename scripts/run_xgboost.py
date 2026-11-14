"""Run the XGBoost model end to end and save its metrics.

Usage: uv run python scripts/run_xgboost.py

Steps: load sessions -> split -> count vectors (vocabulary from train only) -> pick the tree
settings on validation PR-AUC -> pick the F1 cutoff on validation -> apply both once to test.
"""

import json
from pathlib import Path

from hdfs_log_anomaly.data import SEED, SPLIT, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models.xgboost_model import (
    LEARNING_RATE,
    feature_importances,
    scores,
    tune,
)
from hdfs_log_anomaly.plots import plot_confusion_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTPUT = RESULTS / "metrics" / "xgboost.json"
FIGURES = RESULTS / "figures"


def main() -> None:
    train, val, test = make_splits(load_sessions())

    vectorizer = EventCountVectorizer().fit(train)
    X_train, X_val, X_test = (vectorizer.transform(df) for df in (train, val, test))
    y_train, y_val, y_test = (df.label.to_numpy() for df in (train, val, test))

    model, tuning = tune(X_train, y_train, X_val, y_val)
    val_scores = scores(model, X_val)
    test_scores = scores(model, X_test)

    threshold = best_f1_threshold(y_val, val_scores)
    metrics = {
        split: {k: float(v) for k, v in evaluate(y, (s >= threshold).astype(int), s).items()}
        for split, y, s in (("val", y_val, val_scores), ("test", y_test, test_scores))
    }

    params = model.get_params()
    results = {
        "model": "xgboost",
        "seed": SEED,
        "split": SPLIT,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "scale_pos_weight": params["scale_pos_weight"],
        "learning_rate": LEARNING_RATE,
        "chosen": {"max_depth": params["max_depth"], "n_estimators": params["n_estimators"]},
        "tuning": tuning,
        "f1_threshold_from_val": threshold,
        "metrics": metrics,
        "feature_importances": feature_importances(model, vectorizer.feature_names_).to_dict(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, default=float))

    plot_confusion_matrix(
        y_test, (test_scores >= threshold).astype(int),
        "XGBoost: confusion matrix (test set)",
        FIGURES / "xgboost_confusion_matrix.png",
    )

    print(f"chosen {results['chosen']}, threshold = {threshold:.4f}")
    for split, values in metrics.items():
        print(f"{split:>4}: " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
