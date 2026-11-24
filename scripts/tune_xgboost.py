"""Wider XGBoost hyperparameter search, compared against the model already in the report.

Usage: uv run python scripts/tune_xgboost.py

Random search over learning rate, depth, number of trees, min_child_weight, row and column
subsampling (40 configurations plus the current report configuration). Every configuration
is trained on the training split and ranked by validation PR-AUC. The best one is then
thresholded the same way as every other model (F1 cutoff from validation, applied once to
test) and compared with the current configuration. Writes
results/metrics/xgboost_tuning.json; it does not change xgboost.json or the report model.
"""

import json
import random
from pathlib import Path

from sklearn.metrics import average_precision_score

from hdfs_log_anomaly.data import SEED, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models import xgboost_model as xgb

RESULTS = Path(__file__).resolve().parents[1] / "results" / "metrics"
SPACE = {
    "learning_rate": [0.05, 0.1, 0.3],
    "max_depth": [4, 6, 8],
    "n_estimators": [200, 500],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
N_CONFIGS = 40


def sample_configs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, configs = set(), []
    while len(configs) < n:
        config = {name: rng.choice(values) for name, values in SPACE.items()}
        key = tuple(config.values())
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def test_metrics(model, X_val, y_val, X_test, y_test) -> dict:
    cutoff = best_f1_threshold(y_val, xgb.scores(model, X_val))
    test_scores = xgb.scores(model, X_test)
    metrics = evaluate(y_test, (test_scores >= cutoff).astype(int), test_scores)
    errors = int(((test_scores >= cutoff).astype(int) != y_test).sum())
    return {k: float(v) for k, v in metrics.items()} | {"test_errors": errors}


def main() -> None:
    train, val, test = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train)
    X_train, X_val, X_test = (vectorizer.transform(df) for df in (train, val, test))
    y_train, y_val, y_test = (df.label.to_numpy() for df in (train, val, test))

    current = json.loads((RESULTS / "xgboost.json").read_text())["chosen"]
    rows, best_model, best = [], None, None
    for i, config in enumerate(sample_configs(N_CONFIGS, SEED), start=1):
        model = xgb.train(X_train, y_train, **config)
        pr_auc = float(average_precision_score(y_val, xgb.scores(model, X_val)))
        rows.append({**config, "val_pr_auc": pr_auc})
        if best is None or pr_auc > best["val_pr_auc"]:
            best_model, best = model, {**config, "val_pr_auc": pr_auc}
        print(f"{i:2d}/{N_CONFIGS} {config} val PR-AUC {pr_auc:.5f}", flush=True)

    current_model = xgb.train(X_train, y_train, **current)
    out = {
        "seed": SEED,
        "n_configs": N_CONFIGS,
        "space": SPACE,
        "current_config": current
        | {
            "val_pr_auc": float(
                average_precision_score(y_val, xgb.scores(current_model, X_val))
            )
        },
        "current_test": test_metrics(current_model, X_val, y_val, X_test, y_test),
        "best_config": best,
        "best_test": test_metrics(best_model, X_val, y_val, X_test, y_test),
        "tuning": sorted(rows, key=lambda r: -r["val_pr_auc"]),
    }
    (RESULTS / "xgboost_tuning.json").write_text(json.dumps(out, indent=2))
    print("current:", out["current_config"], out["current_test"])
    print("best:   ", out["best_config"], out["best_test"])


if __name__ == "__main__":
    main()
