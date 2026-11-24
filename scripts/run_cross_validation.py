"""5-fold cross-validation of logistic regression and XGBoost on train + validation.

Usage: uv run python scripts/run_cross_validation.py

The test split is left untouched. For each fold: fit the count-vector vocabulary and the
model on 8/9 of the other folds, pick the F1 cutoff on the remaining 1/9, then score the
held-out fold once. Hyperparameters are the ones already chosen on the original validation
split. Writes results/metrics/cross_validation.json with every fold's metrics plus the
mean and standard deviation across folds.
"""

import json
from pathlib import Path

import numpy as np

from hdfs_log_anomaly.cross_validation import SEED, cv_splits, pooled_sessions
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.models import logistic_regression as lr
from hdfs_log_anomaly.models import xgboost_model as xgb

RESULTS = Path(__file__).resolve().parents[1] / "results" / "metrics"
N_SPLITS = 5
METRICS = ("precision", "recall", "f1", "pr_auc")


def main() -> None:
    pooled = pooled_sessions()
    y = pooled.label.to_numpy()

    chosen_C = json.loads((RESULTS / "logistic_regression.json").read_text())[
        "chosen_C"
    ]
    xgb_params = json.loads((RESULTS / "xgboost.json").read_text())["chosen"]
    models = {
        "logistic_regression": (lr, lambda X, y_: lr.train(X, y_, C=chosen_C)),
        "xgboost": (xgb, lambda X, y_: xgb.train(X, y_, **xgb_params)),
    }

    folds = {name: [] for name in models}
    for fold, (fit_idx, threshold_idx, eval_idx) in enumerate(
        cv_splits(y, N_SPLITS), start=1
    ):
        vectorizer = EventCountVectorizer().fit(pooled.iloc[fit_idx])
        X_fit, X_threshold, X_eval = (
            vectorizer.transform(pooled.iloc[idx])
            for idx in (fit_idx, threshold_idx, eval_idx)
        )
        for name, (module, fit) in models.items():
            model = fit(X_fit, y[fit_idx])
            cutoff = best_f1_threshold(
                y[threshold_idx], module.scores(model, X_threshold)
            )
            eval_scores = module.scores(model, X_eval)
            metrics = evaluate(
                y[eval_idx], (eval_scores >= cutoff).astype(int), eval_scores
            )
            folds[name].append(
                {k: float(v) for k, v in metrics.items()} | {"fold": fold}
            )
            print(
                f"fold {fold} {name:20s} F1 {metrics['f1']:.4f}  PR-AUC {metrics['pr_auc']:.4f}"
            )

    summary = {
        name: {
            metric: {
                "mean": float(np.mean([f[metric] for f in rows])),
                "std": float(np.std([f[metric] for f in rows], ddof=1)),
            }
            for metric in METRICS
        }
        for name, rows in folds.items()
    }
    out = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "sessions": len(pooled),
        "folds": folds,
        "summary": summary,
    }
    (RESULTS / "cross_validation.json").write_text(json.dumps(out, indent=2))
    for name, stats in summary.items():
        print(name, {m: f"{s['mean']:.4f} ± {s['std']:.4f}" for m, s in stats.items()})


if __name__ == "__main__":
    main()
