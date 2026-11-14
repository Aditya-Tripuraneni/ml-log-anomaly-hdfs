"""Label-fraction experiment: how do the supervised models do with fewer labels?

Usage: uv run python scripts/run_label_fraction.py

Trains logistic regression and XGBoost on 1%, 10% and 100% of the training sessions
(stratified, several random draws for the partial fractions), using the hyperparameters
already chosen on validation with the full training set. The F1 cutoff is re-picked on the
full validation split for every run and applied once to test. Writes
results/metrics/label_fraction.json.
"""

import json
from pathlib import Path

import numpy as np

from hdfs_log_anomaly.data import SEED, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import EventCountVectorizer
from hdfs_log_anomaly.label_fraction import subsample_labels
from hdfs_log_anomaly.models import logistic_regression as lr
from hdfs_log_anomaly.models import xgboost_model as xgb

RESULTS = Path(__file__).resolve().parents[1] / "results"
FRACTIONS = (0.01, 0.10, 1.0)
DRAWS = 5  # random subsamples per fraction below 100%


def main() -> None:
    train, val, test = make_splits(load_sessions())
    vectorizer = EventCountVectorizer().fit(train)
    X_train, X_val, X_test = (vectorizer.transform(df) for df in (train, val, test))
    y_train, y_val, y_test = (df.label.to_numpy() for df in (train, val, test))

    chosen_C = json.loads((RESULTS / "metrics" / "logistic_regression.json").read_text())["chosen_C"]
    xgb_params = json.loads((RESULTS / "metrics" / "xgboost.json").read_text())["chosen"]
    models = {
        "logistic_regression": (lr, lambda X, y: lr.train(X, y, C=chosen_C)),
        "xgboost": (xgb, lambda X, y: xgb.train(X, y, **xgb_params)),
    }

    rows = []
    for name, (module, fit) in models.items():
        for fraction in FRACTIONS:
            runs = []
            for draw in range(1 if fraction == 1.0 else DRAWS):
                X_sub, y_sub = subsample_labels(X_train, y_train, fraction, seed=SEED + draw)
                model = fit(X_sub, y_sub)
                threshold = best_f1_threshold(y_val, module.scores(model, X_val))
                test_scores = module.scores(model, X_test)
                runs.append(
                    {k: float(v) for k, v in evaluate(
                        y_test, (test_scores >= threshold).astype(int), test_scores
                    ).items()}
                    | {"train_sessions": len(y_sub), "train_anomalies": int(y_sub.sum())}
                )
            row = {"model": name, "fraction": fraction, "runs": runs}
            for metric in ("precision", "recall", "f1", "pr_auc"):
                values = [r[metric] for r in runs]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
            rows.append(row)
            print(f"{name:20s} {fraction:5.0%}  F1 {row['f1_mean']:.3f} ± {row['f1_std']:.3f}"
                  f"  PR-AUC {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f}"
                  f"  ({runs[0]['train_sessions']:,} sessions, {runs[0]['train_anomalies']:,} anomalies)")

    out = RESULTS / "metrics" / "label_fraction.json"
    out.write_text(json.dumps({"seed": SEED, "draws": DRAWS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
