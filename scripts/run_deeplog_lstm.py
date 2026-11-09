"""Run the DeepLog-style LSTM end to end and save its metrics.

Usage: uv run python scripts/run_deeplog_lstm.py

Steps: load sessions -> split -> ordered event sequences -> train the next-event LSTM on a
random subset of train sessions (no labels) -> score every session by its most surprising
next token (-log P, end-of-session included) -> pick the F1 cutoff on validation -> apply
once to test. The original DeepLog top-k rule is scored the same way as an ablation.
"""

import json
import time
from pathlib import Path

from hdfs_log_anomaly.data import SEED, SPLIT, load_sessions, make_splits
from hdfs_log_anomaly.evaluation import best_f1_threshold, evaluate
from hdfs_log_anomaly.features import event_sequences
from hdfs_log_anomaly.models import deeplog_lstm as dl
from hdfs_log_anomaly.plots import plot_confusion_matrix

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTPUT = RESULTS / "metrics" / "deeplog_lstm.json"
FIGURES = RESULTS / "figures"


def main() -> None:
    train_df, val, test = make_splits(load_sessions())
    y_val, y_test = val.label.to_numpy(), test.label.to_numpy()

    started = time.perf_counter()
    model = dl.train(event_sequences(train_df), verbose=True)
    train_seconds = time.perf_counter() - started

    val_seqs, test_seqs = event_sequences(val), event_sequences(test)

    def run(method: str):
        val_scores = dl.scores(model, val_seqs, method)
        test_scores = dl.scores(model, test_seqs, method)
        threshold = best_f1_threshold(y_val, val_scores)
        metrics = {
            split: {k: float(v) for k, v in evaluate(y, (s >= threshold).astype(int), s).items()}
            for split, y, s in (("val", y_val, val_scores), ("test", y_test, test_scores))
        }
        return threshold, metrics, test_scores

    threshold, metrics, test_scores = run("nll")
    topk_threshold, topk_metrics, _ = run("rank")

    results = {
        "model": "deeplog_lstm",
        "seed": SEED,
        "split": SPLIT,
        "split_sizes": {"train": len(train_df), "val": len(val), "test": len(test)},
        "train_sessions_used": min(dl.TRAIN_SESSIONS, len(train_df)),
        "trained_on_labels": False,
        "hyperparameters": {
            "window": dl.WINDOW,
            "embed_dim": dl.EMBED_DIM,
            "hidden_size": dl.HIDDEN_SIZE,
            "num_layers": dl.NUM_LAYERS,
            "epochs": dl.EPOCHS,
            "batch_size": dl.BATCH_SIZE,
            "learning_rate": dl.LEARNING_RATE,
            "loss": "cross-entropy (next event)",
            "optimizer": "Adam",
        },
        "event_types_seen_in_training": model.n_events,
        "train_loss_per_epoch": model.train_losses,
        "train_seconds": round(train_seconds, 1),
        "score": "max -log P(next token) over the session, end-of-session token included",
        "f1_threshold_from_val": threshold,
        "metrics": metrics,
        "ablation_topk_rank": {
            "score": "worst rank of a true next token (original DeepLog top-k rule)",
            "top_k_from_val": int(topk_threshold),
            "metrics": topk_metrics,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))

    plot_confusion_matrix(
        y_test, (test_scores >= threshold).astype(int),
        "DeepLog LSTM: confusion matrix (test set)",
        FIGURES / "deeplog_lstm_confusion_matrix.png",
    )

    print(f"threshold = {threshold:.4f} (flag if some next token has P < {2.718**-threshold:.4f})")
    for split, values in metrics.items():
        print(f"{split:>4}: " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    print(f"ablation, top-k rule (k = {int(topk_threshold)}):")
    for split, values in topk_metrics.items():
        print(f"{split:>4}: " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
