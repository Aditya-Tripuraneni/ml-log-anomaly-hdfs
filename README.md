# Log-Based Anomaly Detection on HDFS: Supervised vs. Unsupervised

SFWRENG 4AL3, Group 07. We detect anomalous HDFS block sessions (ordered sequences of
Drain-parsed event IDs) and compare supervised classifiers against label-free methods
on identical preprocessing and splits.

| Family | Models |
|---|---|
| Supervised | Logistic Regression, XGBoost |
| Unsupervised | Isolation Forest, DeepLog-style LSTM (fallback: PCA) |

Metrics: precision, recall, F1 and PR-AUC on the anomalous class (no accuracy).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # creates .venv and installs everything
uv run pytest           # run tests
uv run ruff check .     # lint
```

Get the data first: see [data/README.md](data/README.md).

## Layout

```
src/hdfs_log_anomaly/   shared code (data, features, evaluation) + models/
tests/                  pytest tests
data/                   raw/ and processed/ (gitignored, see data/README.md)
notebooks/              exploration and result analysis
results/                metrics/ and figures/ produced by experiments
scripts/                runnable entry points
docs/                   overview, task split, progress log
```

Start with [docs/overview.md](docs/overview.md) and [docs/task-split.md](docs/task-split.md).
