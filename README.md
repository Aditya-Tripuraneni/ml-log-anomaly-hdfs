# Log-Based Anomaly Detection on HDFS: Supervised vs. Unsupervised

SFWRENG 4AL3, Group 07. We detect anomalous HDFS block sessions (ordered sequences of
event-template IDs, taken from Loghub's pre-parsed traces) and compare supervised classifiers
against label-free methods on identical preprocessing and splits.

| Family | Models |
|---|---|
| Supervised | Logistic Regression, XGBoost |
| Unsupervised | Isolation Forest, DeepLog-style LSTM |

Metrics: precision, recall, F1 and PR-AUC on the anomalous class (no accuracy).

Who owns what and the current status of each model: [docs/task-split.md](docs/task-split.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # creates .venv and installs everything
uv run pytest           # run tests
uv run ruff check .     # lint
```

Get the data first: see [data/README.md](data/README.md), or run
`uv run python scripts/download_data.py`.

Run an experiment (each script writes its metrics JSON and confusion matrix to `results/`):

```bash
uv run python scripts/run_logistic_regression.py
uv run python scripts/run_isolation_forest.py
uv run python scripts/run_deeplog_lstm.py
```

## Layout

```
src/hdfs_log_anomaly/   shared code (data, features, evaluation, plots) + models/
tests/                  pytest tests
data/                   raw/ and processed/ (gitignored, see data/README.md)
notebooks/              exploration and result analysis
results/                metrics/ and figures/ produced by experiments
scripts/                runnable entry points
docs/                   dataset primer, overview, task split, progress log
project proposal/       the progress report (LaTeX source, figures, PDF)
```

Start with [docs/dataset-primer.md](docs/dataset-primer.md) (what the data is),
[docs/overview.md](docs/overview.md) and [docs/task-split.md](docs/task-split.md).
