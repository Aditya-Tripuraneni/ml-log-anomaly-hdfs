# Task Split

Each person owns their models end to end: script, training, evaluation, and presenting
results at team meetings. Everyone reviews someone else's work.

| Owner | Models / work |
|---|---|
| Aditya | Logistic Regression, XGBoost (both supervised); shared pipeline (`data.py`, `features.py`, `evaluation.py`); label-fraction experiment (1% / 10% / 100%) |
| TBD | Isolation Forest (unsupervised) |
| TBD | DeepLog-style LSTM (unsupervised; PCA fallback) |

Teammates: Harmanpreet Singh Sagar, Vanessa Lai. Fill in the TBD owners once agreed.

## Shared work (everyone)

- [ ] Download HDFS_v1 and confirm the 575,061 / 16,838 counts
- [ ] Drain parsing + sessionization + label join
- [ ] Shared split and feature code merged to `main` before model work starts
- [ ] Final comparison table and plots
- [ ] Project submissions and write-up

## Review pairing

| Author | Reviewer |
|---|---|
| Aditya | TBD |
| TBD | TBD |
| TBD | TBD |

## Extension (only if time allows)

The contract mentions 6 models. If we add 2, candidates: a GRU/BiLSTM classifier
(supervised, sequence-aware) and PCA (unsupervised).
