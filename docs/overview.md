# Project Overview

Source of truth: the Group 07 Project Proposal and Team Contract (Google Doc, "ML Project Fall 2026").

## Task

- **Type:** binary sequence classification
- **Input:** one HDFS block session, an ordered sequence of parsed log event-template IDs
- **Output:** normal or anomalous
- **Data:** Loghub HDFS_v1, 575,061 sessions, 16,838 (2.93%) anomalous

## Question

How much are labels actually worth? Compare supervised classifiers against unsupervised
detectors on the same pipeline.

## Pipeline rules (do not break these)

1. Same preprocessing for every model: Loghub's pre-parsed event traces (we did not run
   Drain), grouped by block ID and joined with the labels.
2. Same stratified 70/10/20 train/val/test split (`hdfs_log_anomaly.data.make_splits`, seed 42).
3. Unsupervised models never see labels in training. Labels are for evaluation only.
   Unsupervised thresholds are tuned on the validation set only.
4. Report precision, recall, F1, PR-AUC on the anomalous class. Never accuracy.
5. Supervised models use class weighting. They are also retrained on 1%, 10% and 100% of
   the training labels (`scripts/run_label_fraction.py`).

## Features

- Event-count vectors, scaled with log(1 + x) (used by logistic regression and Isolation Forest)
- Raw ordered event sequence (used by the LSTM)
- Planned, not done yet: TF-IDF and event n-gram variants of the count vectors (n-grams keep
  some order)

## Models (4, agreed in the proposal)

| Model | Family | Input |
|---|---|---|
| Logistic Regression | Supervised (interpretable baseline) | count vectors |
| XGBoost | Supervised | count vectors |
| Isolation Forest | Unsupervised | count vectors |
| DeepLog-style LSTM | Unsupervised | event sequences |

Training data: logistic regression and Isolation Forest train on all 402,541 training
sessions. The LSTM trains on a random subset of 50,000 of them (the subset is not filtered, so
it still contains the usual ~3% anomalous sessions). XGBoost trains on all 402,541 training
sessions, with `scale_pos_weight` set to the negative/positive ratio of the training split.
