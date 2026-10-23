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

1. Same preprocessing for every model: Drain parsing, group by block ID, join labels.
2. Same stratified 70/10/20 train/val/test split (`hdfs_log_anomaly.data.make_splits`, seed 42).
3. Unsupervised models never see labels in training. Labels are for evaluation only.
   Unsupervised thresholds are tuned on the validation set only.
4. Report precision, recall, F1, PR-AUC on the anomalous class. Never accuracy.
5. Supervised models use class weighting. Also retrain them on 1%, 10%, 100% of labels.

## Features

- Event-count vectors, with TF-IDF and n-gram variants (n-grams keep some order)
- Raw ordered event sequence (LSTM)

## Models (4, agreed in the proposal)

| Model | Family | Input |
|---|---|---|
| Logistic Regression | Supervised (interpretable baseline) | count vectors |
| XGBoost | Supervised | count vectors |
| Isolation Forest | Unsupervised | count vectors |
| DeepLog-style LSTM | Unsupervised (fallback: PCA) | event sequences |

Fallback: if the full dataset is too costly for the LSTM, use a stratified subset that keeps
the original anomaly rate.

## Open item

The Team Contract (Section 3, item 1) says "2 models per individual ... a total of 6
(3 each)", but the proposal and TA sign-off say 4 models. We are building 4 for now and may
add 2 later. The contract wording should be amended to match.
