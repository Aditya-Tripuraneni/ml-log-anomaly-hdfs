"""Stratified k-fold splits for cross-validating the models on train + validation.

The test split is never touched here. Each fold holds one part out for evaluation, and the
remaining sessions are divided again: most are used to fit, a small stratified part is
used only to pick the F1 cutoff, so the held-out fold never influences the threshold.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from hdfs_log_anomaly.data import load_sessions, make_splits

SEED = 42


def cv_splits(
    y: np.ndarray, n_splits: int = 5, seed: int = SEED
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (fit_idx, threshold_idx, eval_idx) row indices for each fold.

    eval_idx is the held-out fold (each session appears in exactly one). The other folds
    are split into fit_idx (8/9) and threshold_idx (1/9), both stratified on the label,
    so the threshold part is about the size of the original validation split.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, eval_idx in folds.split(np.zeros(len(y)), y):
        fit_idx, threshold_idx = train_test_split(
            train_idx, test_size=1 / 9, stratify=y[train_idx], random_state=seed
        )
        yield np.sort(fit_idx), np.sort(threshold_idx), eval_idx


def pooled_sessions() -> pd.DataFrame:
    """Return the train and validation sessions stacked into one DataFrame (test excluded).

    cv_splits(pooled.label.to_numpy()) indexes into this frame, so every model that calls
    both gets identical folds. Fit anything learned from data (vocabulary, scaler, model)
    on pooled.iloc[fit_idx] only.
    """
    train, val, _ = make_splits(load_sessions())
    return pd.concat([train, val], ignore_index=True)
