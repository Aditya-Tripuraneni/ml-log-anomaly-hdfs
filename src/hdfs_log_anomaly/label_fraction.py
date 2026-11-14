"""Label-fraction experiment helper: train on only a fraction of the labelled sessions.

Answers "how much are labels worth?" by shrinking the supervised training set while the
validation and test splits stay untouched.
"""

import numpy as np


def subsample_labels(
    X: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a random `fraction` of the rows, keeping the normal/anomalous ratio.

    Each class is sampled separately, so the anomaly rate stays about the same and even a
    tiny fraction keeps at least one session of each class. The same seed gives the same
    sample; rows stay aligned with their labels.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        n = max(1, round(len(rows) * fraction))
        keep.append(rng.choice(rows, size=n, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]
