"""Shared data loading and the single train/val/test split used by every model.

Owner: shared (Aditya). Do not change split logic without telling the team,
since all models must be compared on identical splits.
"""

SEED = 42
SPLIT = (0.70, 0.10, 0.20)  # train / val / test, stratified on the label


def load_sessions():
    """Return one row per block session: block_id, ordered event IDs, label."""
    raise NotImplementedError


def make_splits(sessions):
    """Return stratified (train, val, test) sessions using SEED and SPLIT."""
    raise NotImplementedError
