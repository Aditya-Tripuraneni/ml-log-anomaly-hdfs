"""Shared data loading and the single train/val/test split used by every model.

Owner: Shared. Do not change split logic without telling the team,
since all models must be compared on identical splits.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT = (0.70, 0.10, 0.20)  # train / val / test, stratified on the label

PREPROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "preprocessed"


def parse_events(features: str) -> list[int]:
    """Turn the "[E5,E22,E5]" string from Event_traces.csv into [5, 22, 5]."""
    return [int(event[1:]) for event in features.strip("[]").split(",")]


def load_sessions(directory: Path = PREPROCESSED_DIR) -> pd.DataFrame:
    """Return one row per block session: block_id, ordered event IDs, label (1 = anomalous).

    Only the event sequence and the label are exposed. The Type, Latency and TimeInterval
    columns of Event_traces.csv are deliberately dropped: Type is filled only for anomalous
    sessions (label leakage) and the others are outside the agreed input.
    """
    traces = pd.read_csv(directory / "Event_traces.csv", usecols=["BlockId", "Features"])
    labels = pd.read_csv(directory / "anomaly_label.csv")

    sessions = traces.merge(labels, on="BlockId", how="inner", validate="one_to_one")
    if not len(sessions) == len(traces) == len(labels):
        raise ValueError("Event_traces.csv and anomaly_label.csv do not cover the same blocks")

    sessions["label"] = sessions.pop("Label").map({"Normal": 0, "Anomaly": 1})
    if sessions.label.isna().any():
        raise ValueError("anomaly_label.csv contains labels other than Normal/Anomaly")

    sessions["events"] = sessions.pop("Features").map(parse_events)
    return sessions.rename(columns={"BlockId": "block_id"})[["block_id", "events", "label"]]


def make_splits(
    sessions: pd.DataFrame,
    seed: int = SEED,
    split: tuple[float, float, float] = SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stratified (train, val, test) sessions.

    Sessions are sorted by block_id first, so the result depends only on the seed and the
    data, not on row order. Every teammate therefore gets identical splits without sharing files.
    """
    train_size, val_size, test_size = split
    if abs(train_size + val_size + test_size - 1) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {split}")

    sessions = sessions.sort_values("block_id").reset_index(drop=True)
    train_val, test = train_test_split(
        sessions, test_size=test_size, stratify=sessions.label, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (train_size + val_size),
        stratify=train_val.label,
        random_state=seed,
    )
    return tuple(df.reset_index(drop=True) for df in (train, val, test))
