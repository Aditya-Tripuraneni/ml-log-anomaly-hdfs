"""Shared feature builders.

Owner: shared. Count vectors feed the classical models (Logistic Regression, XGBoost,
Isolation Forest); the raw ordered sequence feeds the LSTM.
"""

import numpy as np
import pandas as pd


class EventCountVectorizer:
    """Turn each session's event list into one row of per-event counts.

    fit() learns which events get a column, from the sessions it is given (pass the train
    split only). transform() then counts events for any split using those same columns, so
    train, val and test always share one layout. Events not seen during fit are ignored.
    """

    def fit(self, sessions: pd.DataFrame) -> "EventCountVectorizer":
        event_ids = sorted({event for events in sessions.events for event in events})
        self.event_ids_ = np.array(event_ids)
        self.feature_names_ = [f"E{event}" for event in event_ids]
        return self

    def transform(self, sessions: pd.DataFrame) -> np.ndarray:
        """Return an (n_sessions, n_events) integer matrix of event counts.

        Input: a sessions DataFrame (block_id, events, label), where each `events` cell is
        one block's ordered list of event IDs, e.g. [5, 5, 22, 5]. Lists differ in length.

        Output: a NumPy matrix with one row per session (same order as the input) and one
        column per fitted event type, in `feature_names_` order. Each cell is how many times
        that event occurred in that session. Order within the session is lost, and the
        label is not included.

        Example, fitted on event types [5, 7, 22]:

            events           ->  E5  E7  E22
            [5, 5, 22, 5]        3   0   1
            [22]                 0   0   1
            [7, 5]               1   1   0
        """
        if not hasattr(self, "event_ids_"):
            raise RuntimeError("EventCountVectorizer must be fitted before transform")

        n_sessions, n_events = len(sessions), len(self.event_ids_)
        lengths = sessions.events.map(len).to_numpy()
        events = np.concatenate(sessions.events.to_list())
        rows = np.repeat(np.arange(n_sessions), lengths)

        # Map each event ID to its column; drop events that were not in the fitted vocabulary.
        cols = np.searchsorted(self.event_ids_, events).clip(max=n_events - 1)
        known = self.event_ids_[cols] == events

        flat_cell = rows[known] * n_events + cols[known]
        counts = np.bincount(flat_cell, minlength=n_sessions * n_events)
        return counts.reshape(n_sessions, n_events).astype(np.int32)

    def fit_transform(self, sessions: pd.DataFrame) -> np.ndarray:
        return self.fit(sessions).transform(sessions)


def event_sequences(sessions: pd.DataFrame) -> list[np.ndarray]:
    """Return each session's ordered event IDs as an integer array, in input row order.

    This is the LSTM's input: unlike the count vectors, event order is kept, and sessions
    keep their own lengths (no padding here; the model pads its own history windows).

        events           ->  array \
        [5, 5, 22, 5]        array([ 5,  5, 22,  5]) \
        [22]                 array([22]) \
    """
    return [np.asarray(events, dtype=np.int64) for events in sessions.events]
