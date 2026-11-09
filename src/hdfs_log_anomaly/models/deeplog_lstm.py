"""Unsupervised: DeepLog-style LSTM on ordered event sequences.

Labels are never passed to train(). They are used only later, on the validation split, to
pick an F1 cutoff (see evaluation.best_f1_threshold). Input is the ordered event list per
session from features.event_sequences, so unlike the other models it sees event order.

The LSTM learns to predict each event from the WINDOW events before it. Each session also
ends with an end-of-session token, so a session that stops early (e.g. [22, 5]) looks
surprising too. A session's score is its most surprising next token (-log P);
method="rank" instead uses DeepLog's original top-k rule and is kept as an ablation.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

RANDOM_STATE = 42
WINDOW = 10  # h in DeepLog
EMBED_DIM = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
TRAIN_SESSIONS = 50_000
UNSEEN_SURPRISE = 100.0  # -log P for an event type never seen in training (P ~ e^-100)
SCORE_CHUNK = 20_000  # sessions scored per chunk, keeps memory flat on the full splits

# Tokens: 0 = PAD (pads short histories), 1..V = known events, V + 1 = end of session,
# V + 2 = event never seen in training. The model predicts V + 1 classes: events and EOS.
PAD = 0


class NextEventLSTM(nn.Module):
    """Embedding -> stacked LSTM -> linear layer over the V known events plus EOS."""

    def __init__(self, n_events: int):
        super().__init__()
        self.embed = nn.Embedding(n_events + 3, EMBED_DIM, padding_idx=PAD)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN_SIZE, n_events + 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(self.embed(windows))
        return self.out(hidden[:, -1])  # logits for the event after the window


@dataclass
class DeepLogModel:
    net: NextEventLSTM
    event_ids: np.ndarray  # sorted event IDs seen in training; token = position + 1
    window: int = WINDOW
    train_losses: list[float] = field(default_factory=list)  # mean loss per epoch

    @property
    def n_events(self) -> int:
        return len(self.event_ids)

    @property
    def n_classes(self) -> int:
        return self.n_events + 1  # known events + EOS

    def tokenize(self, sequence) -> np.ndarray:
        """Event IDs -> tokens 1..V followed by EOS (V + 1); unseen events become V + 2."""
        sequence = np.asarray(sequence)
        pos = np.searchsorted(self.event_ids, sequence).clip(max=self.n_events - 1)
        known = self.event_ids[pos] == sequence
        tokens = np.where(known, pos + 1, self.n_events + 2)
        return np.append(tokens, self.n_events + 1).astype(np.int64)


def make_windows(token_seqs: list[np.ndarray], window: int = WINDOW):
    """Return (windows, targets, session_index), one row per event of every session.

    windows[i] holds the `window` tokens before event i (PAD where the session had not
    started yet), targets[i] is event i's token, session_index[i] is its session.
    """
    lengths = np.array([len(s) for s in token_seqs])
    padded = np.concatenate(
        [np.concatenate([np.full(window, PAD, dtype=np.int64), s]) for s in token_seqs]
    )
    session_starts = np.concatenate([[0], np.cumsum(lengths + window)[:-1]])
    session_index = np.repeat(np.arange(len(token_seqs)), lengths)
    offset_in_session = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    target_pos = session_starts[session_index] + window + offset_in_session

    windows = sliding_window_view(padded, window)[target_pos - window]
    return windows.copy(), padded[target_pos], session_index


def train(
    sequences: list,
    n_sessions: int = TRAIN_SESSIONS,
    epochs: int = EPOCHS,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> DeepLogModel:
    """Fit the next-event LSTM on (a random subset of) training sequences. No labels."""
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    if n_sessions is not None and n_sessions < len(sequences):
        chosen = rng.choice(len(sequences), size=n_sessions, replace=False)
        sequences = [sequences[i] for i in np.sort(chosen)]

    event_ids = np.array(sorted({e for s in sequences for e in s}))
    model = DeepLogModel(NextEventLSTM(len(event_ids)), event_ids)
    windows, targets, _ = make_windows([model.tokenize(s) for s in sequences], model.window)
    X = torch.from_numpy(windows)
    y = torch.from_numpy(targets - 1)  # tokens 1..V+1 -> classes 0..V (V = EOS)

    optimizer = torch.optim.Adam(model.net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(random_state)

    model.net.train()
    for epoch in range(epochs):
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for start in range(0, len(X), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model.net(X[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.train_losses.append(total / len(X))
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss {model.train_losses[-1]:.4f}")

    model.net.eval()
    return model


@torch.no_grad()
def event_surprise(model: DeepLogModel, windows: np.ndarray, targets: np.ndarray):
    """Per event: rank of the true next token among the predictions (0 = top guess) and
    its negative log-probability. Unseen events get the worst rank and UNSEEN_SURPRISE."""
    ranks = np.empty(len(targets), dtype=np.int64)
    nll = np.empty(len(targets))
    for start in range(0, len(targets), 8 * BATCH_SIZE):
        stop = start + 8 * BATCH_SIZE
        log_probs = torch.log_softmax(model.net(torch.from_numpy(windows[start:stop])), dim=1)
        tgt = torch.from_numpy(targets[start:stop])
        known = tgt <= model.n_classes
        true = log_probs.gather(1, (tgt.clamp(max=model.n_classes) - 1).unsqueeze(1))
        rank = (log_probs > true).sum(dim=1)
        ranks[start:stop] = torch.where(known, rank, model.n_classes).numpy()
        nll[start:stop] = torch.where(known, -true.squeeze(1), UNSEEN_SURPRISE).numpy()
    return ranks, nll


def scores(model: DeepLogModel, sequences: list, method: str = "nll") -> np.ndarray:
    """Return one anomaly score per session; higher means more anomalous.

    method="nll" (default): the largest -log P of any of its next tokens (events + EOS).
    method="rank": the worst rank of any next token; cutoff k = DeepLog's top-k rule.
    """
    if method not in ("nll", "rank"):
        raise ValueError(f"method must be 'nll' or 'rank', got {method!r}")
    result = np.empty(len(sequences))
    for start in range(0, len(sequences), SCORE_CHUNK):
        chunk = [model.tokenize(s) for s in sequences[start : start + SCORE_CHUNK]]
        windows, targets, session_index = make_windows(chunk, model.window)
        ranks, nll = event_surprise(model, windows, targets)
        per_event = ranks if method == "rank" else nll
        worst = np.full(len(chunk), -np.inf)
        np.maximum.at(worst, session_index, per_event)
        result[start : start + len(chunk)] = worst
    return result
