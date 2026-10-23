"""Shared feature builders.

Owner: shared. Count vectors (with TF-IDF and n-gram variants) feed the
classical models; the raw ordered sequence feeds the LSTM.
"""


def count_vectors(sessions, ngram_range=(1, 1), tfidf=False):
    """Return an event-count (optionally TF-IDF / n-gram) matrix per session."""
    raise NotImplementedError


def event_sequences(sessions):
    """Return the ordered event-ID sequence per session."""
    raise NotImplementedError
