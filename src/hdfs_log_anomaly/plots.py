"""Shared plots. They take true labels and predictions, not a model, so every model reuses them.

Owner: shared. Static light-mode charts: blue ramp, near-black text.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#e8f0fb", "#86b6ef", "#1c5fb0"])


def plot_confusion_matrix(y_true, y_pred, title: str, path=None) -> Figure:
    """Counts of normal/anomalous sessions by actual vs. predicted class.

    Color and the percentage show each actual class's share, so the rare anomalous row stays
    readable next to the huge normal row. The count is the raw number of sessions.
    """
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1])
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    fig = Figure(figsize=(5.2, 4.8), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.pcolormesh(share, cmap=BLUE_RAMP, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for row in range(2):
        for col in range(2):
            ax.text(
                col + 0.5, row + 0.5, f"{counts[row, col]:,}\n{share[row, col]:.2%}",
                ha="center", va="center", fontsize=12,
                color="white" if share[row, col] > 0.6 else INK,
            )
    ax.set_xticks([0.5, 1.5], ["Normal", "Anomalous"])
    ax.set_yticks([0.5, 1.5], ["Normal", "Anomalous"])
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.tick_params(colors=INK, labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.text(0.02, 0.01, "Color and % show each actual class's share", color=INK_SECONDARY, fontsize=8.5)

    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor=SURFACE)
    return fig
