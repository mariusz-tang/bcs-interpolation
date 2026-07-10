"""Data visualization utilities."""

from collections.abc import Collection, Sequence
from copy import deepcopy

import matplotlib.pyplot as plt


def diff_comparison(data: Sequence[dict], dataset_names: Collection) -> plt.Figure:
    """Plot mesh diff data."""
    if len(data) < 1:
        raise ValueError("no datasets provided")

    if len(data) != len(dataset_names):
        raise ValueError(
            f"the number of datasets ({len(data)}) must equal the number of "
            f"dataset names ({len(dataset_names)})"
        )

    labels = list(data[0])
    maxes = {name: [] for name in dataset_names}
    means = deepcopy(maxes)
    medians = deepcopy(maxes)
    stds = deepcopy(maxes)

    for name, d in zip(dataset_names, data, strict=True):
        for entry in d.values():
            maxes[name].append(entry["max"])
            means[name].append(entry["mean"])
            medians[name].append(entry["median"])
            stds[name].append(entry["std"])

    fig, _ = plt.subplots(2, 2, layout="constrained")

    def _subplot(position: int, data: dict, title: str) -> None:
        ax = plt.subplot(position)
        ax.grouped_bar(data, tick_labels=labels)
        ax.set_title(title)
        ax.set_xlabel("frame")

    _subplot(221, maxes, "max")
    # Draw the legend at this point to prevent duplicate keys.
    fig.legend()
    _subplot(222, means, "mean")
    _subplot(223, medians, "median")
    _subplot(224, stds, "std")

    return fig


def show() -> None:
    """Show figures."""
    plt.show()
