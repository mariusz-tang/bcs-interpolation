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
    plt.subplot(221).grouped_bar(maxes, tick_labels=labels)
    plt.subplot(221).set_title("max")
    plt.subplot(221).set_xlabel("frame")
    fig.legend()

    plt.subplot(222).grouped_bar(means, tick_labels=labels)
    plt.subplot(222).set_title("mean")
    plt.subplot(222).set_xlabel("frame")

    plt.subplot(223).grouped_bar(medians, tick_labels=labels)
    plt.subplot(223).set_title("median")
    plt.subplot(223).set_xlabel("frame")

    plt.subplot(224).grouped_bar(stds, tick_labels=labels)
    plt.subplot(224).set_title("std")
    plt.subplot(224).set_xlabel("frame")

    return fig


def show() -> None:
    """Show figures."""
    plt.show()
