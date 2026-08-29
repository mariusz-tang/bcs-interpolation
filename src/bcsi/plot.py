"""Data visualization utilities."""

from collections.abc import Collection, Sequence
from copy import deepcopy

import matplotlib.pyplot as plt
import torch


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
        ax.set_yscale("log")

    _subplot(221, means, "mean")
    # Draw the legend at this point to prevent duplicate keys.
    fig.legend()
    _subplot(222, maxes, "max")
    _subplot(223, medians, "median")
    _subplot(224, stds, "std")

    return fig


def show() -> None:
    """Show figures."""
    plt.show()


def energy_distributions(
    distributions: list[torch.Tensor], labels: list[str]
) -> plt.Figure:
    """Plot cumulative and nominal energy distributions."""
    if len(distributions) != len(labels):
        raise ValueError(
            f"number of distributions ({len(distributions)}) must match the "
            f"number of labels ({len(labels)})"
        )

    fig, _ = plt.subplots(1, 2, layout="constrained")

    cumulative = plt.subplot(121)
    cumulative.set_title("Cumulative energy")
    cumulative.set_xlabel("t")

    nominal = plt.subplot(122)
    nominal.set_title("Derivative")
    nominal.set_xlabel("t")

    num_frames = distributions[0].numel()

    for dist, label in zip(distributions, labels, strict=True):
        cumsum = torch.zeros(num_frames + 1)
        cumsum[1:] = dist.cumsum(0)
        cumulative.plot(torch.linspace(0, 1, num_frames + 1), cumsum, label=label)
        nominal.plot(torch.linspace(0, 1, num_frames), dist * num_frames)

    fig.legend()
    return fig


def nominal_values(
    values_distributions: list[torch.Tensor], labels: list[str]
) -> plt.Figure:
    """Plot nominal value distributions."""
    if len(values_distributions) != len(labels):
        raise ValueError(
            f"number of distributions ({len(values_distributions)}) must match the "
            f"number of labels ({len(labels)})"
        )

    fig = plt.figure()

    axes = plt.axes()
    axes.set_xlabel("t")

    num_frames = values_distributions[0].numel()

    for dist, label in zip(values_distributions, labels, strict=True):
        axes.plot(torch.linspace(0, 1, num_frames), dist, label=label)

    fig.legend()
    return fig
