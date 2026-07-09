"""Data visualization utilities."""

import matplotlib.pyplot as plt


def diff_comparison(data: dict) -> plt.Figure:
    """Plot mesh diff data."""
    names = []
    maxes = []
    means = []
    medians = []
    stds = []

    for label, entry in data.items():
        names.append(label)
        maxes.append(entry["max"])
        means.append(entry["mean"])
        medians.append(entry["median"])
        stds.append(entry["std"])

    fig, _ = plt.subplots(2, 2, layout="constrained")
    plt.subplot(221).bar(names, maxes)
    plt.subplot(221).set_title("max")

    plt.subplot(222).bar(names, means)
    plt.subplot(222).set_title("mean")

    plt.subplot(223).bar(names, medians)
    plt.subplot(223).set_title("median")

    plt.subplot(224).bar(names, stds)
    plt.subplot(224).set_title("std")

    return fig


def show() -> None:
    """Show figures."""
    plt.show()
