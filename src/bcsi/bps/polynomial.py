"""Utilities for working with polynomial patch functions R2 -> R3.

Basis terms are always taken to be in the same order throughout this module:
[1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ...]
"""

import torch


def num_coeffs(degree: int) -> int:
    """Return the number of coefficients for a polynomial of specified degree."""
    if degree < 0:
        raise ValueError(f"polynomial degree cannot be negative (received {degree})")
    return (degree + 1) * (degree + 2) // 2


def basis(x: torch.Tensor, y: torch.Tensor, degree: int) -> torch.Tensor:
    """Return basis terms up to the specified degree.

    Each scalar term in `x` and `y` is replaced by a vector representing the
    basis.

    For example:
    [[1,2],[3,4]], [[5,6],[7,8]], 1 -> [[[1,1,5],[1,2,6]],[[1,3,7],[1,4,8]]]
    """
    if degree < 0:
        raise ValueError(f"polynomial degree cannot be negative (received {degree})")

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape "
            f"(received shapes {x.shape} and {y.shape})"
        )

    results = [torch.ones_like(x)]
    for sum in range(1, degree + 1):
        for i in range(sum, -1, -1):
            j = sum - i
            results.append(x**i * y**j)

    return torch.stack(results, dim=-1)
