"""Functions for doing maths on triangles.

This module assumes we are working with equilateral triangles whose vertices
are (0,0), (1,0), and (1/2,sqrt(3)/2), in that order.

The functions take tensors of 2D vertices, of shape (..., 2), as input,
and return tensors with each vertex replaced with the corresponding output
tensor.

The functions in this module assume that each input vertex is inside the
triangle (including the boundary).
"""

import math

import torch


def angles(vertices: torch.Tensor) -> torch.Tensor:
    """Return the (counter-clockwise) angle from each edge to each vertex."""
    _raise_if_bad_vertices_shape(vertices)
    x = vertices[..., 0]
    y = vertices[..., 1]
    angles_raw = torch.stack(
        [
            torch.atan(y / x),
            torch.pi / 3 - torch.atan(y / (1 - x)),
            # If x > 1/2 then the angle from the third edge will be offset by pi.
            # Apply mod before subtraction to avoid numerical issues.
            (
                torch.atan((math.sqrt(3) / 2 - y) / (1 / 2 - x)) % torch.pi
                - torch.pi / 3
            ),
        ],
        dim=-1,
    )
    # We can get nan if an input vertex is exactly on a triangle vertex.
    return torch.where(torch.isnan(angles_raw), 0, angles_raw).double()


def distances(vertices: torch.Tensor) -> torch.Tensor:
    """Return the distance from each input vertex to each triangle vertex."""
    _raise_if_bad_vertices_shape(vertices)
    triangle_vertices = torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    offsets = triangle_vertices - vertices[..., torch.newaxis, :]
    return torch.norm(offsets, dim=-1).double()


def blend_coefficients(vertices: torch.Tensor, beta: float) -> torch.Tensor:
    """Return the smooth blending coefficients at (`x`,`y`).

    Corresponds to B^{2D} in the BCS paper.

    :param beta: controls the overlap size.
    """
    _raise_if_bad_vertices_shape(vertices)
    dist = distances(vertices)
    coefficients_nominal = _blend_1d(dist, beta)
    return (
        coefficients_nominal / coefficients_nominal.sum(dim=-1, keepdim=True).double()
    )


def _blend_1d(x: torch.Tensor, beta: float) -> torch.Tensor:
    # Corresponds to B in the BCS paper.
    return _g((torch.abs(x) - (1 - beta) / 2) / beta)


def _g(x: torch.Tensor) -> torch.Tensor:
    # As defined in the BCS paper.
    return _f(1 - x) / (_f(x) + _f(1 - x))


def _f(x: torch.Tensor) -> torch.Tensor:
    # As defined in the BCS paper.
    return torch.where(x > 0, torch.exp(-1 / x), 0)


def _raise_if_bad_vertices_shape(vertices: torch.Tensor) -> None:
    if vertices.dim() < 2 or vertices.shape[-1] != 2:
        raise ValueError(f"bad shape for vertices {vertices.shape}, expected (..., 2)")
