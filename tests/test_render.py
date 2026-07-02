import math

import pytest
import torch

from bcsi import render


@pytest.mark.parametrize(
    ("valence", "expected_vertices", "expected_triangles"),
    [
        (
            3,
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1 / 2, math.sqrt(3) / 2, 0],
                [-1 / 2, -math.sqrt(3) / 2, 0],
            ],
            [[0, 1, 2], [0, 2, 3], [0, 3, 1]],
        ),
        (
            4,
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
            ],
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
        ),
    ],
)
def test_onering_patch_resolution_zero(
    valence: int, expected_vertices: list, expected_triangles: list
) -> None:
    result = render.onering_patch(valence, resolution=0)

    assert result.vertices.shape == torch.tensor(expected_vertices).shape
    assert torch.allclose(result.vertices, torch.tensor(expected_vertices).double())
    assert torch.equal(result.triangles, torch.tensor(expected_triangles))


@pytest.mark.parametrize(
    ("valence", "resolution"),
    [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (5, 3),
    ],
)
def test_onering_patch_resolution_nonzero(valence: int, resolution: int) -> None:
    result_base = render.onering_patch(valence, resolution=0)
    result_base_subdivided = result_base.subdivide_midpoint(resolution)
    result_actual = render.onering_patch(valence, resolution)

    assert result_actual.vertices.shape == result_base_subdivided.vertices.shape
    assert torch.allclose(result_actual.vertices, result_base_subdivided.vertices)
    assert torch.equal(result_actual.triangles, result_base_subdivided.triangles)


def test_triangle_patch_resolution_zero() -> None:
    result = render.triangle_patch(resolution=0)

    expected_vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1 / 2, math.sqrt(3) / 2, 0]]
    )
    assert result.vertices.shape == expected_vertices.shape
    assert torch.allclose(result.vertices, expected_vertices.double())

    expected_triangles = torch.tensor([[0, 1, 2]])
    assert torch.equal(result.triangles, expected_triangles)


@pytest.mark.parametrize("resolution", range(5))
def test_triangle_patch_resolution_nonzero(resolution: int) -> None:
    result_base = render.triangle_patch(resolution=0)
    result_base_subdivided = result_base.subdivide_midpoint(resolution)
    result_actual = render.triangle_patch(resolution)

    assert result_actual.vertices.shape == result_base_subdivided.vertices.shape
    assert torch.allclose(result_actual.vertices, result_base_subdivided.vertices)
    assert torch.equal(result_actual.triangles, result_base_subdivided.triangles)
