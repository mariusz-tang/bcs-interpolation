import math

import pytest
import torch

from bcsi import triangle


@pytest.mark.parametrize(
    ("vertices", "expected_result"),
    [
        (
            torch.tensor([[0.5, 1 / (math.sqrt(3) * 2)]]),
            torch.tensor([[math.sqrt(3) / 2 - 1 / (math.sqrt(3) * 2)] * 3]),
        ),
        (
            torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]),
            torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        ),
        (
            torch.tensor([[0.5, 0]]),
            torch.tensor([[0.5, 0.5, math.sqrt(3) / 2]]),
        ),
        (
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
            torch.tensor([[[0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1]]]),
        ),
    ],
)
def test_distances(vertices: torch.Tensor, expected_result: torch.Tensor) -> None:
    result = triangle.distances(vertices)
    assert result.shape == expected_result.shape
    assert torch.allclose(result, expected_result.double())


def test_distances_wrong_input_shape_raises() -> None:
    with pytest.raises(ValueError, match="bad shape for vertices"):
        triangle.distances(torch.tensor([[0], [0]]))


@pytest.mark.parametrize(
    ("vertices", "beta", "expected_result"),
    [
        (
            torch.tensor([[0.5, 1 / (math.sqrt(3) * 2)]]),
            0.73,
            torch.tensor([[1 / 3, 1 / 3, 1 / 3]]),
        ),
        (
            torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]),
            0.73,
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        (
            torch.tensor([[0.5, 0]]),
            0.5,
            torch.tensor([[0.5, 0.5, 0]]),
        ),
        (
            torch.tensor([[[0.5, 0], [0.5, 0]], [[0.5, 0], [0.5, 0]]]),
            0.5,
            torch.tensor(
                [[[0.5, 0.5, 0], [0.5, 0.5, 0]], [[0.5, 0.5, 0], [0.5, 0.5, 0]]]
            ),
        ),
    ],
)
def test_blend_coefficients(
    vertices: torch.Tensor, beta: float, expected_result: torch.Tensor
) -> None:
    result = triangle.blend_coefficients(vertices, beta)
    assert result.shape == expected_result.shape
    assert torch.allclose(result, expected_result.double())


def test_blend_coefficients_wrong_input_shape_raises() -> None:
    with pytest.raises(ValueError, match="bad shape for vertices"):
        triangle.blend_coefficients(torch.tensor([[0], [0]]), 0.5)


@pytest.mark.parametrize(
    ("vertices", "expected_result"),
    [
        (
            torch.tensor([[0.5, 1 / (math.sqrt(3) * 2)]]),
            torch.tensor([[torch.pi / 6] * 3]),
        ),
        (
            torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]),
            torch.tensor(
                [[0, torch.pi / 3, 0], [0, 0, torch.pi / 3], [torch.pi / 3, 0, 0]]
            ),
        ),
        (
            torch.tensor([[0.75, math.sqrt(3) / 4]]),
            torch.tensor([[torch.pi / 6, 0, torch.pi / 3]]),
        ),
        (
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
            torch.tensor(
                [
                    [[0, torch.pi / 3, 0], [0, torch.pi / 3, 0]],
                    [[0, torch.pi / 3, 0], [0, torch.pi / 3, 0]],
                ]
            ),
        ),
    ],
)
def test_angles(vertices: torch.Tensor, expected_result: torch.Tensor) -> None:
    result = triangle.angles(vertices)
    assert result.shape == expected_result.shape
    assert torch.allclose(result, expected_result.double())


def test_angles_wrong_input_shape_raises() -> None:
    with pytest.raises(ValueError, match="bad shape for vertices"):
        triangle.angles(torch.tensor([[0], [0]]))
