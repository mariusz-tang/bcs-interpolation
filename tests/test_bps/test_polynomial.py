import pytest
import torch

from bcsi.bps import polynomial


@pytest.mark.parametrize(
    ("degree", "num_coeffs"),
    [
        (0, 1),
        (1, 3),
        (2, 6),
        (3, 10),
    ],
)
def test_num_coeffs(degree: int, num_coeffs: int) -> None:
    assert polynomial.num_coeffs(degree) == num_coeffs


def test_num_coeffs_negative_degree_raises() -> None:
    with pytest.raises(ValueError, match="degree cannot be negative"):
        polynomial.num_coeffs(-1)


@pytest.mark.parametrize(
    ("x", "y", "degree", "expected_output"),
    [
        (torch.tensor([0]), torch.tensor([0]), 1, torch.tensor([[1, 0, 0]])),
        (
            torch.tensor([0, 1, 3]),
            torch.tensor([0, 2, -4]),
            1,
            torch.tensor([[1, 0, 0], [1, 1, 2], [1, 3, -4]]),
        ),
        (
            torch.tensor([3, 3]),
            torch.tensor([5, 0]),
            2,
            torch.tensor([[1, 3, 5, 9, 15, 25], [1, 3, 0, 9, 0, 0]]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
            1,
            torch.tensor([[[1, 1, 5], [1, 2, 6]], [[1, 3, 7], [1, 4, 8]]]),
        ),
    ],
)
def test_basis(
    x: torch.Tensor, y: torch.Tensor, degree: int, expected_output: torch.Tensor
) -> None:
    assert torch.equal(polynomial.basis(x, y, degree), expected_output)


def test_basis_inconsistent_shape_raises() -> None:
    with pytest.raises(ValueError, match="x and y must have the same shape"):
        polynomial.basis(torch.tensor([0]), torch.tensor([0, 1]), 1)


def test_basis_negative_degree_raises() -> None:
    with pytest.raises(ValueError, match="degree cannot be negative"):
        polynomial.basis(torch.tensor([0]), torch.tensor([0]), -1)
