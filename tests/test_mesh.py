import torch

from bcsi import mesh


def test_trivert_adjacency_matrix() -> None:
    mesh_ = mesh.from_tensors(
        torch.tensor([[0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )
    expected_matrix = torch.tensor(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
    )
    assert torch.equal(mesh_.trivert_adjacency_matrix, expected_matrix)
