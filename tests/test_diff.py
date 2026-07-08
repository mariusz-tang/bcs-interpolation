import torch

from bcsi import diff, mesh


def test_vertex_to_vertex() -> None:
    source = mesh.from_tensors(
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 2]]),
        torch.tensor([[0, 1, 2]]),
    )
    target = mesh.from_tensors(
        torch.tensor([[0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )
    assert torch.equal(diff.vertex_to_vertex(source, target), torch.tensor([1, 1, 2]))


def test_vertex_to_mesh() -> None:
    source = mesh.from_tensors(
        torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 2]]),
        torch.tensor([[0, 1, 2]]),
    )
    target = mesh.from_tensors(
        torch.tensor([[0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0]]),
        torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )
    assert torch.equal(diff.vertex_to_mesh(source, target), torch.tensor([0, 1, 2]))
