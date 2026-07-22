import pytest
import torch

from bcsi import mesh
from bcsi.mesh import arap


@pytest.mark.parametrize(
    "rigid_component",
    [
        torch.tensor([1, 1, 1, 1, 1, -1]).double(),
        torch.tensor([2, 1, 2, 1, 2, 1]).double(),
        torch.tensor([1, 1, -1, 1, -1, 1]).double(),
        torch.tensor([1, -2, 1, 1, 0, 1]).double(),
    ],
)
def test_residue_zero_on_rigid_deformation_with_corresponding_rigid_component(
    cube_mesh: mesh.TriangleMesh,
    rigid_component: torch.Tensor,
) -> None:
    p = cube_mesh.vertices
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    deformation_field = k + torch.linalg.cross(c, p)

    assert arap.residue(rigid_component, cube_mesh, deformation_field.reshape(-1)) == 0


@pytest.mark.parametrize(
    "rigid_component",
    [
        torch.tensor([1, 1, 1, 1, 1, -1]).double(),
        torch.tensor([2, 1, 2, 1, 2, 1]).double(),
        torch.tensor([1, 1, -1, 1, -1, 1]).double(),
        torch.tensor([1, -2, 1, 1, 0, 1]).double(),
    ],
)
def test_raw_zero_on_rigid_deformation(
    cube_mesh: mesh.TriangleMesh,
    rigid_component: torch.Tensor,
) -> None:
    p = cube_mesh.vertices
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    deformation_field = k + torch.linalg.cross(c, p)

    assert arap.raw(cube_mesh, deformation_field.reshape(-1)) == pytest.approx(
        0, abs=1e-6
    )


def test_l2() -> None:
    mesh_ = mesh.from_tensors(
        torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]]), torch.tensor([[0, 1, 2]])
    )
    deformation_field = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0, 0]).double()

    assert arap.l2(mesh_, deformation_field) == pytest.approx(0.5)
