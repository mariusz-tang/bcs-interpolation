import pytest
import torch

from bcsi import arap, mesh


@pytest.mark.parametrize(
    "rigid_component",
    [
        torch.tensor([1, 1, 1, 1, 1, -1]).float(),
        torch.tensor([2, 1, 2, 1, 2, 1]).float(),
        torch.tensor([1, 1, -1, 1, -1, 1]).float(),
        torch.tensor([1, -2, 1, 1, 0, 1]).float(),
    ],
)
def test_residue_zero_on_rigid_deformation_with_corresponding_rigid_component(
    cube_mesh: mesh.TriangleMesh,
    rigid_component: torch.Tensor,
) -> None:
    p = cube_mesh.vertices.float()
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    deformation_field = k + torch.linalg.cross(c, p)

    assert arap.residue(rigid_component, cube_mesh, deformation_field.reshape(-1)) == 0
