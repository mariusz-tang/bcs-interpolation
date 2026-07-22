import pytest
import torch

from bcsi import mesh
from bcsi.mesh import submesh


# We define the following meshes inline because we need control of the vertex
# ordering, and open3d does not preserve these when reading from a file.
@pytest.fixture
def pyramid_mesh() -> mesh.TriangleMesh:
    # This may be duplicated with another fixture test_bps.py because they are
    # used for completely unrelated functionality and shouldn't be coupled.
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    triangles = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 4],
            [2, 3, 4],
        ]
    )
    return mesh.from_tensors(vertices, triangles)


@pytest.fixture
def subpyramid_mesh() -> mesh.TriangleMesh:
    vertices = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    triangles = torch.tensor(
        [
            [1, 4, 3],
            [1, 3, 0],
            [1, 0, 4],
            [0, 3, 4],
        ]
    )
    return mesh.from_tensors(vertices, triangles)


@pytest.fixture
def dirty_subpyramid_mesh() -> mesh.TriangleMesh:
    # This one has an extra vertex not present in the parent.
    vertices = torch.tensor(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
        ]
    )
    triangles = torch.tensor(
        [
            [1, 4, 3],
            [1, 3, 0],
            [1, 0, 4],
            [0, 3, 4],
        ]
    )
    return mesh.from_tensors(vertices, triangles)


def test_pair_vertex_correspondences(
    pyramid_mesh: mesh.TriangleMesh, subpyramid_mesh: mesh.TriangleMesh
) -> None:
    result = submesh.Pair(subpyramid_mesh, pyramid_mesh).vertex_correspondences
    assert torch.equal(result, torch.tensor([1, 0, 3, 4]))


def test_pair_vertex_correspondences_not_all_vertices_found_raises(
    pyramid_mesh: mesh.TriangleMesh,
    dirty_subpyramid_mesh: mesh.TriangleMesh,
) -> None:
    with pytest.raises(ValueError, match="unable to find all vertices"):
        _ = submesh.Pair(dirty_subpyramid_mesh, pyramid_mesh).vertex_correspondences
