import open3d as o3d
import pytest
import torch

from bcsi import submesh


# We define the following meshes inline because we need control of the vertex
# ordering, and open3d does not preserve these when reading from a file.
@pytest.fixture
def pyramid_mesh() -> o3d.geometry.TriangleMesh:
    # This may be duplicated with another fixture test_bps.py because they are
    # used for completely unrelated functionality and shouldn't be coupled.
    vertices = o3d.utility.Vector3dVector(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    triangles = o3d.utility.Vector3iVector(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 4],
            [2, 3, 4],
        ]
    )
    return o3d.geometry.TriangleMesh(vertices, triangles)


@pytest.fixture
def subpyramid_mesh() -> o3d.geometry.TriangleMesh:
    vertices = o3d.utility.Vector3dVector(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    triangles = o3d.utility.Vector3iVector(
        [
            [1, 4, 3],
            [1, 3, 0],
            [1, 0, 4],
            [0, 3, 4],
        ]
    )
    return o3d.geometry.TriangleMesh(vertices, triangles)


@pytest.fixture
def dirty_subpyramid_mesh() -> o3d.geometry.TriangleMesh:
    # This one has an extra vertex not present in the parent.
    vertices = o3d.utility.Vector3dVector(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
        ]
    )
    triangles = o3d.utility.Vector3iVector(
        [
            [1, 4, 3],
            [1, 3, 0],
            [1, 0, 4],
            [0, 3, 4],
        ]
    )
    return o3d.geometry.TriangleMesh(vertices, triangles)


def test_find_vertex_indices(
    pyramid_mesh: o3d.geometry.TriangleMesh, subpyramid_mesh: o3d.geometry.TriangleMesh
) -> None:
    result = submesh.find_vertex_indices(submesh=subpyramid_mesh, parent=pyramid_mesh)
    assert torch.equal(result, torch.tensor([1, 0, 3, 4]))


def test_find_vertex_indices_not_all_vertices_found_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh,
    dirty_subpyramid_mesh: o3d.geometry.TriangleMesh,
) -> None:
    with pytest.raises(ValueError, match="unable to find all vertices"):
        submesh.find_vertex_indices(submesh=dirty_subpyramid_mesh, parent=pyramid_mesh)
