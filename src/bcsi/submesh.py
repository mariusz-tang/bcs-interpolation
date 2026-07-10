"""Utilities for working with meshes whose vertices are a subset of another's.

We refer to the submeshes as 'children' and the original meshes as 'parents'.
"""

from functools import cached_property

import open3d as o3d
import torch

from bcsi import bps, mesh


class Pair:
    """Represents a parent-child submesh pair."""

    def __init__(self, child: mesh.TriangleMesh, parent: mesh.TriangleMesh) -> None:
        """Initialize a parent-child pair.

        `child` should be a mesh whose set of vertices is a subset of `parent`'s.
        """
        self.child = child
        self.parent = parent

    @cached_property
    def vertex_correspondences(self) -> torch.Tensor:
        """Calculate indices of vertices found in `child` relative to `parent`.

        Returns a (num_vertices,) shape tensor where the `i`th entry represents
        the index of the `i`th vertex in `child`, in the vertex list of `parent`.
        """
        num_parent_verts = self.parent.vertices.shape[0]

        # Repeat the sub-vertices so we can compare them against ALL vertices in
        # the parent by broadcasting.
        matches = (
            self.child.vertices.repeat(num_parent_verts, 1, 1)
            .transpose(0, 1)
            .isclose(self.parent.vertices)
        )
        matches_complete_vertex = matches.all(dim=2)

        result = matches_complete_vertex.argwhere()[:, 1]

        if result.shape[0] != self.child.vertices.shape[0]:
            raise ValueError(
                f"unable to find all vertices from {self.child.vertices} in "
                f"{self.parent.vertices}"
            )

        return result


def create_submesh(parent: mesh.TriangleMesh, scale: float) -> mesh.TriangleMesh:
    """Create a suitable 'child' submesh from a `parent` mesh.

    :param scale: a float between 0 and 1 (exclusive) which represents the
    target number of triangles in the result, as a fraction of the number of
    triangles in `parent`.

    This function works by decimating the parent, snapping the resulting
    vertices to vertices in the parent, and then merging duplicate vertices.
    """
    if not 0 < scale < 1:
        raise ValueError(
            f"scale should be between 0 and 1 (exclusive), received {scale}"
        )

    child_unaligned = o3d.t.geometry.TriangleMesh.from_legacy(
        parent.open3d.simplify_quadric_decimation(int(parent.num_triangles * scale))
    )
    parent_verts = o3d.core.Tensor(parent.vertices.float().numpy())
    nns = o3d.core.nns.NearestNeighborSearch(parent_verts)
    nns.knn_index()
    closest_point_ids, _ = nns.knn_search(child_unaligned.vertex.positions, 1)

    closest_points = parent_verts[closest_point_ids[:, 0]]
    child_o3d = o3d.t.geometry.TriangleMesh(
        closest_points, child_unaligned.triangle.indices
    )
    return mesh.TriangleMesh(child_o3d.to_legacy()).merge_close_vertices(eps=1e-6)


def new_frame(
    pair: Pair,
    new_parent: mesh.TriangleMesh,
) -> Pair:
    """Return a new mesh obtained by posing a child mesh according to a new parent."""
    new_vertices = new_parent.vertices[pair.vertex_correspondences]
    new_child = mesh.from_tensors(new_vertices, pair.child.triangles)
    new_pair = Pair(new_child, new_parent)
    # Transfer vertex correspondences since they will be the same.
    new_pair.vertex_correspondences = pair.vertex_correspondences
    return new_pair


def create_bps_degree_one(
    pair: Pair,
    degree: int,
    scale: float,
    beta: float,
) -> bps.BlendedPolynomialSurface:
    """Create a BPS from the child of a submesh pair using data from the parent.

    The patch functions are unit planes whose normals are determined by the
    normals at each vertex of the child in the parent.
    """
    base_surface = bps.BlendedPolynomialSurface(pair.child, degree, scale, beta=beta)

    # Map parent vertex normals to child patch function space.
    normals = torch.einsum(
        # Multiply by the transpose instead of explicitly calculating inverse rotations.
        "vji,vj->vi",
        base_surface.vertex_rotations.double(),
        pair.parent.vertex_normals[pair.vertex_correspondences],
    )

    # Project x-direction onto normal plane.
    e_1 = torch.tensor([1, 0, 0])
    e_1_projected = e_1 - normals[:, 0:1] * normals
    e_1_normalized = e_1_projected / torch.linalg.vector_norm(
        e_1_projected, dim=1, keepdim=True
    )

    # Calculate y-direction by cross product.
    e_2 = torch.linalg.cross(normals, e_1_normalized)

    # Create coefficients matrix.
    coefficients = torch.zeros_like(base_surface.coefficients)
    coefficients[..., 1] = e_1_normalized
    coefficients[..., 2] = e_2

    # Create BPS with the new coefficients.
    return bps.BlendedPolynomialSurface(pair.child, degree=1, coefficients=coefficients)
