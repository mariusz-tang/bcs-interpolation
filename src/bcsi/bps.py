"""Defines the blended polynomial chart surface class."""

import logging
from functools import cached_property

import numpy as np
import open3d as o3d
import torch

from bcsi import polynomial, triangle

logger = logging.getLogger(__name__)


class BlendedPolynomialSurface:
    """Represents a blended chart surface using polynomial patches."""

    coefficients: torch.Tensor
    """(vertex, dimension, coefficient)"""

    def __init__(
        self,
        proxy_mesh: o3d.geometry.TriangleMesh,
        degree: int,
        scale: float = 0.5,
        coefficients: torch.Tensor | None = None,
        beta: float = 0.73,
    ) -> None:
        """Initialize a BPS instance.

        The patches are initialized to planes with unit gradient in each
        direction.

        :param coefficients: if specified and not `None`, should be a tensor of
        coefficients for one vertex (3, num_coeffs) or for all vertices
        (num_vertices, 3, num_coeffs). By default, each patch function is set to
        (x,y) -> (x,y,0) if `degree` is at least 1.
        """
        logger.info("BPS initialization start")
        self.proxy_mesh = proxy_mesh

        # Convert key mesh properties to tensors for later reuse.
        self.triangles = torch.tensor(np.asarray(proxy_mesh.triangles))
        self.vertices = torch.tensor(np.asarray(proxy_mesh.vertices))
        if not proxy_mesh.has_vertex_normals():
            proxy_mesh.compute_vertex_normals()
        self.vertex_normals = torch.tensor(np.asarray(proxy_mesh.vertex_normals))

        # Ensure adjacency list is available.
        if not self.proxy_mesh.has_adjacency_list():
            self.proxy_mesh.compute_adjacency_list()

        self.num_vertices = self.vertices.shape[0]
        self.num_triangles = self.triangles.shape[0]

        self.degree = degree
        num_coeffs = polynomial.num_coeffs(self.degree)

        if coefficients is None:
            # If no coefficients are given, initialize to zeros for degree 0...
            self.coefficients = torch.zeros((self.num_vertices, 3, num_coeffs)).float()
            if self.degree >= 1:
                # ...or xy planes for degree at least 1.
                self.coefficients[:, 0, 1] = 1
                self.coefficients[:, 1, 2] = 1
        elif coefficients.shape == (3, num_coeffs):
            # If coefficients are given for a single vertex, use them for all vertices.
            self.coefficients = torch.stack([coefficients] * self.num_vertices).float()
        elif coefficients.shape == (self.num_vertices, 3, num_coeffs):
            # If coefficients are given for all vertices, use them all.
            self.coefficients = coefficients.float()
        else:
            raise ValueError(
                f"invalid shape for coefficients {coefficients.shape} for "
                f"{self.__class__.__name__} with {self.num_vertices} proxy "
                f"vertices and degree {self.degree}"
            )

        self.global_scale = scale

        if not 0 < beta < 1:
            raise ValueError(f"expected beta in range (0,1), received: {beta}")
        self.beta = beta
        logger.info("BPS initialization end")

    @cached_property
    def vertex_scales(self) -> torch.Tensor:
        """Local scale factor at each vertex.

        The local scale at a vertex is proportional to the mean length of the
        edges incident to that vertex.

        Note that the calculation is incorrect for boundary vertices because it
        assumes that every edge is connected to two faces.

        Shape: (num_vertices)
        """
        logger.info("vertex_scales begin")
        # Accumulate edge lengths and counts.
        edge_lengths = torch.zeros(self.num_vertices)
        edge_counts = torch.zeros(self.num_vertices)

        # For each vertex of a face.
        for i in range(3):
            # Find the next vertex.
            j = (i + 1) % 3

            vi = self.triangles[:, i]
            vj = self.triangles[:, j]

            # Calculate the edge length.
            length = torch.linalg.norm(
                self.vertices[vi] - self.vertices[vj], dim=1
            ).float()

            # Update the accumulators.
            edge_lengths.index_add_(0, vi, length)
            edge_lengths.index_add_(0, vj, length)
            edge_counts.index_add_(0, vi, torch.ones_like(length))
            edge_counts.index_add_(0, vj, torch.ones_like(length))

        mean_edge_length = edge_lengths / edge_counts

        logger.info("vertex_scales end")
        return mean_edge_length * self.global_scale

    @cached_property
    def vertex_rotations(self) -> torch.Tensor:
        """Local rotation matrix associated with each vertex.

        The local rotation matrix for a vertex orients the positive z-axis
        towards the vertex normal. The x-axis gets mapped to the edge
        corresponding to the neighbour with the lowest id.

        Shape: (num_vertices, 3, 3)
        """
        logger.info("vertex_rotations begin")
        rotations = torch.zeros((self.num_vertices, 3, 3))

        for vertex_id in range(self.num_vertices):
            vertex = self.vertices[vertex_id]
            normal = self.vertex_normals[vertex_id]

            # Select the neighbour with the lowest id.
            neighbour_id = min(self.proxy_mesh.adjacency_list[vertex_id])
            neighbour = self.vertices[neighbour_id]

            # Project the edge onto the tangent plane and normalize to a direction.
            neighbour_direction = neighbour - vertex
            neighbour_direction -= normal * torch.dot(normal, neighbour_direction)
            neighbour_direction /= torch.linalg.norm(neighbour_direction)

            # Map +z to the normal, and +x to the projected neighbour direction.
            rotations[vertex_id, :, 0] = neighbour_direction
            rotations[vertex_id, :, 1] = torch.linalg.cross(normal, neighbour_direction)
            rotations[vertex_id, :, 2] = normal

        logger.info("vertex_rotations end")
        return rotations

    def evaluate_patch(self, vertex_id: int, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the patch function for `vertex_id` at `coordinates`.

        :param coordinates: should have shape (num_coordinates, 2).

        :returns: the evaluated coordinates as a tensor of shape (num_coordinates, 3).
        """
        if not -self.num_vertices <= vertex_id <= self.num_vertices - 1:
            raise ValueError(
                f"vertex_id {vertex_id} out of range for "
                f"{self.__class__.__name__} with {self.num_vertices} proxy vertices"
            )

        if coordinates.dim() != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"invalid shape for coordinates {coordinates.shape}, expected (N, 2)"
            )

        coefficients = self.coefficients[vertex_id]

        x = coordinates[:, 0]
        y = coordinates[:, 1]
        basis = polynomial.basis(x, y, self.degree).float()

        result_local = torch.einsum("dc,pc->pd", coefficients, basis)
        rotation_matrix = self.vertex_rotations[vertex_id]

        result_rotated = torch.einsum("ij,pj->pi", rotation_matrix, result_local)
        vertex = self.vertices[vertex_id]
        scale = self.vertex_scales[vertex_id]

        return (scale * result_rotated + vertex).float()

    @cached_property
    def triangle_onering_indices(self) -> torch.Tensor:
        """Indices of each triangle in each one ring it is a part of.

        Triangles are ordered anticlockwise from the edge to the neighbour
        vertex with the lowest id.

        Shape: (num_triangles, 3)
        The second coordinate corresponds the perspectives from each vertex of
        the triangle.
        """
        return self._triangle_onering_indices_and_flips[0]

    @cached_property
    def triangle_onering_flips(self) -> torch.Tensor:
        """Whether or not each triangle is flipped w.r.t. each one ring it's in.

        Triangles are ordered anticlockwise from the edge to the neighbour
        vertex with the lowest id.

        Each element is 1 or -1.
        1 indicates not flipped.
        -1 indicates flipped.

        If a triangle is flipped for a given one-ring, that means the angle
        within that triangle for that one-ring should be measured from the
        opposite edge.

        Shape: (num_triangles, 3)
        The second coordinate corresponds the perspectives from each vertex of
        the triangle.
        """
        return self._triangle_onering_indices_and_flips[1]

    @cached_property
    def _triangle_onering_indices_and_flips(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Onering indices and flips for each triangle.

        Access these values via `self.triangle_onering_indices` and
        `self.triangle_onering_flips`.

        Implemented in this way because the methods for computing these values
        are very closely related.
        """
        logger.info("onerings begin")
        onering_indices = torch.zeros((self.num_triangles, 3))
        onering_flips = torch.zeros_like(onering_indices)
        halfedge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
            self.proxy_mesh
        )

        # For each vertex...
        for vertex_id in range(self.num_vertices):
            ordered_halfedges = list(
                halfedge_mesh.ordered_half_edge_from_vertex[vertex_id]
            )

            # ...shift the halfedge list so the edge corresponding to the
            # lowest neighbour id vertex is first...
            lowest_neighbour_id = min(self.proxy_mesh.adjacency_list[vertex_id])
            for i, halfedge_id in enumerate(ordered_halfedges):
                if (
                    halfedge_mesh.half_edges[halfedge_id].vertex_indices[1]
                    == lowest_neighbour_id
                ):
                    num_halfedges = len(ordered_halfedges)
                    (
                        ordered_halfedges[: num_halfedges - i],
                        ordered_halfedges[num_halfedges - i :],
                    ) = ordered_halfedges[i:], ordered_halfedges[:i]
                    break

            # ...then go through the one-ring, assigning indices and flips to
            # triangles as we go.
            for i, halfedge_id in enumerate(ordered_halfedges):
                halfedge = halfedge_mesh.half_edges[halfedge_id]
                triangle_id = halfedge.triangle_index
                # Find the index of the current vertex in the current triangle.
                triangle_vertices = list(self.triangles[triangle_id])
                tri_vert_id = triangle_vertices.index(vertex_id)

                # Assign the current index to the triangle.
                onering_indices[triangle_id, tri_vert_id] = i

                # Check that the centre of the one-ring and the next vertex
                # appear in the right order in the current triangle, and mark
                # the triangle as flipped if not.
                next_tri_vert_id = triangle_vertices.index(halfedge.vertex_indices[1])
                if (next_tri_vert_id - tri_vert_id) % 3 == 1:
                    onering_flips[triangle_id, tri_vert_id] = 1
                else:
                    onering_flips[triangle_id, tri_vert_id] = -1

        logger.info("onerings end")
        return onering_indices, onering_flips

    @cached_property
    def _valences(self) -> torch.Tensor:
        """Valence of each vertex in the proxy mesh."""
        return torch.tensor(list(map(len, self.proxy_mesh.adjacency_list)))

    def get_onering_coordinates(
        self, triangle_id: int, vertices: torch.Tensor
    ) -> torch.Tensor:
        """Convert from cartesian to onering coordinates on a given triangle.

        The input vertices should be relative to the 'canonical' equilateral
        triangle with vertices at (0,0), (1,0), and (0.5, sqrt(3)), in that
        order. Triangles on the proxy have their ordered vertices mapped to the
        canonical triangle vertices in the same order.

        The input should have shape (..., 2). The output will have shape
        (..., 3, 2). The second-to-last coordinate represents the vertex at the
        centre of each one-ring, and the last coordinate corresponds to the r
        and theta values for each input vertex.
        """
        if not -self.num_triangles <= triangle_id < self.num_triangles:
            raise ValueError(
                f"triangle_id {triangle_id} out of range for "
                f"{self.__class__.__name__} with {self.num_triangles} proxy "
                "triangles"
            )

        radii = triangle.distances(vertices)

        local_angles = triangle.angles(vertices)
        oriented_angles = local_angles.where(
            self.triangle_onering_flips[triangle_id] == 1,
            other=torch.pi / 3 - local_angles,
        )
        angles_before_flattening = (
            oriented_angles + torch.pi / 3 * self.triangle_onering_indices[triangle_id]
        )
        angles = (
            angles_before_flattening / self._valences[self.triangles[triangle_id]] * 6
        )

        return torch.stack([radii, angles], dim=-1)

    def get_unblended_patch_vertices(
        self, triangle_id: int, vertices: torch.Tensor
    ) -> torch.Tensor:
        """Convert from cartesian coordinates to unblended patches.

        :param triangle_id: specifies which triangle on the proxy mesh we are
        mapping from.

        The input vertices should be relative to the 'canonical' equilateral
        triangle with vertices at (0,0), (1,0), and (0.5, sqrt(3)), in that
        order. Triangles on the proxy have their ordered vertices mapped to the
        canonical triangle vertices in the same order.

        The input should have shape (num_vertices, 2). The output will have shape
        (3, num_vertices, 3). The first coordinate represents the vertex at the
        centre of each one-ring, and the last coordinate corresponds to the 3D
        cartesian coordinates of the output.
        """
        onering_coords = self.get_onering_coordinates(triangle_id, vertices)
        r = onering_coords[..., 0]
        theta = onering_coords[..., 1]

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        basis = polynomial.basis(x, y, self.degree).float()

        origin_vertex_ids = self.triangles[triangle_id]

        # The einsum indices represent:
        # p: perspective (vertex at the centre of one-ring)
        # d: dimension (output dimension x/y/z)
        # c: coefficient
        # v: input vertex
        # i: row
        # j: column
        coefficients = self.coefficients[origin_vertex_ids]
        result_local = torch.einsum("pdc,vpc->vpd", coefficients, basis)
        rotation_matrix = self.vertex_rotations[origin_vertex_ids]

        result_rotated = torch.einsum("pij,vpj->pvi", rotation_matrix, result_local)
        origin_vertices = self.vertices[origin_vertex_ids]
        scale = self.vertex_scales[origin_vertex_ids]

        return (
            scale[:, torch.newaxis, torch.newaxis] * result_rotated
            + origin_vertices[:, torch.newaxis, :]
        ).float()

    def get_blended_patch_vertices(
        self, triangle_id: int, vertices: torch.Tensor
    ) -> torch.Tensor:
        """Convert from cartesian coordinates to blended patches.

        :param triangle_id: specifies which triangle on the proxy mesh we are
        mapping from.

        The input vertices should be relative to the 'canonical' equilateral
        triangle with vertices at (0,0), (1,0), and (0.5, sqrt(3)), in that
        order. Triangles on the proxy have their ordered vertices mapped to the
        canonical triangle vertices in the same order.

        The input should have shape (num_vertices, 2). The output will have shape
        (num_vertices, 3). The last coordinate corresponds to the 3D cartesian
        coordinates of the output.
        """
        logger.info(f"blended patch vertices for triangle {triangle_id}")
        unblended_coords = self.get_unblended_patch_vertices(triangle_id, vertices)
        blend_coefficients = triangle.blend_coefficients(vertices, self.beta).float()
        return torch.einsum("pvd,vp->vd", unblended_coords, blend_coefficients)
