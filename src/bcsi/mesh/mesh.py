"""Utilities for working with Open3D meshes in a torch-focused workflow."""

from functools import cached_property
from typing import Any

import numpy as np
import open3d as o3d
import torch


class TriangleMesh:
    """Triangle mesh representation."""

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        vertex_colors: torch.Tensor | None = None,
    ) -> None:
        """Initialize an instance using an Open3D triangle mesh."""
        if vertex_colors is None:
            vertex_colors = torch.zeros_like(vertices)
        self.update(vertices, triangles, vertex_colors)

    def update(
        self,
        vertices: torch.Tensor | None = None,
        triangles: torch.Tensor | None = None,
        vertex_colors: torch.Tensor | None = None,
    ) -> None:
        """Update triangles and vertices."""
        vertices = self.vertices if vertices is None else vertices.double()
        triangles = self.triangles if triangles is None else triangles.long()
        vertex_colors = (
            self.vertex_colors if vertex_colors is None else vertex_colors.double()
        )
        self._raise_if_invalid(vertices, triangles, vertex_colors)

        self._vertices = vertices
        self._triangles = triangles
        self._vertex_colors = vertex_colors
        self._clear_cache()

    @classmethod
    def from_open3d_legacy(cls, mesh: o3d.geometry.TriangleMesh) -> "TriangleMesh":
        """Construct a `TriangleMesh` instance from a legacy open3d mesh."""
        vertex_colors = (
            _tensor(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        )
        return cls(_tensor(mesh.vertices), _tensor(mesh.triangles), vertex_colors)

    @property
    def vertices(self) -> torch.Tensor:
        """Vertex positions."""
        return self._vertices

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return self.vertices.shape[0]

    @property
    def triangles(self) -> torch.Tensor:
        """Triangles, given as a set of vertex ids."""
        return self._triangles

    @property
    def num_triangles(self) -> int:
        """Number of triangles."""
        return self.triangles.shape[0]

    @property
    def vertex_colors(self) -> torch.Tensor:
        """Vertex positions."""
        return self._vertex_colors

    @staticmethod
    def _raise_if_invalid(
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        vertex_colors: torch.Tensor | None,
    ) -> None:
        """Raise an error if the data is invalid."""
        if vertices.ndim != 2:
            raise ValueError(f"vertices must have 2 dimensions but has {vertices.ndim}")
        if vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n, 3) but has shape {vertices.shape}"
            )
        if vertices.shape[0] <= (max_id := triangles.max().item()):
            raise ValueError(
                "there must be more vertices than the maximum triangle "
                f"vertex id ({max_id}) but there are only {vertices.shape[0]}"
            )

        if triangles.ndim != 2:
            raise ValueError(
                f"triangles must have 2 dimensions but has {triangles.ndim}"
            )
        if triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (n, 3) but has shape {triangles.shape}"
            )
        if triangles.min().item() < 0:
            raise ValueError("triangles must not contain negative ids")

        if vertex_colors is not None and vertex_colors.shape != vertices.shape:
            raise ValueError(
                f"vertex colors ({vertex_colors.shape}) must have the same "
                f"shape as vertices ({vertices.shape})"
            )

    def _clear_cache(self) -> None:
        """Clear cached properties.

        Useful if the underlying mesh is changed.
        """
        self.__dict__.pop("triangle_normals", None)
        self.__dict__.pop("vertex_normals", None)
        self.__dict__.pop("triangle_areas", None)
        self.__dict__.pop("vertex_areas", None)
        self.__dict__.pop("adjacency_list", None)

    @cached_property
    def triangle_areas(self) -> torch.Tensor:
        """The area of each triangle."""
        vi, vj, vk = self.vertices[self.triangles].permute(1, 0, 2)
        a = vj - vi
        b = vk - vi
        return (torch.linalg.vector_norm(torch.linalg.cross(a, b), dim=-1)) / 2

    @cached_property
    def vertex_areas(self) -> torch.Tensor:
        """One third the area of the triangles incident to each vertex."""
        triangle_areas = self.triangle_areas.repeat_interleave(3)
        ids = self.triangles.flatten()
        vertex_areas = torch.zeros(self.num_vertices).double()
        vertex_areas = torch.index_add(vertex_areas, 0, ids, triangle_areas)
        return vertex_areas / 3

    @cached_property
    def triangle_normals(self) -> torch.Tensor:
        """The outward-facing unit normal corresponding to each triangle."""
        vi, vj, vk = self.vertices[self.triangles].permute(1, 0, 2)
        a = vj - vi
        b = vk - vi
        unscaled_normals = torch.linalg.cross(a, b)
        return unscaled_normals / torch.linalg.vector_norm(
            unscaled_normals, dim=-1, keepdim=True
        )

    @cached_property
    def vertex_normals(self) -> torch.Tensor:
        """Vertex normals, calculated as the average of adjacent face normals.

        The face normals are not normalized before taking the average. The
        average is normalized to produce the final result.

        Shape: (num_vertices, 3)
        """
        vi, vj, vk = self.vertices[self.triangles].permute(1, 0, 2)
        a = vj - vi
        b = vk - vi
        unscaled_triangle_normals = torch.linalg.cross(a, b).repeat_interleave(3, dim=0)
        ids = self.triangles.flatten()

        incident_normals = torch.zeros_like(self.vertices)
        incident_normals = torch.index_add(
            incident_normals, 0, ids, unscaled_triangle_normals
        )
        return incident_normals / torch.linalg.vector_norm(
            incident_normals, dim=-1, keepdim=True
        )

    def __add__(self, other: Any) -> "TriangleMesh":  # noqa: ANN401 dynamic typing
        """Add `TriangleMesh` instances by joining their geometries together."""
        if type(other) is not TriangleMesh:
            return NotImplemented

        # Concatenate vertices and triangles, offseting the vertex ids so they
        # don't overlap.
        vertices = torch.cat([self.vertices, other.vertices])
        triangles = torch.cat([self.triangles, other.triangles + self.num_vertices])
        return TriangleMesh(vertices, triangles)

    def open3d_legacy(self) -> o3d.geometry.TriangleMesh:
        """Convert to legacy open3d triangle mesh."""
        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(self.vertices.detach())),
            o3d.utility.Vector3iVector(np.asarray(self.triangles.detach())),
        )
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors.numpy())
        return mesh_o3d

    @cached_property
    def adjacency_list(self) -> list[set[int]]:
        """List of vertex adjacency sets."""
        mesh = self.open3d_legacy()
        mesh.compute_adjacency_list()
        return mesh.adjacency_list

    def merge_close_vertices(self) -> "TriangleMesh":
        """Merge close vertices.

        "Close" vertices are those that are equal when rounded to 5 decimal
        places. Close vertex groups are collapsed to their mean positions.

        Vertex colors are averaged among merged vertices.

        Returns `self`.
        """
        unique, inverses, counts = self.vertices.round(decimals=5).unique(
            dim=0, return_inverse=True, return_counts=True
        )
        new_vertices = torch.zeros_like(unique)
        new_vertices = torch.index_add(new_vertices, 0, inverses, self.vertices)
        new_vertices /= counts[:, None]

        new_triangles = inverses[self.triangles]

        new_vertex_colors = torch.zeros_like(unique)
        new_vertex_colors = torch.index_add(
            new_vertex_colors, 0, inverses, self.vertex_colors
        )
        new_vertex_colors /= counts[:, None]

        self.update(new_vertices, new_triangles, new_vertex_colors)
        return self

    def subdivide_midpoint(self, number_of_iterations: int) -> "TriangleMesh":
        """Subdivide the mesh at edge midpoints.

        Returns a new mesh.
        """
        return self.__class__.from_open3d_legacy(
            self.open3d_legacy().subdivide_midpoint(number_of_iterations)
        )


def _tensor(
    a: o3d.utility.Vector3dVector | o3d.utility.Vector3iVector,
) -> torch.Tensor:
    return torch.tensor(np.asarray(a))


def show(mesh: TriangleMesh) -> None:
    """Visualize a mesh using open3d's visualizer."""
    mesh_o3d = mesh.open3d_legacy()
    o3d.visualization.draw_geometries([mesh_o3d])
