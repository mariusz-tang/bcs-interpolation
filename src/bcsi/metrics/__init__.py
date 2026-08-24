"""Riemannian metrics in shape space.

https://graphics.stanford.edu/~niloy/research/docs/shape_space_sig_07.pdf
"""

import torch

from bcsi import mesh

from . import _arap


def l2(mesh: mesh.TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the L2 shape space metric regularization term."""
    plain_l2 = torch.linalg.norm(deformation_field.reshape(-1, 3), dim=-1)
    return torch.sum(plain_l2 * mesh.vertex_areas)


arap = _arap.raw
arap_residue = _arap.residue


def arap_regularized(
    mesh: mesh.TriangleMesh, deformation_field: torch.Tensor, lamda: float = 1e-6
) -> torch.Tensor:
    """Calculate the full, regularized ARAP metric."""
    return arap(mesh, deformation_field) + lamda * l2(mesh, deformation_field)


def aiap(mesh: mesh.TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the raw (before regularization) AIAP shape space metric."""
    vertex_deformations = deformation_field.reshape(-1, 3)

    # Vertices of each triangle.
    vi, vj, vk = mesh.vertices[mesh.triangles].permute(1, 0, 2)
    xi, xj, xk = vertex_deformations[mesh.triangles].permute(1, 0, 2)

    return (
        (
            torch.einsum("vd,vd->v", xi - xj, vi - vj) ** 2
            + torch.einsum("vd,vd->v", xi - xk, vi - vk) ** 2
            + torch.einsum("vd,vd->v", xj - xk, vj - vk) ** 2
        ).sum()
        / 2
        * mesh.num_triangles
    )


def aiap_regularized(
    mesh: mesh.TriangleMesh, deformation_field: torch.Tensor, lamda: float = 1e-3
) -> torch.Tensor:
    """Calculate the full, regularized ARAP metric."""
    return aiap(mesh, deformation_field) + lamda * l2(mesh, deformation_field)


__all__ = [
    "arap",
    "arap_regularized",
    "arap_residue",
    "aiap",
    "aiap_regularized",
    "l2",
    "Metric",
]
