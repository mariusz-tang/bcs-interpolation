"""As-rigid-as-possible shape space metric.

As given in Geometric Modeling in Shape Space:
https://graphics.stanford.edu/~niloy/research/docs/shape_space_sig_07.pdf
"""

from functools import partial

import torch
import torchmin

from bcsi import mesh


def residue(
    rigid_component: torch.Tensor,
    mesh: mesh.TriangleMesh,
    deformation_field: torch.Tensor,
) -> torch.Tensor:
    """Calculate the residue of a deformation field relative to a rigid deformation.

    A deformation field represents a rigid deformation iff it can be expressed as
    X(t) = k(t) + c(t) x p(t). k is the 'constant' term and c is the 'cross
    product term'.

    :param mesh: the mesh being deformed, as it is at time t.
    :param deformation_field: (3 * num_vertices)-length tensor representing a
    deformation field at time t.
    :param rigid_component: six-length tensor representing a rigid transformation.
    The first three components represent the constant term and the last three the
    cross product term.
    """
    p = mesh.vertices.float()
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    rigid_deformation_field = k + torch.linalg.cross(c, p)

    diff = deformation_field - rigid_deformation_field.reshape(-1)
    return torch.linalg.norm(diff)


def raw(mesh: mesh.TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the raw (before regularization) ARAP shape space metric.

    This is simply the minimum residue between the deformation field and rigid
    component, for all possible rigid components.
    """
    result = torchmin.minimize(
        partial(residue, mesh=mesh, deformation_field=deformation_field),
        torch.zeros(6),
        "newton-cg",
    )
    return result.fun


def l2(mesh: mesh.TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the L2 shape space metric regularization term."""
    plain_l2 = torch.linalg.norm(deformation_field.reshape(-1, 3), dim=-1)
    vertex_areas = mesh.trivert_adjacency_matrix.float() @ mesh.triangle_areas
    return torch.sum(plain_l2 * vertex_areas / 3)


def metric(
    mesh: mesh.TriangleMesh, deformation_field: torch.Tensor, lamda: float
) -> torch.Tensor:
    """Calculate the full, regularized ARAP metric."""
    return raw(mesh, deformation_field) + lamda * l2(mesh, deformation_field)
