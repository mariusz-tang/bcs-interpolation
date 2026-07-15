"""As-rigid-as-possible shape space metric.

As given in Geometric Modeling in Shape Space:
https://graphics.stanford.edu/~niloy/research/docs/shape_space_sig_07.pdf
"""

import torch

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
