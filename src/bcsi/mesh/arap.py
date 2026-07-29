"""As-rigid-as-possible shape space metric.

As given in Geometric Modeling in Shape Space:
https://graphics.stanford.edu/~niloy/research/docs/shape_space_sig_07.pdf
"""

import torch

from . import TriangleMesh


def residue(
    rigid_component: torch.Tensor,
    mesh: TriangleMesh,
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
    p = mesh.vertices
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    rigid_deformation_field = k + torch.linalg.cross(c, p)

    diff = deformation_field - rigid_deformation_field.reshape(-1)
    return torch.linalg.norm(diff)


def _jacobian_of_residue(
    rigid_component: torch.Tensor,
    mesh: TriangleMesh,
    deformation_field: torch.Tensor,
) -> torch.Tensor:
    p = mesh.vertices
    k = rigid_component[:3]
    c = rigid_component[None, 3:]
    x = deformation_field.reshape(-1, 3)

    diff = k + torch.linalg.cross(c, p) - x

    v_primes = torch.eye(3).double()
    dr_dk = 2 * torch.einsum("id,vd->i", v_primes, diff)

    num_vertices = p.shape[0]
    cp_prime = torch.zeros(num_vertices, 3, 3).double()
    ids = torch.arange(num_vertices)
    cp_prime[ids] = torch.linalg.cross(
        v_primes.unsqueeze(0).repeat(num_vertices, 1, 1), p[ids, None]
    )
    dr_dc = 2 * torch.einsum("vid,vd->i", cp_prime, diff)
    return torch.cat([dr_dk, dr_dc])


def _hessian_of_residue(
    mesh: TriangleMesh,
) -> torch.Tensor:
    p = mesh.vertices

    v_primes = torch.eye(3).double()

    num_vertices = p.shape[0]
    cp_prime = torch.zeros(num_vertices, 3, 3).double()
    ids = torch.arange(num_vertices)
    cp_prime[ids] = torch.linalg.cross(
        v_primes.unsqueeze(0).repeat(num_vertices, 1, 1), p[ids, None]
    )

    hessian = torch.zeros(6, 6).double()
    hessian[:3, :3] = torch.eye(3) * 2 * num_vertices
    hessian[:3, 3:] = torch.einsum("id,vjd->ij", v_primes, cp_prime)
    hessian[3:, :3] = hessian[:3, 3:].T
    hessian[3:, 3:] = torch.einsum("vid,vjd->ij", cp_prime, cp_prime)
    return hessian


def raw(mesh: TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the raw (before regularization) ARAP shape space metric.

    This is simply the minimum residue between the deformation field and rigid
    component, for all possible rigid components. The minimum is calculated using
    the Newton-Raphson method.

    This operation supports recording gradients for autograd.
    """
    # Initialize at the mean translation with no rotation.
    avg = deformation_field.mean()
    x = torch.zeros(6).double()
    x[:3] = avg
    gamma = 0.9

    hess = _hessian_of_residue(mesh).inverse()

    # Implement the method manually so we keep gradient information.
    while True:
        x_prev = x
        x = x - gamma * hess @ _jacobian_of_residue(x, mesh, deformation_field)
        if torch.linalg.vector_norm(x - x_prev) < 1e-6:
            return residue(x, mesh, deformation_field)


def l2(mesh: TriangleMesh, deformation_field: torch.Tensor) -> torch.Tensor:
    """Calculate the L2 shape space metric regularization term."""
    plain_l2 = torch.linalg.norm(deformation_field.reshape(-1, 3), dim=-1)
    return torch.sum(plain_l2 * mesh.vertex_areas)


def metric(
    mesh: TriangleMesh, deformation_field: torch.Tensor, lamda: float = 0.001
) -> torch.Tensor:
    """Calculate the full, regularized ARAP metric."""
    return raw(mesh, deformation_field) + lamda * l2(mesh, deformation_field)
