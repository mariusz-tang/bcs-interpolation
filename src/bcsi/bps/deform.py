"""Conversion from deformation fields in BPS space to shape space.

Deformations are assumed to be linear between frames.
"""

import math

import numpy as np
import scipy.optimize
import torch

from bcsi import mesh

from . import BlendedPolynomialSurface, polynomial, render, triangle


def energy(
    start: BlendedPolynomialSurface,
    finish: BlendedPolynomialSurface,
    resolution: int,
    num_frames: int = 2,
) -> torch.Tensor:
    """Calculate BPS deformation energy.

    :param start: BPS at the start of the deformation.
    :param finish: BPS at the end of the deformation.
    :param resolution: resolution at which to render the BPS when calculating
    shape-space metrics.
    :param num_frames: the total number of frames at which to take the metric,
    including `start` and `finish`. Must be at least 2.
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be at least 2 but was {num_frames}")

    total = torch.tensor(0).double()

    current = mesh.arap.metric(*bps_to_shape_space(start, finish, 0, resolution))

    for i in range(num_frames - 1):
        t = (1 + i) / (num_frames - 1)
        total += current
        current = mesh.arap.metric(*bps_to_shape_space(start, finish, t, resolution))
        total += current

    return total


def make_frame(
    start: BlendedPolynomialSurface, finish: BlendedPolynomialSurface, t: float
) -> BlendedPolynomialSurface:
    """Construct an intermediate BPS by linear interpolation."""
    dv_dt = finish.proxy.vertices - start.proxy.vertices
    dcoeffs_dt = finish.coefficients - start.coefficients

    # Construct frame BPS.
    proxy = mesh.TriangleMesh(start.proxy.vertices + t * dv_dt, start.proxy.triangles)
    frame = BlendedPolynomialSurface(
        proxy,
        start.degree,
        start.global_scale,
        start.coefficients + t * dcoeffs_dt,
        start.beta,
    )

    # Transfer computationally-expensive data which is needed for rendering step.
    frame.triangle_onering_flips = start.triangle_onering_flips
    frame.triangle_onering_indices = start.triangle_onering_indices

    return frame


def bps_to_shape_space(
    start: BlendedPolynomialSurface,
    finish: BlendedPolynomialSurface,
    t: float,
    resolution: int,
) -> tuple[mesh.TriangleMesh, torch.Tensor]:
    """Convert linear BPS deformation to shape space deformation.

    :param start: BPS at the start of the deformation.
    :param finish: BPS at the end of the deformation.
    :param t: The current time, used to construct the BPS at the current 'frame'.
    :param resolution: Number of subdivisions to apply to the triangular patch
    representing each face in the proxy mesh.

    :returns: The frame BPS, rendered at `resolution`, with the corresponding
    deformation field tensor.
    """
    frame = make_frame(start, finish, t)

    patch = render.triangle_patch(resolution)
    # Ignore the z coordinate, which is zero everywhere.
    patch_coordinates = patch.vertices[:, :2]
    # Calculate all vertex positions and flatten the result.
    vertices = frame.get_blended_patch_vertices(patch_coordinates).reshape(-1, 3)

    # Duplicate the topology tensor for each patch, increasing the vertex indices
    # by the number of vertices per patch each time. Finally, flatten the result.
    triangles = (
        patch.triangles.tile(frame.proxy.num_triangles, 1, 1)
        + torch.ones(frame.proxy.num_triangles, patch.num_triangles, 3)
        * torch.arange(frame.proxy.num_triangles)[:, None, None]
        * patch.num_vertices
    ).reshape(-1, 3)

    # Construct the rendered mesh
    rendered_mesh = mesh.TriangleMesh(vertices, triangles)

    # Calculate deformation field and flatten the result.
    dv_dt = finish.proxy.vertices - start.proxy.vertices
    dcoeffs_dt = finish.coefficients - start.coefficients
    dp_dt = blended_patch_derivatives(
        dv_dt, dcoeffs_dt, frame, patch_coordinates
    ).reshape(-1, 3)

    # Store the deformation field in the vertex colors. This is so we can
    # associate each vertex with its deformation vector before merging vertices.
    rendered_mesh.update(vertex_colors=dp_dt)
    rendered_mesh.merge_close_vertices()

    # Extract the deformation field.
    deformation_field = rendered_mesh.vertex_colors.flatten()

    # Merge the patches into one cohesive mesh.
    return rendered_mesh, deformation_field


def blended_patch_derivatives(
    dv_dt: torch.Tensor,
    dcoeffs_dt: torch.Tensor,
    frame: BlendedPolynomialSurface,
    vertices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate patch derivates between BPS meshes at specified vertices.

    This function is equivalent to
    `BlendedPolynomialSurface.get_blended_patch_vertices()`, where the
    patches are substituted for their derivatives with respect to time.
    """
    unblended = unblended_patch_derivatives(dv_dt, dcoeffs_dt, frame, vertices)
    blend_coefficients = triangle.blend_coefficients(vertices, frame.beta).double()
    return torch.einsum("tpvd,vp->tvd", unblended, blend_coefficients)


def unblended_patch_derivatives(
    dv_dt: torch.Tensor,
    dcoeffs_dt: torch.Tensor,
    frame: BlendedPolynomialSurface,
    vertices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate patch derivates between BPS meshes at specified vertices.

    This function is equivalent to
    `BlendedPolynomialSurface.get_unblended_patch_vertices()`, where the
    patches are substituted for their derivatives with respect to time.
    """
    x, y = frame.get_onering_coordinates(vertices)
    basis = polynomial.basis(x, y, frame.degree)

    origin_vertex_ids = frame.proxy.triangles

    # The einsum indices represent:
    # t: triangle
    # p: perspective (vertex at the centre of one-ring)
    # d: dimension (output dimension x/y/z)
    # c: coefficient
    # v: input vertex
    # i: row
    # j: column
    coefficients = frame.coefficients[origin_vertex_ids]
    m = torch.einsum("tpdc,tvpc->tvpd", coefficients, basis)  # tvpd
    dm_dt = patch_derivatives_function(dcoeffs_dt, frame, vertices)  # tvpd

    r_ = frame.vertex_rotations[origin_vertex_ids]  # tpij
    dr_dt = vertex_rotations_derivative(dv_dt, frame.proxy)[origin_vertex_ids]  # tpij

    s = frame.vertex_scales[origin_vertex_ids, None, None]  # tv
    ds_dt = vertex_scales_derivative(dv_dt, frame)[origin_vertex_ids, None, None]  # tv

    ds_term = ds_dt * torch.einsum("tpij,tvpj->tpvi", r_, m)
    dr_term = s * torch.einsum("tpij,tvpj->tpvi", dr_dt, m)
    dm_term = s * torch.einsum("tpij,tvpj->tpvi", r_, dm_dt)
    dv_term = dv_dt[origin_vertex_ids][:, :, None, :]

    return ds_term + dr_term + dm_term + dv_term


def patch_derivatives_function(
    dcoeffs_dt: torch.Tensor,
    frame: BlendedPolynomialSurface,
    vertices: torch.Tensor,
) -> torch.Tensor:
    """Get a function to evaluate patch derivates between BPS meshes.

    This function is equivalent to
    `BlendedPolynomialSurface.get_unblended_patch_vertices()`, where the
    patches are substituted for their derivatives with respect to time instead,
    and the patches are not transformed.

    :param dcoeffs_dt: rate of change of patch coefficients over time at time t.
    :param frame: BPS at time t.
    """
    origin_vertex_ids = frame.proxy.triangles
    dcoeff_dt = dcoeffs_dt[origin_vertex_ids]

    x, y = frame.get_onering_coordinates(vertices)
    basis = polynomial.basis(x, y, frame.degree)

    # The einsum indices represent:
    # t: triangle
    # p: perspective (vertex at the centre of one-ring)
    # d: dimension (output dimension x/y/z)
    # c: coefficient
    # v: input vertex
    return torch.einsum("tpdc,tvpc->tvpd", dcoeff_dt, basis)


def vertex_scales_derivative(
    dv_dt: torch.Tensor, frame: BlendedPolynomialSurface
) -> torch.Tensor:
    """Evaluate the rate of change of vertex scales over time.

    The implementation is based on the related function
    `BlendedPolynomialSurface.vertex_scales`.

    :param dv_dt: rate of change of proxy vertex positions over time at time t.
    :param frame: BPS at time t.
    """
    # Accumulate edge lengths and counts.
    edge_length_derivatives = torch.zeros(frame.proxy.num_vertices).double()
    edge_counts = torch.zeros_like(edge_length_derivatives)

    # For each vertex of a face.
    for i in range(3):
        # Find the next vertex.
        j = (i + 1) % 3

        vi = frame.proxy.triangles[:, i]
        vj = frame.proxy.triangles[:, j]
        ids = torch.cat([vi, vj])

        # Calculate the edge length derivative.
        u = frame.proxy.vertices[vi] - frame.proxy.vertices[vj]
        du_dt = dv_dt[vi] - dv_dt[vj]
        dmag_u_dt = _derivative_of_norm(u, du_dt).flatten()

        # Repeat the tensor to match the full list of indexes `ids`.
        dmag_u_dt = dmag_u_dt.repeat(2)

        # Update the accumulators.
        edge_length_derivatives = torch.index_add(
            edge_length_derivatives, 0, ids, dmag_u_dt
        )
        edge_counts = torch.index_add(edge_counts, 0, ids, torch.ones_like(dmag_u_dt))

    mean_edge_length = edge_length_derivatives / edge_counts

    return mean_edge_length * frame.global_scale


def vertex_rotations_derivative(
    dv_dt: torch.Tensor, proxy: mesh.TriangleMesh
) -> torch.Tensor:
    """Evaluate the rate of change of vertex rotation matrices over time.

    This corresponds to the derivative of the
    `BlendedPolynomialSurface.vertex_rotations` property.

    :param dv_dt: rate of change of proxy vertex positions over time at time t.
    :param proxy: proxy at time t.
    """
    dnormals_dt = _derivative_of_vertex_normals(dv_dt, proxy)
    dneighbours_dt, neighbours = _derivative_of_neighbour_directions(
        dv_dt, dnormals_dt, proxy
    )

    drotations_dt = torch.zeros(proxy.num_vertices, 3, 3).double()

    drotations_dt[..., 0] = dneighbours_dt
    drotations_dt[..., 1] = torch.linalg.cross(
        proxy.vertex_normals, dneighbours_dt
    ) + torch.linalg.cross(dnormals_dt, neighbours)
    drotations_dt[..., 2] = dnormals_dt

    return drotations_dt


def _derivative_of_vertex_normals(
    dv_dt: torch.Tensor, proxy: mesh.TriangleMesh
) -> torch.Tensor:
    """Evaluate the rate of change of vertex normals over time.

    The implementation is based on the related function
    `BlendedPolynomialSurface.vertex_rotations`.

    :param dv_dt: rate of change of proxy vertex positions over time at time t.
    :param proxy: proxy at time t.
    """
    triangle_vertices = proxy.vertices[proxy.triangles].double()
    x = triangle_vertices[:, 0]
    y = triangle_vertices[:, 1]
    z = triangle_vertices[:, 2]

    triangle_v_primes = dv_dt[proxy.triangles].double()
    x_prime = triangle_v_primes[:, 0]
    y_prime = triangle_v_primes[:, 1]
    z_prime = triangle_v_primes[:, 2]

    face_normal_derivatives = torch.linalg.cross(
        y_prime - x_prime, z - x
    ) + torch.linalg.cross(y - x, z_prime - x_prime)

    vertex_normal_derivatives = torch.zeros_like(proxy.vertex_normals)
    vertex_face_counts = torch.zeros(proxy.num_vertices, 1)

    # In the following, we assume that there are no boundary vertices.
    # For each vertex of a face...
    for i in range(3):
        # ...add the normal derivative for that face.
        vi = proxy.triangles[:, i]

        # Update the accumulators.
        vertex_normal_derivatives = torch.index_add(
            vertex_normal_derivatives, 0, vi, face_normal_derivatives
        )
        vertex_face_counts = torch.index_add(
            vertex_face_counts, 0, vi, torch.ones((proxy.num_triangles, 1))
        )

    vertex_normal_derivatives /= vertex_face_counts
    return _derivative_of_unit(proxy.vertex_normals, vertex_normal_derivatives)


def _derivative_of_neighbour_directions(
    dv_dt: torch.Tensor, dnormals_dt: torch.Tensor, proxy: mesh.TriangleMesh
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the rate of change of 'canonical' neighbour directions over time.

    Also returns the neighbour directions themselves.

    The implementation is based on the related function
    `BlendedPolynomialSurface.vertex_rotations`.

    :param dv_dt: rate of change of proxy vertex positions over time at time t.
    :param dnormals_dt: rate of change of vertex normals over time at time t.
    :param proxy: proxy at time t.
    """
    neighbour_directions_derivative = torch.zeros_like(proxy.vertices)
    neighbour_directions = torch.zeros_like(proxy.vertices)

    for vertex_id in range(proxy.num_vertices):
        vertex = proxy.vertices[vertex_id].double()
        normal = proxy.vertex_normals[vertex_id]

        # Select the neighbour with the lowest id.
        neighbour_id = min(proxy.adjacency_list[vertex_id])
        neighbour = proxy.vertices[neighbour_id]

        # Project the edge onto the tangent plane.
        diff = neighbour - vertex
        diff_projected = diff - normal * torch.dot(normal, diff)
        neighbour_directions[vertex_id] = diff_projected / torch.linalg.norm(
            diff_projected, dim=-1
        )

        diff_prime = dv_dt[neighbour_id] - dv_dt[vertex_id]
        normal_prime = dnormals_dt[vertex_id]

        diff_projected_prime = (
            diff_prime
            - torch.linalg.vecdot(normal_prime, diff)[..., None] * normal
            - torch.linalg.vecdot(normal, diff_prime)[..., None] * normal
            - torch.linalg.vecdot(normal, diff)[..., None] * normal_prime
        )

        neighbour_directions_derivative[vertex_id] = _derivative_of_unit(
            diff_projected, diff_projected_prime
        )

    return neighbour_directions_derivative, neighbour_directions


def _derivative_of_unit(v: torch.Tensor, v_prime: torch.Tensor) -> torch.Tensor:
    """Calculate the derivative of v/|v|."""
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v_prime / norm_v - v * torch.linalg.vecdot(v, v_prime)[..., None] / norm_v**3


def _derivative_of_norm(v: torch.Tensor, v_prime: torch.Tensor) -> torch.Tensor:
    """Calculate the derivative of |v|."""
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    return torch.linalg.vecdot(v, v_prime)[..., None] / norm_v


class Polyline:
    """A piece-wise linear deformation between keyframes."""

    def __init__(self, *frames: BlendedPolynomialSurface) -> None:
        """Initialize a polyline deformation from a set of keyframes."""
        self.frames = frames
        self.num_segments = len(frames) - 1

    def get_frame(self, t: float) -> BlendedPolynomialSurface:
        """Get the frame at time t.

        t is clamped to [0, 1].
        """
        if t >= 1:
            return self.frames[-1]

        # Clamp t from below and scale to the number of segments.
        t = max(0, t) * self.num_segments

        # Find the relevant linear segment.
        segment_start = math.floor(t)
        segment_progress = t % 1
        return make_frame(
            self.frames[segment_start], self.frames[segment_start + 1], segment_progress
        )


def split_segment_arap(
    start: BlendedPolynomialSurface, finish: BlendedPolynomialSurface
) -> Polyline:
    """Find a two-segment polyline to connect two BPSs using the ARAP metric."""
    num_vert_coeffs = start.proxy.vertices.numel()

    def make_bps(x: torch.Tensor) -> BlendedPolynomialSurface:
        verts = x[:num_vert_coeffs].reshape(start.proxy.vertices.shape)
        coeffs = x[num_vert_coeffs:].reshape(start.coefficients.shape)
        proxy = mesh.TriangleMesh(verts, start.proxy.triangles)
        return BlendedPolynomialSurface(
            proxy, start.degree, start.global_scale, coeffs, start.beta
        )

    def calc_energy(x: np.typing.ArrayLike) -> torch.Tensor:
        x = torch.as_tensor(x)
        bps = make_bps(x)
        return energy(start, bps, 0, 4) + energy(bps, finish, 0, 4)

    midpoint = make_frame(start, finish, 0.5)

    x0 = torch.cat(
        [midpoint.proxy.vertices.reshape(-1), midpoint.coefficients.reshape(-1)]
    )
    result = scipy.optimize.minimize(calc_energy, x0.numpy())
    intermediate_frame = make_bps(torch.as_tensor(result.x))

    return Polyline(start, intermediate_frame, finish)
