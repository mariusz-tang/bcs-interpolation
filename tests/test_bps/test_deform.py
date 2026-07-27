import math

import pytest
import torch

from bcsi import bps, mesh


def test_vertex_scales_derivative() -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_finish = 2 * v_start
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 1)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 1)

    dv_dt = finish.proxy.vertices - start.proxy.vertices
    result = bps.deform.vertex_scales_derivative(dv_dt, start)

    assert torch.allclose(
        result,
        torch.tensor(
            [
                1,
                (2 * math.sqrt(2) + 1) / 3,
                (2 * math.sqrt(2) + 1) / 3,
                (2 * math.sqrt(2) + 1) / 3,
            ]
        ).double()
        * start.global_scale,
    )


def test_patch_derivatives_function() -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_finish = 2 * v_start
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 1)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 1)
    finish.coefficients *= 2

    dcoeffs_dt = finish.coefficients - start.coefficients

    result = bps.deform.patch_derivatives_function(
        dcoeffs_dt, start, torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    )

    assert torch.allclose(
        result[0, :, 0],
        torch.tensor([[0, 0, 0], [1, 0, 0], [-0.5, math.sqrt(3) / 2, 0]]).double(),
    )


def test_vertex_rotations_derivative() -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0.1], [0, 1, 0.1], [0, 0, 1]])
    v_finish = 2 * v_start
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 1)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 1)

    dv_dt = finish.proxy.vertices - start.proxy.vertices
    result = bps.deform.vertex_rotations_derivative(dv_dt, start.proxy)

    assert torch.allclose(result, torch.zeros_like(result))


def test_unblended_patch_derivatives() -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0.1], [0, 1, 0.1], [0, 0, 1]])
    v_finish = 2 * v_start
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    dv_dt = (v_finish - v_start).double()

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 2)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 2)

    dcoeffs_dt = finish.coefficients - start.coefficients

    result = bps.deform.unblended_patch_derivatives(
        dv_dt, dcoeffs_dt, start, torch.tensor([[0, 0]])
    )

    assert torch.allclose(result[:, 0, 0], dv_dt[tris[:, 0]])


def test_blended_patch_derivatives() -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0.1], [0, 1, 0.1], [0, 0, 1]])
    v_finish = 2 * v_start
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    dv_dt = (v_finish - v_start).double()

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 2)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 2)

    dcoeffs_dt = finish.coefficients - start.coefficients

    result = bps.deform.blended_patch_derivatives(
        dv_dt,
        dcoeffs_dt,
        start,
        torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]),
    )

    assert torch.allclose(result, dv_dt[tris])


@pytest.mark.parametrize("t", torch.arange(-1, 2, 0.1))
def test_from_bps_translation_only(t: float) -> None:
    v_start = torch.tensor([[0, 0, 0], [1, 0, 0.1], [0, 1, 0.1], [0, 0, 1]])
    v_finish = v_start + 10
    tris = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])

    start = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_start, tris), 2)
    finish = bps.BlendedPolynomialSurface(mesh.TriangleMesh(v_finish, tris), 2)

    m, d = bps.deform.bps_to_shape_space(start, finish, t, 2)
    assert torch.allclose(d, torch.ones_like(m.vertices).flatten() * 10)
