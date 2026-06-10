import math
import pathlib

import open3d as o3d
import pytest
import torch

from bcsi import bps

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
CUBE_PATH = TEST_DATA_DIR / "cube.obj"


@pytest.fixture
def cube_mesh() -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(CUBE_PATH)


def test_constructor_proxy_mesh(cube_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, 0)
    assert surface.proxy_mesh == cube_mesh


def test_constructor_num_vertices(cube_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, 0)
    assert surface.num_vertices == 8


def test_constructor_num_triangles(cube_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, 0)
    assert surface.num_triangles == 12


@pytest.mark.parametrize("degree", range(5))
def test_constructor_degree(cube_mesh: o3d.geometry.TriangleMesh, degree: int) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, degree)
    assert surface.degree == degree


def test_constructor_degree_negative_raises(
    cube_mesh: o3d.geometry.TriangleMesh,
) -> None:
    with pytest.raises(ValueError, match="degree cannot be negative"):
        bps.BlendedPolynomialSurface(cube_mesh, -1)


@pytest.mark.parametrize(
    ("degree", "expected_coefficients"),
    [
        (0, torch.tensor([[[0], [0], [0]]] * 8)),
        (1, torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]] * 8)),
        (
            2,
            torch.tensor(
                [[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]] * 8
            ),
        ),
    ],
)
def test_constructor_coefficients_default(
    cube_mesh: o3d.geometry.TriangleMesh,
    degree: int,
    expected_coefficients: torch.Tensor,
) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, degree)
    assert torch.equal(surface.coefficients, expected_coefficients)


def test_constructor_coefficients_single_vertex(
    cube_mesh: o3d.geometry.TriangleMesh,
) -> None:
    coeffs_single = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 1, 1]])
    coeffs_all = torch.tensor([[[1, 1, 0], [0, 1, 1], [0, 1, 1]]] * 8)
    surface = bps.BlendedPolynomialSurface(cube_mesh, 1, coefficients=coeffs_single)
    assert torch.equal(surface.coefficients, coeffs_all)


def test_constructor_coefficients_all(
    cube_mesh: o3d.geometry.TriangleMesh,
) -> None:
    coefficients = torch.tensor([[[1, 1, 0], [0, 1, 1], [0, 1, 1]]] * 8)
    surface = bps.BlendedPolynomialSurface(cube_mesh, 1, coefficients=coefficients)
    assert torch.equal(surface.coefficients, coefficients)


def test_constructor_coefficients_invalid_shape_raises(
    cube_mesh: o3d.geometry.TriangleMesh,
) -> None:
    coefficients = torch.tensor([[[1, 1, 0], [0, 1, 1], [0, 1, 1]]] * 7)
    with pytest.raises(ValueError, match="invalid shape for coefficients"):
        bps.BlendedPolynomialSurface(cube_mesh, 1, coefficients=coefficients)


@pytest.mark.parametrize("scale", torch.linspace(0, 1, 10))
def test_constructor_global_scale(
    cube_mesh: o3d.geometry.TriangleMesh, scale: float
) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, 2, scale=scale)
    assert surface.global_scale == scale


@pytest.mark.parametrize("beta", torch.linspace(0, 1, 10))
def test_constructor_beta(cube_mesh: o3d.geometry.TriangleMesh, beta: float) -> None:
    surface = bps.BlendedPolynomialSurface(cube_mesh, 2, beta=beta)
    assert surface.beta == beta


@pytest.fixture
def pyramid_mesh() -> o3d.geometry.TriangleMesh:
    # We use this fixture to test the model against a small set of manually
    # calculated values. This approach has already caught at least one bug :)

    # We define this inline because we need control of the vertex and face
    # ordering, and open3d does not preserve these when reading from a file.
    # For convenience, this mesh is also available at tests/data/pyramid.obj.
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


@pytest.mark.parametrize("scale_global", torch.linspace(0, 1, 10))
def test_vertex_scales(
    pyramid_mesh: o3d.geometry.TriangleMesh, scale_global: float
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 2, scale=scale_global)
    expected_scales = (
        torch.tensor(
            [
                (3 + math.sqrt(2)) / 4,
                (2 + math.sqrt(2)) / 3,
                (2 + math.sqrt(2) + math.sqrt(3)) / 4,
                (2 + math.sqrt(2)) / 3,
                (1 + 2 * math.sqrt(2) + math.sqrt(3)) / 4,
            ]
        )
        * scale_global
    )

    assert surface.vertex_scales.shape == expected_scales.shape
    assert torch.allclose(surface.vertex_scales, expected_scales)


def test_vertex_rotations(pyramid_mesh: o3d.geometry.TriangleMesh) -> None:
    # This test assumes that vertex normals are calculated by taking an
    # unweighted average of the adjacent face normals.
    # The selected neighbour vertex is vertex 1 (0-indexed) at (1,0,0).
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 2)
    normal = torch.tensor([-1 / math.sqrt(6), -1 / math.sqrt(6), -2 / math.sqrt(6)])
    neighbour_direction = torch.tensor([5 / 6, -1 / 6, -1 / 3])
    neighbour_direction /= torch.linalg.norm(neighbour_direction)
    expected_rotation_matrix = torch.stack(
        [
            neighbour_direction,
            torch.linalg.cross(normal, neighbour_direction),
            normal,
        ]
    ).t()
    assert surface.vertex_rotations[0].shape == expected_rotation_matrix.shape
    assert torch.allclose(surface.vertex_rotations[0], expected_rotation_matrix)


@pytest.mark.parametrize(
    ("degree", "coefficients", "vertex_id", "coordinates", "expected_output"),
    [
        (1, None, 0, torch.tensor([[0, 0]]), torch.tensor([[0, 0, 0]])),
        (
            1,
            None,
            0,
            torch.tensor([[0, 0], [0, 0]]),
            torch.tensor([[0, 0, 0], [0, 0, 0]]),
        ),
        (
            1,
            torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 1]]),
            0,
            torch.tensor([[1, 0], [0, 2], [1, 2]]),
            torch.tensor(
                [
                    [
                        math.sqrt(5 / 6),
                        -1 / math.sqrt(30),
                        -2 / math.sqrt(30),
                    ],
                    [
                        -2 / math.sqrt(6),
                        -2 / math.sqrt(6),
                        -4 / math.sqrt(6),
                    ],
                    [
                        math.sqrt(5 / 6) - 2 / math.sqrt(6),
                        -1 / math.sqrt(30) - 2 / math.sqrt(6),
                        -2 / math.sqrt(30) - 4 / math.sqrt(6),
                    ],
                ]
            )
            # Apply local scale.
            * (3 + math.sqrt(2))
            / 4
            * 0.5,
        ),
    ],
)
def test_evaluate_patch(
    pyramid_mesh: o3d.geometry.TriangleMesh,
    degree: int,
    coefficients: torch.Tensor | None,
    vertex_id: int,
    coordinates: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    surface = bps.BlendedPolynomialSurface(
        pyramid_mesh, degree, coefficients=coefficients
    )
    result = surface.evaluate_patch(vertex_id, coordinates)
    assert result.shape == expected_output.shape
    assert torch.allclose(result, expected_output.float())


@pytest.mark.parametrize("vertex_id", [-6, 5])
def test_evaluate_patch_invalid_vertex_id_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh, vertex_id: int
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match=f"vertex_id {vertex_id} out of range"):
        surface.evaluate_patch(vertex_id, torch.tensor([[0, 0]]))


def test_evaluate_patch_coordinates_wrong_shape_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh,
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match="invalid shape for coordinates"):
        surface.evaluate_patch(0, torch.tensor([[0]]))


def test_triangle_onering_indices(pyramid_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    assert torch.equal(
        surface.triangle_onering_indices,
        torch.tensor(
            [[3, 3, 0], [2, 2, 0], [0, 2, 0], [1, 3, 0], [1, 2, 1], [1, 1, 2]]
        ),
    )


def test_triangle_onering_flips(pyramid_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    assert torch.all(surface.triangle_onering_flips == 1)


@pytest.mark.parametrize(
    ("triangle_id", "vertices", "expected_output"),
    [
        (
            0,
            torch.tensor([[0, 0]]),
            torch.tensor([[[0, 3 * torch.pi / 2], [1, 2 * torch.pi], [1, 0]]]),
        ),
        (
            0,
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
            torch.tensor(
                [
                    [
                        [[0, 3 * torch.pi / 2], [1, 2 * torch.pi], [1, 0]],
                        [[0, 3 * torch.pi / 2], [1, 2 * torch.pi], [1, 0]],
                    ],
                    [
                        [[0, 3 * torch.pi / 2], [1, 2 * torch.pi], [1, 0]],
                        [[0, 3 * torch.pi / 2], [1, 2 * torch.pi], [1, 0]],
                    ],
                ]
            ),
        ),
        (
            1,
            torch.tensor([[0.5, 1 / (math.sqrt(3) * 2)]]),
            torch.tensor(
                [
                    [
                        [1 / math.sqrt(3), 5 * torch.pi / 4],
                        [1 / math.sqrt(3), 5 * torch.pi / 3],
                        [1 / math.sqrt(3), torch.pi / 4],
                    ]
                ]
            ),
        ),
    ],
)
def test_get_onering_coordinates(
    pyramid_mesh: o3d.geometry.TriangleMesh,
    triangle_id: int,
    vertices: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    result = surface.get_onering_coordinates(triangle_id, vertices)
    assert result.shape == expected_output.shape
    assert torch.allclose(result, expected_output)


def test_get_onering_coordinates_wrong_input_shape_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh,
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match="bad shape for vertices"):
        surface.get_onering_coordinates(0, torch.tensor([1, 1, 1]))


@pytest.mark.parametrize("triangle_id", [-7, 6])
def test_get_onering_coordinates_triangle_id_out_of_range_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh, triangle_id: int
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match=f"triangle_id {triangle_id} out of range"):
        surface.get_onering_coordinates(triangle_id, torch.tensor([[1, 1]]))


def test_get_unblended_patch_vertices(pyramid_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    result = surface.get_unblended_patch_vertices(0, torch.tensor([[0, 0], [1, 0]]))
    assert result.shape == (3, 2, 3)
    assert torch.allclose(
        result[0],
        surface.evaluate_patch(0, torch.tensor([[0, 0], [0, -1]])),
    )


def test_get_unblended_patch_vertices_wrong_input_shape_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh,
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match="bad shape for vertices"):
        surface.get_unblended_patch_vertices(0, torch.tensor([1, 1, 1]))


@pytest.mark.parametrize("triangle_id", [-7, 6])
def test_get_unblended_patch_vertices_triangle_id_out_of_range_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh, triangle_id: int
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match=f"triangle_id {triangle_id} out of range"):
        surface.get_unblended_patch_vertices(triangle_id, torch.tensor([[1, 1]]))


def test_get_blended_patch_vertices(pyramid_mesh: o3d.geometry.TriangleMesh) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    result = surface.get_blended_patch_vertices(
        0, torch.tensor([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    )
    assert result.shape == (3, 3)
    assert torch.allclose(
        result, torch.tensor([[0, 0, 0], [1, 1, 0], [1, 0, 0]]).float()
    )


def test_get_blended_patch_vertices_wrong_input_shape_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh,
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match="bad shape for vertices"):
        surface.get_blended_patch_vertices(0, torch.tensor([1, 1, 1]))


@pytest.mark.parametrize("triangle_id", [-7, 6])
def test_get_blended_patch_vertices_triangle_id_out_of_range_raises(
    pyramid_mesh: o3d.geometry.TriangleMesh, triangle_id: int
) -> None:
    surface = bps.BlendedPolynomialSurface(pyramid_mesh, 1)
    with pytest.raises(ValueError, match=f"triangle_id {triangle_id} out of range"):
        surface.get_blended_patch_vertices(triangle_id, torch.tensor([[1, 1]]))
