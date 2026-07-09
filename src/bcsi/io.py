"""File system utilities."""

import json
import pathlib

import open3d as o3d

from bcsi import mesh

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent


def output_dir(name: str) -> pathlib.Path:
    """Get an output directory with the given `name`, creating it if necessary."""
    output_dir = ROOT_DIR / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: pathlib.Path, data: dict) -> None:
    """Write `data` to a json file."""
    print(f"Writing data to {path}")
    with path.open("w") as f:
        json.dump(data, f)


def read_mesh(path: pathlib.Path) -> mesh.TriangleMesh:
    """Load a mesh from a file."""
    o3d_mesh = o3d.io.read_triangle_mesh(path)
    return mesh.TriangleMesh(o3d_mesh)


def write_mesh(
    path: pathlib.Path, mesh: mesh.TriangleMesh, write_vertex_colors: bool = False
) -> None:
    """Write a mesh to a file."""
    print(f"Writing mesh to {path}")
    o3d.io.write_triangle_mesh(
        path,
        mesh.open3d,
        write_vertex_normals=False,
        write_vertex_colors=write_vertex_colors,
        write_triangle_uvs=False,
    )
