"""File system utilities."""

import json
import pathlib

import matplotlib.pyplot as plt
import open3d as o3d

from bcsi import ROOT_DIR, mesh


def output_dir(name: str | None = None) -> pathlib.Path:
    """Get an output directory with the given `name`, creating it if necessary.

    If `name` is none, returns the root output directory.
    """
    output_dir = ROOT_DIR / "output" / name if name else ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_json(path: pathlib.Path) -> dict:
    """Read a json file."""
    with path.open("r") as f:
        return json.load(f)


def write_json(data: dict, path: pathlib.Path) -> None:
    """Write `data` to a json file."""
    print(f"Writing data to {path}")
    with path.open("w") as f:
        json.dump(data, f)


def read_mesh(path: pathlib.Path) -> mesh.TriangleMesh:
    """Load a mesh from a file."""
    o3d_mesh = o3d.io.read_triangle_mesh(path)
    return mesh.TriangleMesh(o3d_mesh)


def write_mesh(
    mesh: mesh.TriangleMesh, path: pathlib.Path, write_vertex_colors: bool = False
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


def write_figure(figure: plt.Figure, path: pathlib.Path) -> None:
    """Write a matplotlib figure."""
    print(f"Writing figure to {path}")
    figure.savefig(path)
