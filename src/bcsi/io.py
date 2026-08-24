"""File system utilities."""

import json
import pathlib

import matplotlib.pyplot as plt
import open3d as o3d
import torch

from bcsi import ROOT_DIR, bps, deform, mesh


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
    return mesh.TriangleMesh.from_open3d_legacy(o3d_mesh)


def write_mesh(
    mesh: mesh.TriangleMesh, path: pathlib.Path, write_vertex_colors: bool = False
) -> None:
    """Write a mesh to a file."""
    print(f"Writing mesh to {path}")
    o3d.io.write_triangle_mesh(
        path,
        mesh.open3d_legacy(),
        write_vertex_normals=False,
        write_vertex_colors=write_vertex_colors,
        write_triangle_uvs=False,
    )


def write_polyline(
    polyline: deform.polyline.Polyline[bps.BlendedPolynomialSurface], path: pathlib.Path
) -> None:
    """Write a BPS polyline deformation to a file."""
    result = []
    for bps_ in polyline.frames:
        result.append(
            {
                "vertices": bps_.proxy.vertices,
                "triangles": bps_.proxy.triangles,
                "coefficients": bps_.coefficients,
                "degree": bps_.degree,
                "global_scale": bps_.global_scale,
                "beta": bps_.beta,
                "onering_flips": bps_.triangle_onering_flips,
                "onering_indices": bps_.triangle_onering_indices,
            }
        )

    print(f"Writing polyline to {path}")
    torch.save(result, path)


def read_polyline(
    path: pathlib.Path,
) -> deform.polyline.Polyline[bps.BlendedPolynomialSurface]:
    """Read a BPS polyline deformation from a file."""
    data = torch.load(path)
    frames = []
    for bps_data in data:
        bps_ = bps.BlendedPolynomialSurface(
            mesh.TriangleMesh(bps_data["vertices"], bps_data["triangles"]),
            bps_data["degree"],
            bps_data["global_scale"],
            bps_data["coefficients"],
            bps_data["beta"],
        )
        bps_.triangle_onering_flips = bps_data["onering_flips"]
        bps_.triangle_onering_indices = bps_data["onering_indices"]
        frames.append(bps_)
    return deform.bps.Polyline(frames)


def write_figure(figure: plt.Figure, path: pathlib.Path) -> None:
    """Write a matplotlib figure."""
    print(f"Writing figure to {path}")
    figure.savefig(path)
