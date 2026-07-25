"""Screenshot generation."""

import pathlib

import numpy as np
import open3d as o3d
import torch

import bcsi.bps
import bcsi.mesh


def bps(
    bps: bcsi.bps.BlendedPolynomialSurface,
    save_dir: pathlib.Path,
    filename: str,
    resolution: int = 5,
) -> None:
    """Save a linear deformation in BPS space as a series of screenshots."""
    mesh_ = bcsi.bps.render.surface(bps, resolution)
    mesh(mesh_, save_dir, f"{filename}")


def mesh(mesh: bcsi.mesh.TriangleMesh, save_dir: pathlib.Path, filename: str) -> None:
    """Save screenshots of a mesh from various angles."""
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh.open3d)

    mesh.open3d.compute_vertex_normals()
    mesh.vertex_colors = torch.ones_like(mesh.vertices) * 0.8

    control = visualizer.get_view_control()
    control.set_zoom(0.7)

    camera_views = {
        "x": {"front": [1, 0, 0], "up": [0, 0, 1]},
        "y": {"front": [0, 1, 0], "up": [0, 0, 1]},
        "z": {"front": [0, 0, 1], "up": [0, 1, 0]},
    }
    centre = mesh.vertices.mean(dim=0).numpy()

    for axis, params in camera_views.items():
        control.set_lookat(centre)
        control.set_front(np.array(params["front"]))
        control.set_up(np.array(params["up"]))

        visualizer.poll_events()
        visualizer.update_renderer()

        path = save_dir / f"{axis}-{filename}.png"
        visualizer.capture_screen_image(path, do_render=True)
        print(f"Image saved to {path}.")

    visualizer.destroy_window()
