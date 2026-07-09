"""Commands available via the CLI.

The commands are separated from the CLI itself because they require heavy
imports, which we want to defer until after argument parsing.
"""

import argparse
import pathlib

import torch

from bcsi import bps, cache, diff, io, mesh, plot, render, submesh


def initialize_bps(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create a BPS from a proxy mesh."""
    m = io.read_mesh(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(m, args.degree, args.scale, beta=args.beta)
    cache.bps_onerings(args.mesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    if args.visualize:
        mesh.show(surface_rendered)
    io.write_mesh(output_dir / "bps.ply", surface_rendered)


def submesh_bps(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create a BPS from a fine mesh and a coarse submesh."""
    child = io.read_mesh(args.submesh_path)
    parent = io.read_mesh(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    surface = submesh.create_bps_degree_one(pair, args.degree, args.scale, args.beta)
    cache.bps_onerings(args.submesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    diff_func = _diff_functions[args.diff_metric]
    if diff_func:
        print(f"Diff ({args.diff_metric}) between result BPS and input parent mesh:")
        diff_ = diff_func(surface_rendered, parent)
        io.write_json(output_dir / "diff.json", {"submesh": diff.summary(diff_)})
        _add_diff_colors(surface_rendered, diff_)

    if args.visualize:
        mesh.show(surface_rendered)

    io.write_mesh(
        output_dir / "bps-submesh.ply",
        surface_rendered,
        write_vertex_colors=True,
    )


def create_submesh_frames(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create poses from fine meshes and a coarse proxy mesh."""
    child = io.read_mesh(args.submesh_path)
    parent = io.read_mesh(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    reference_bps = submesh.create_bps_degree_one(
        pair, args.degree, args.scale, args.beta
    )
    cache.bps_onerings(args.submesh_path.name, reference_bps)

    diffs = {}
    rendered_meshes = []

    diff_func = _diff_functions[args.diff_metric]

    for i, frame_path in enumerate(args.frame_paths):
        # Make new submesh pair.
        frame_parent = io.read_mesh(frame_path)
        frame_pair = submesh.new_frame(pair, frame_parent)

        # Save the new proxy.
        io.write_mesh(output_dir / f"frame-proxy-{i}.ply", frame_pair.child)

        # Construct BPS according to selected coefficient transfer method.
        if args.method == "individual":
            frame_bps = submesh.create_bps_degree_one(
                frame_pair, args.degree, args.scale, args.beta
            )
        elif args.method == "use-reference":
            frame_bps = bps.BlendedPolynomialSurface(
                frame_pair.child,
                args.degree,
                args.scale,
                reference_bps.coefficients,
                args.beta,
            )

        # Recover onering data from cache.
        cache.bps_onerings(args.submesh_path.name, frame_bps)

        # Render the new BPS and save it.
        frame_bps_rendered = render.blended_polynomial_surface(
            frame_bps, resolution=args.resolution
        )
        frame_bps_rendered.open3d.compute_vertex_normals()
        rendered_meshes.append(frame_bps_rendered)

        # Save the diff for display at the end.
        if diff_func:
            diff_ = diff_func(frame_bps_rendered, frame_parent)
            diffs[f"{i}"] = diff.summary(diff_)
            _add_diff_colors(frame_bps_rendered, diff_)

        io.write_mesh(output_dir / f"frame-bps-{i}.ply", frame_bps_rendered)

    # Save diffs.
    if diff_func:
        diffs["ref"] = diff.summary(
            diff_func(
                render.blended_polynomial_surface(reference_bps, args.resolution),
                parent,
            )
        )
        io.write_json(output_dir / "frame-diff.json", diffs)

    if args.visualize:
        for m in rendered_meshes:
            mesh.show(m)


_diff_functions = {
    "vertex-to-vertex": diff.vertex_to_vertex,
    "vertex-to-mesh": diff.vertex_to_mesh,
    None: None,
}


def _add_diff_colors(mesh_: mesh.TriangleMesh, diff_: torch.Tensor) -> None:
    colors = torch.ones_like(mesh_.vertices)
    colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
    mesh_.set_vertex_colors(colors)


def plot_diffs(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Plot diff data from JSON files."""
    data = io.read_json(args.diff_path)
    fig = plot.diff_comparison(data)
    io.write_figure(output_dir / "diff.svg", fig)
