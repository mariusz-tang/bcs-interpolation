"""Commands available via the CLI.

The commands are separated from the CLI itself because they require heavy
imports, which we want to defer until after argument parsing.
"""

import argparse
import json
import pathlib

import torch

from bcsi import bps, cache, diff, mesh, render, submesh


def get_output_dir(name: str) -> pathlib.Path:
    """Get an output directory with the given `name`, creating it if necessary."""
    output_dir = pathlib.Path(__file__).parent.parent.parent / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def initialize_bps(args: argparse.Namespace) -> None:
    """Create a BPS from a proxy mesh."""
    m = mesh.read_from_file(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(m, args.degree, args.scale, beta=args.beta)
    cache.bps_onerings(args.mesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    if args.visualize:
        mesh.show(surface_rendered)
    mesh.write_to_file(get_output_dir(args.output_name) / "bps.ply", surface_rendered)


def submesh_bps(args: argparse.Namespace) -> None:
    """Create a BPS from a fine mesh and a coarse submesh."""
    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    surface = submesh.create_bps_degree_one(pair, args.degree, args.scale, args.beta)
    cache.bps_onerings(args.submesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    outdir = get_output_dir(args.output_name)

    if args.diff_metric:
        metric_func = {
            "vertex-to-vertex": diff.vertex_to_vertex,
            "vertex-to-mesh": diff.vertex_to_mesh,
        }[args.diff_metric]
        print(f"Diff ({args.diff_metric}) between result BPS and input parent mesh:")
        diff_ = metric_func(surface_rendered, parent)
        _write_data(outdir / "diff.json", {"submesh": diff.summary(diff_)})
        colors = torch.ones_like(surface_rendered.vertices)
        colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
        surface_rendered.set_vertex_colors(colors)

    if args.visualize:
        mesh.show(surface_rendered)

    mesh.write_to_file(
        outdir / "bps-submesh.ply",
        surface_rendered,
        write_vertex_colors=True,
    )


def _write_data(path: pathlib.Path, data: dict) -> None:
    print(f"Writing data to {path}")
    with path.open("w") as f:
        json.dump(data, f)


def create_submesh_frames(args: argparse.Namespace) -> None:
    """Create poses from fine meshes and a coarse proxy mesh."""
    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    reference_bps = submesh.create_bps_degree_one(
        pair, args.degree, args.scale, args.beta
    )
    cache.bps_onerings(args.submesh_path.name, reference_bps)

    outdir = get_output_dir(args.output_name)

    diffs = {}
    rendered_meshes = []

    metric_func = {
        "vertex-to-vertex": diff.vertex_to_vertex,
        "vertex-to-mesh": diff.vertex_to_mesh,
        None: None,
    }[args.diff_metric]

    for i, frame_path in enumerate(args.frame_paths):
        # Make new submesh pair.
        frame_parent = mesh.read_from_file(frame_path)
        frame_pair = submesh.new_frame(pair, frame_parent)

        # Save the new proxy.
        mesh.write_to_file(outdir / f"frame-proxy-{i}.ply", frame_pair.child)

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
        if metric_func:
            diff_ = metric_func(frame_bps_rendered, frame_parent)
            diffs[f"frame-{i}"] = diff.summary(diff_)
            colors = torch.ones_like(frame_bps_rendered.vertices)
            colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
            frame_bps_rendered.set_vertex_colors(colors)

        mesh.write_to_file(outdir / f"frame-bps-{i}.ply", frame_bps_rendered)

    # Save diffs.
    if metric_func:
        diffs["reference"] = diff.summary(
            metric_func(
                render.blended_polynomial_surface(reference_bps, args.resolution),
                parent,
            )
        )
        _write_data(outdir / "frame-diff.json", diffs)

    if args.visualize:
        for m in rendered_meshes:
            mesh.show(m)
