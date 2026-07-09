# PYTHON_ARGCOMPLETE_OK
"""Command-line interface for BCSI."""

import argparse
import logging
import pathlib

import argcomplete


def get_parser() -> argparse.ArgumentParser:
    """Create the BCSI argument parser."""
    bps_parser = argparse.ArgumentParser(add_help=False)
    bps_parser.add_argument(
        "--degree",
        default=1,
        help="degree of polynomials to use to represent the surface",
    )
    bps_parser.add_argument(
        "--scale",
        default=0.5,
        help="global scale to use for blended chart surfaces (default: 0.5)",
    )
    bps_parser.add_argument(
        "--beta",
        default=0.73,
        help="beta ('blending overlap') to use for blended chart surfaces "
        "(default: 0.73)",
    )
    bps_parser.add_argument(
        "--resolution",
        default=3,
        help="resolution with which to render blended chart surfaces (default: 3)",
    )
    bps_parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether or not to show the rendered mesh (default: false)",
    )
    bps_parser.add_argument(
        "--output-name",
        default="result",
        help="name of directory in ./output/ in which to place output files "
        "(default: 'result')",
    )

    diff_parser = argparse.ArgumentParser(add_help=False)
    diff_parser.add_argument(
        "--diff-metric",
        choices=[None, "vertex-to-vertex", "vertex-to-mesh"],
        default=None,
        help="metric to use to compare rendered surfaces to the target surfaces",
    )

    parser = argparse.ArgumentParser(
        description="Blended chart surface interpolation utility",
    )
    subparsers = parser.add_subparsers()

    create_bps = subparsers.add_parser("create-bps", parents=[bps_parser])
    create_bps.add_argument(
        "mesh_path", help="path to proxy mesh file", type=pathlib.Path
    )
    create_bps.set_defaults(func=_initialize_bps)

    submesh_bps = subparsers.add_parser(
        "submesh-bps", parents=[bps_parser, diff_parser]
    )
    submesh_bps.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    submesh_bps.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    submesh_bps.set_defaults(func=_submesh_bps)

    create_submesh_frames = subparsers.add_parser(
        "create-submesh-frames", parents=[bps_parser, diff_parser]
    )
    create_submesh_frames.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    create_submesh_frames.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    create_submesh_frames.add_argument(
        "frame_paths",
        help="path(s) to new poses of the parent mesh to create frames from",
        nargs="+",
        type=pathlib.Path,
    )
    create_submesh_frames.add_argument(
        "--method",
        help="method of coefficient transfer between frames (default: individual)",
        choices=["individual", "use-reference"],
        default="individual",
    )
    create_submesh_frames.set_defaults(func=_create_submesh_frames)

    return parser


def get_output_dir(name: str) -> pathlib.Path:
    """Get an output directory with the given `name`, creating it if necessary."""
    output_dir = pathlib.Path(__file__).parent.parent.parent / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """Run the parser."""
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    logging.basicConfig(
        filename="logs/log",
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    args.func(args)


def _initialize_bps(args: argparse.Namespace) -> None:
    # Defer heavy imports.

    from bcsi import bps, cache, mesh, render

    m = mesh.read_from_file(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(m, args.degree, args.scale, beta=args.beta)
    cache.bps_onerings(args.mesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    if args.visualize:
        mesh.show(surface_rendered)
    mesh.write_to_file(get_output_dir(args.output_name) / "bps.ply", surface_rendered)


def _submesh_bps(args: argparse.Namespace) -> None:
    # Defer heavy imports.
    import torch

    from bcsi import cache, diff, mesh, render, submesh

    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    surface = submesh.create_bps_degree_one(pair, args.degree, args.scale, args.beta)
    cache.bps_onerings(args.submesh_path.name, surface)
    surface_rendered = render.blended_polynomial_surface(surface, args.resolution)
    surface_rendered.open3d.compute_vertex_normals()

    if args.diff_metric:
        metric_func = {
            "vertex-to-vertex": diff.vertex_to_vertex,
            "vertex-to-mesh": diff.vertex_to_mesh,
        }[args.diff_metric]
        print(f"Diff ({args.diff_metric}) between result BPS and input parent mesh:")
        diff_ = metric_func(surface_rendered, parent)
        diff.print(diff_)
        colors = torch.ones_like(surface_rendered.vertices)
        colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
        surface_rendered.set_vertex_colors(colors)

    if args.visualize:
        mesh.show(surface_rendered)

    mesh.write_to_file(
        get_output_dir(args.output_name) / "bps-submesh.ply",
        surface_rendered,
        write_vertex_colors=True,
    )


def _create_submesh_frames(args: argparse.Namespace) -> None:
    # Defer heavy imports.
    import torch

    from bcsi import bps, cache, diff, mesh, render, submesh

    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    reference_bps = submesh.create_bps_degree_one(
        pair, args.degree, args.scale, args.beta
    )
    cache.bps_onerings(args.submesh_path.name, reference_bps)

    diffs = []
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
        mesh.write_to_file(
            get_output_dir(args.output_name) / f"frame-proxy-{i}.ply", frame_pair.child
        )

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
            diffs.append(diff_)
            colors = torch.ones_like(frame_bps_rendered.vertices)
            colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
            frame_bps_rendered.set_vertex_colors(colors)

        mesh.write_to_file(
            get_output_dir(args.output_name) / f"frame-bps-{i}.ply", frame_bps_rendered
        )

    # Display diffs.
    if metric_func:
        print(
            f"Diffs ({args.diff_metric}) between BPS results and input parent meshes:"
        )
        print("Reference:")
        diff.print(
            metric_func(
                render.blended_polynomial_surface(reference_bps, args.resolution),
                parent,
            )
        )
        print()

        for path, diff_ in zip(args.frame_paths, diffs, strict=True):
            print(str(path))
            diff.print(diff_)
            print()

    if args.visualize:
        for m in rendered_meshes:
            mesh.show(m)


if __name__ == "__main__":
    main()
