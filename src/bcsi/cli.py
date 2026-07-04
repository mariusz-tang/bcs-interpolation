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

    parser = argparse.ArgumentParser(
        description="Blended chart surface interpolation utility",
    )
    subparsers = parser.add_subparsers()

    create_bps = subparsers.add_parser("create-bps", parents=[bps_parser])
    create_bps.add_argument(
        "mesh_path", help="path to proxy mesh file", type=pathlib.Path
    )
    create_bps.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, color each mapped 'face' differently (default: false)",
    )
    create_bps.add_argument(
        "--output-name",
        default="result",
        help="name to give the output mesh, which will be saved as a .obj file "
        "in ./output (default: 'result')",
    )
    create_bps.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether or not to show the rendered mesh (default: true)",
    )
    create_bps.set_defaults(func=_initialize_bps)

    submesh_bps = subparsers.add_parser("submesh-bps", parents=[bps_parser])
    submesh_bps.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    submesh_bps.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    submesh_bps.set_defaults(func=_submesh_bps)

    create_submesh_frames = subparsers.add_parser(
        "create-submesh-frames", parents=[bps_parser]
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
    create_submesh_frames.set_defaults(func=_create_submesh_frames)

    return parser


def get_output_dir() -> pathlib.Path:
    """Get the output directory, creating it if necessary."""
    output_dir = pathlib.Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
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
    import open3d as o3d

    from bcsi import bps, mesh, render

    m = mesh.read_from_file(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(m, args.degree, args.scale, beta=args.beta)
    surface_rendered = render.blended_polynomial_surface(
        surface, args.resolution, color_patches=args.color
    )
    surface_rendered.open3d.compute_vertex_normals()

    if args.visualize:
        o3d.visualization.draw_geometries([surface_rendered.open3d])
    mesh.write_to_file(get_output_dir() / f"{args.output_name}.obj", surface_rendered)


def _submesh_bps(args: argparse.Namespace) -> None:
    # Defer heavy imports.
    from bcsi import mesh, render, submesh

    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    surface = submesh.create_bps_degree_one(pair, args.degree, args.scale, args.beta)
    surface_rendered = render.blended_polynomial_surface(
        surface, args.resolution, color_patches=False
    )
    surface_rendered.open3d.compute_vertex_normals()

    mesh.write_to_file(get_output_dir() / "result-submesh.obj", surface_rendered)


def _create_submesh_frames(args: argparse.Namespace) -> None:
    # Defer heavy imports.
    from bcsi import mesh, render, submesh

    child = mesh.read_from_file(args.submesh_path)
    parent = mesh.read_from_file(args.parent_mesh_path)
    pair = submesh.Pair(child, parent)

    for i, frame_path in enumerate(args.frame_paths):
        frame_parent = mesh.read_from_file(frame_path)
        frame_pair = submesh.new_frame(pair, frame_parent)

        mesh.write_to_file(get_output_dir() / f"result-frame-{i}.obj", frame_pair.child)

        frame_bps = submesh.create_bps_degree_one(
            frame_pair, args.degree, args.scale, args.beta
        )
        frame_bps_rendered = render.blended_polynomial_surface(
            frame_bps, resolution=args.resolution
        )
        frame_bps_rendered.open3d.compute_vertex_normals()
        mesh.write_to_file(
            get_output_dir() / f"result-frame-bps-{i}.obj", frame_bps_rendered
        )


if __name__ == "__main__":
    main()
