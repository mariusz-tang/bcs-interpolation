# PYTHON_ARGCOMPLETE_OK
"""Command-line interface for BCSI."""

import argparse
import logging
import pathlib

import argcomplete


def get_parser() -> argparse.ArgumentParser:
    """Create the BCSI argument parser."""
    parser = argparse.ArgumentParser(
        description="Blended chart surface interpolation utility",
    )
    subparsers = parser.add_subparsers()

    create_bps = subparsers.add_parser("create-bps")
    create_bps.add_argument(
        "mesh_path", help="path to proxy mesh file", type=pathlib.Path
    )
    create_bps.add_argument(
        "--degree",
        default=2,
        help="degree of polynomials to use to represent the surface",
    )
    create_bps.add_argument(
        "--scale",
        default=0.5,
        help="global scale to use for blended chart surfaces (default: 0.5)",
    )
    create_bps.add_argument(
        "--beta",
        default=0.73,
        help="beta ('blending overlap') to use for blended chart surfaces "
        "(default: 0.73)",
    )
    create_bps.add_argument(
        "--resolution",
        default=3,
        help="resolution with which to render blended chart surfaces (default: 3)",
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

    submesh_bps = subparsers.add_parser("submesh-bps")
    submesh_bps.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    submesh_bps.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    submesh_bps.set_defaults(func=_submesh_bps)

    create_submesh_frames = subparsers.add_parser("create-submesh-frames")
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

    from bcsi import bps, render

    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(
        mesh, args.degree, args.scale, beta=args.beta
    )
    surface_rendered = render.blended_polynomial_surface(
        surface, args.resolution, color_patches=args.color
    )
    surface_rendered.compute_vertex_normals()

    if args.visualize:
        o3d.visualization.draw_geometries([surface_rendered])
    o3d.io.write_triangle_mesh(
        get_output_dir() / f"{args.output_name}.obj",
        surface_rendered,
        write_ascii=True,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=True,
    )


def _submesh_bps(args: argparse.Namespace) -> None:
    import open3d as o3d

    from bcsi import render, submesh

    child = o3d.io.read_triangle_mesh(args.submesh_path)
    parent = o3d.io.read_triangle_mesh(args.parent_mesh_path)

    vertex_indices = submesh.find_vertex_indices(child, parent)
    surface = submesh.create_bps_degree_one(child, parent, vertex_indices)
    surface_rendered = render.blended_polynomial_surface(
        surface, 3, color_patches=False
    )
    surface_rendered.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        get_output_dir() / "result-submesh.obj",
        surface_rendered,
        write_ascii=True,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=True,
    )


def _create_submesh_frames(args: argparse.Namespace) -> None:
    import open3d as o3d

    from bcsi import render, submesh

    child = o3d.io.read_triangle_mesh(args.submesh_path)
    parent = o3d.io.read_triangle_mesh(args.parent_mesh_path)
    vertex_indices = submesh.find_vertex_indices(child, parent)

    for i, frame_path in enumerate(args.frame_paths):
        frame_parent = o3d.io.read_triangle_mesh(frame_path)
        frame_submesh = submesh.new_frame(child, vertex_indices, frame_parent)

        o3d.io.write_triangle_mesh(
            get_output_dir() / f"result-frame-{i}.obj",
            frame_submesh,
            write_ascii=True,
            write_vertex_normals=False,
            write_vertex_colors=False,
            write_triangle_uvs=False,
            print_progress=True,
        )

        frame_bps = submesh.create_bps_degree_one(
            frame_submesh, frame_parent, vertex_indices
        )
        frame_bps_rendered = render.blended_polynomial_surface(frame_bps, resolution=3)
        frame_bps_rendered.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            get_output_dir() / f"result-frame-bps-{i}.obj",
            frame_bps_rendered,
            write_ascii=True,
            write_vertex_normals=False,
            write_vertex_colors=False,
            write_triangle_uvs=False,
            print_progress=True,
        )


if __name__ == "__main__":
    main()
