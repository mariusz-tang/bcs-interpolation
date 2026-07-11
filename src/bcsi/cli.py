# PYTHON_ARGCOMPLETE_OK
"""Command-line interface for BCSI."""

import argparse
import logging
import pathlib

import argcomplete


def get_parser() -> argparse.ArgumentParser:
    """Create the BCSI argument parser."""
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--output-name",
        default="result",
        help="name of directory in ./output/ in which to place output files "
        "(default: 'result')",
    )

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

    create_bps = subparsers.add_parser(
        "create-bps",
        parents=[base_parser, bps_parser],
        help="create a BPS from a coarse proxy mesh",
        description="Create a BPS from a coarse proxy mesh. The coefficients "
        "will be initialized as level unit planes.",
    )
    create_bps.add_argument(
        "mesh_path", help="path to proxy mesh file", type=pathlib.Path
    )
    create_bps.set_defaults(func_name="initialize_bps")

    create_submesh = subparsers.add_parser(
        "create-submesh",
        parents=[base_parser],
        help="create a suitable submesh from a fine parent mesh",
        description="Create a submesh from a fine parent mesh. The vertex set "
        "of the result will be a subset of the vertex set of the input.",
    )
    create_submesh.add_argument(
        "mesh_path", help="path to input mesh file", type=pathlib.Path
    )
    create_submesh.add_argument(
        "--scale",
        help="fraction of triangles to keep from the input mesh (default: 0.1)",
        type=float,
        default=0.1,
    )
    create_submesh.set_defaults(func_name="create_submesh")

    show_mesh = subparsers.add_parser(
        "show-mesh",
        parents=[base_parser, bps_parser, diff_parser],
        help="open a mesh in an interactive window",
        description="Open a mesh in an interactive window.",
    )
    show_mesh.add_argument("mesh_path", help="path to the mesh file", type=pathlib.Path)
    show_mesh.set_defaults(func_name="show_mesh")

    submesh_bps = subparsers.add_parser(
        "submesh-bps",
        parents=[base_parser, bps_parser, diff_parser],
        help="create a BPS from a fine mesh and coarse submesh",
        description="Create a BPS from a fine mesh and coarse submesh. The "
        "coarse mesh will be used as the proxy, while the coefficients will be "
        "derived from properties on the fine mesh.",
    )
    submesh_bps.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    submesh_bps.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    submesh_bps.set_defaults(func_name="submesh_bps")

    create_submesh_frames = subparsers.add_parser(
        "create-submesh-frames",
        parents=[base_parser, bps_parser, diff_parser],
        help="create a series of BPS objects from a set of corresponding fine "
        "meshes and a single coarse submesh",
        description="Create a series of BPS objects from a set of corresponding "
        "fine meshes and a coarse submesh corresponding to one of the fine meshes.",
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
        choices=["individual", "use-reference", "mean-simple", "mean-weighted"],
        default="individual",
    )
    create_submesh_frames.set_defaults(func_name="create_submesh_frames")

    plot_diff_parser = subparsers.add_parser(
        "plot-diffs",
        parents=[base_parser],
        help="plot diff data",
        description="Plot diff data produced by other commands.",
    )
    plot_diff_parser.add_argument(
        "diff_paths", nargs="+", help="paths to diff JSON files", type=pathlib.Path
    )
    plot_diff_parser.add_argument(
        "--dataset-names",
        nargs="+",
        help="names to assign to each JSON file (default: integers starting from 0)",
        type=pathlib.Path,
    )
    plot_diff_parser.set_defaults(func_name="plot_diffs")

    return parser


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

    # Defer heavy imports until after parsing.
    from bcsi import commands, io

    command_func = getattr(commands, args.func_name)
    command_func(args, io.output_dir(args.output_name))
