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

    create_bps = subparsers.add_parser(
        "create-bps",
        parents=[bps_parser],
        help="create a BPS from a coarse proxy mesh",
        description="Create a BPS from a coarse proxy mesh. The coefficients "
        "will be initialized as level unit planes.",
    )
    create_bps.add_argument(
        "mesh_path", help="path to proxy mesh file", type=pathlib.Path
    )
    create_bps.set_defaults(func_name="initialize_bps")

    submesh_bps = subparsers.add_parser(
        "submesh-bps",
        parents=[bps_parser, diff_parser],
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
        parents=[bps_parser, diff_parser],
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
        choices=["individual", "use-reference"],
        default="individual",
    )
    create_submesh_frames.set_defaults(func_name="create_submesh_frames")

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
    from bcsi import commands

    command_func = getattr(commands, args.func_name)
    command_func(args)
