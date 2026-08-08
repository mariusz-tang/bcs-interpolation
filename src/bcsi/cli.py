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
        help="name to associate with output files (default: 'result')",
    )
    base_parser.set_defaults(outfile=False)

    bps_parser = argparse.ArgumentParser(add_help=False)
    bps_parser.add_argument(
        "--degree",
        default=1,
        type=int,
        help="degree of polynomials to use to represent the surface",
    )
    bps_parser.add_argument(
        "--scale",
        default=0.5,
        type=float,
        help="global scale to use for blended chart surfaces (default: 0.5)",
    )
    bps_parser.add_argument(
        "--beta",
        default=0.73,
        type=float,
        help="beta ('blending overlap') to use for blended chart surfaces "
        "(default: 0.73)",
    )
    bps_parser.add_argument(
        "--resolution",
        default=3,
        type=int,
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
    create_submesh.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether or not to show the submesh (default: false)",
    )
    create_submesh.set_defaults(outfile=True, func_name="create_submesh")

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
        "diff_names",
        nargs="+",
        help="output names used to generate the diffs",
    )
    plot_diff_parser.add_argument(
        "--dataset-names",
        nargs="+",
        help="names to assign to each JSON file (default: integers starting from 0)",
        type=pathlib.Path,
    )
    plot_diff_parser.set_defaults(outfile=True, func_name="plot_diffs")

    screenshot_mesh = subparsers.add_parser(
        "screenshot-mesh",
        parents=[base_parser],
        help="save screenshots of a mesh",
        description="Save screenshots of a mesh from various angles.",
    )
    screenshot_mesh.add_argument(
        "mesh_path", help="path to mesh file", type=pathlib.Path
    )
    screenshot_mesh.set_defaults(outfile=True, func_name="screenshot_mesh")

    deform_bps = subparsers.add_parser(
        "deform-bps",
        help="create a deformation from a BPS sequence",
        description="Create a deformation from a sequence of BPS objects.",
        parents=[base_parser, bps_parser],
    )
    deform_bps.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    deform_bps.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    deform_bps.add_argument(
        "frame_mesh_paths",
        help="paths to parent meshes at each frame",
        nargs="+",
        type=pathlib.Path,
    )
    deform_bps.add_argument(
        "--num-frames",
        default=10,
        type=int,
        help="number of frames to generate (default: 10)",
    )
    deform_bps.add_argument(
        "--method",
        choices=["linear", "arap"],
        default="linear",
        help="interpolation method (default: linear)",
    )
    deform_bps.set_defaults(outfile=True, func_name="deform_bps")

    deformation_energy = subparsers.add_parser(
        "deformation-energy",
        help="calculate the energy from a BPS deformation",
        description="Calculate the ARAP energy from a BPS polyline deformation.",
        parents=[base_parser, bps_parser],
    )
    deformation_energy.add_argument(
        "submesh_path", help="path to coarse proxy mesh file", type=pathlib.Path
    )
    deformation_energy.add_argument(
        "parent_mesh_path", help="path to fine parent mesh file", type=pathlib.Path
    )
    deformation_energy.add_argument(
        "frame_paths",
        help="path(s) to new poses of the parent mesh to create frames from",
        nargs="+",
        type=pathlib.Path,
    )
    deformation_energy.add_argument(
        "--intermediate-proxy-paths",
        help="path(s) to proxy meshes in the middle of each segment",
        nargs="*",
        type=pathlib.Path,
    )
    deformation_energy.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="number of frames at which to evaluate the ARAP metric (default: 10)",
    )
    deformation_energy.add_argument(
        "--save-bps-frames",
        type=int,
        default=0,
        help="if set and non-zero, render and save this number of BPS frames "
        "(default: 0)",
    )
    deformation_energy.set_defaults(outfile=True, func_name="deformation_energy")

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
    output_path = args.output_name if args.outfile else io.output_dir(args.output_name)
    command_func(args, output_path)
