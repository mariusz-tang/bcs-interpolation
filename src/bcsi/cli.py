# PYTHON_ARGCOMPLETE_OK
"""Command-line interface for BCSI."""

import argparse
import pathlib

import argcomplete


def get_parser() -> argparse.ArgumentParser:
    """Create the BCSI argument parser."""
    parser = argparse.ArgumentParser(
        description="Blended chart surface interpolation utility",
    )
    parser.add_argument("mesh_path", help="path to proxy mesh file", type=pathlib.Path)
    parser.add_argument(
        "--degree",
        default=2,
        help="degree of polynomials to use to represent the surface",
    )
    parser.add_argument(
        "--scale",
        default=0.5,
        help="global scale to use for blended chart surfaces (default: 0.5)",
    )
    parser.add_argument(
        "--beta",
        default=0.73,
        help="beta ('blending overlap') to use for blended chart surfaces "
        "(default: 0.73)",
    )
    parser.add_argument(
        "--resolution",
        default=3,
        help="resolution with which to render blended chart surfaces (default: 3)",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, color each mapped 'face' differently (default: false)",
    )
    parser.add_argument(
        "--output-name",
        default="result",
        help="name to give the output mesh, which will be saved as a .obj file "
        "in ./output (default: 'result')",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether or not to show the rendered mesh (default: true)",
    )
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


if __name__ == "__main__":
    main()
