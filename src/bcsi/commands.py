"""Commands available via the CLI.

The commands are separated from the CLI itself because they require heavy
imports, which we want to defer until after argument parsing.
"""

import argparse
import pathlib

import torch

from bcsi import bps, deform, io, mesh, metrics, plot, screenshot


def initialize_bps(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create a BPS from a proxy mesh."""
    m = io.read_mesh(args.mesh_path)
    surface = bps.BlendedPolynomialSurface(m, args.degree, args.scale, beta=args.beta)
    bps.cache.onerings(args.mesh_path.name, surface)
    surface_rendered = bps.render.surface(surface, args.resolution)

    if args.visualize:
        mesh.show(surface_rendered, show_colors=False)
    io.write_mesh(surface_rendered, output_dir / "bps.ply")


def show_mesh(args: argparse.Namespace, _: pathlib.Path) -> None:
    """Open a mesh in an interactive visualizer window."""
    mesh_ = io.read_mesh(args.mesh_path)
    mesh.show(mesh_)


def create_submesh(args: argparse.Namespace, output_name: str) -> None:
    """Create a suitable submesh from a parent mesh."""
    mesh_ = io.read_mesh(args.mesh_path)
    child = mesh.submesh.create(mesh_, args.scale)
    io.write_mesh(child, io.output_dir() / f"{output_name}-submesh.ply")
    if args.visualize:
        mesh.show(child, show_colors=False)


def submesh_bps(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create a BPS from a fine mesh and a coarse submesh."""
    child = io.read_mesh(args.submesh_path)
    parent = io.read_mesh(args.parent_mesh_path)
    pair = mesh.submesh.Pair(child, parent)

    surface = mesh.submesh.create_bps_degree_one(
        pair, args.degree, args.scale, args.beta
    )
    bps.cache.onerings(args.submesh_path.name, surface)
    surface_rendered = bps.render.surface(surface, args.resolution)

    diff_func = _diff_functions[args.diff_metric]
    if diff_func:
        print(f"Diff ({args.diff_metric}) between result BPS and input parent mesh:")
        diff_ = diff_func(surface_rendered, parent)
        io.write_json({"submesh": mesh.diff.summary(diff_)}, output_dir / "diff.json")
        _add_diff_colors(surface_rendered, diff_)

    if args.visualize:
        mesh.show(surface_rendered)

    io.write_mesh(
        surface_rendered,
        output_dir / "bps-submesh.ply",
        write_vertex_colors=True,
    )


def create_submesh_frames(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Create poses from fine meshes and a coarse proxy mesh."""
    child = io.read_mesh(args.submesh_path)
    parent = io.read_mesh(args.parent_mesh_path)
    reference_pair = mesh.submesh.Pair(child, parent)

    reference_bps = mesh.submesh.create_bps_degree_one(
        reference_pair, args.degree, args.scale, args.beta
    )
    bps.cache.onerings(args.submesh_path.name, reference_bps)

    # Construct pairs for each frame.
    frame_pairs = []
    for i, frame_path in enumerate(args.frame_paths):
        # Make new submesh pair.
        frame_parent = io.read_mesh(frame_path)
        frame_pair = mesh.submesh.new_frame(reference_pair, frame_parent)
        frame_pairs.append(frame_pair)

        # Save the new proxy.
        io.write_mesh(
            frame_pair.child,
            output_dir / f"frame-proxy-{i}.ply",
            write_vertex_colors=True,
        )

    # Construct BPSs for each frame.
    if args.method == "use-reference":
        frame_bpss = _construct_bps_list_from_reference(
            reference_pair, frame_pairs, args
        )
    elif args.method == "mean-simple":
        frame_bpss = _construct_bps_list_from_mean(reference_pair, frame_pairs, args)
    elif args.method == "mean-weighted":
        frame_bpss = _construct_bps_list_from_weighted_mean(
            reference_pair, frame_pairs, args
        )
    else:
        frame_bpss = _construct_bps_list_individual(reference_pair, frame_pairs, args)

    diff_func = _diff_functions[args.diff_metric]
    diffs = {}
    rendered_meshes = []

    for i, (bps_, pair) in enumerate(zip(frame_bpss, frame_pairs, strict=True)):
        # Render the new BPS and save it.
        bps_rendered = bps.render.surface(bps_, resolution=args.resolution)
        rendered_meshes.append(bps_rendered)

        # Save the diff for display at the end.
        if diff_func:
            diff_ = diff_func(bps_rendered, pair.parent)
            diffs[f"{i}"] = mesh.diff.summary(diff_)
            _add_diff_colors(bps_rendered, diff_)

        io.write_mesh(
            bps_rendered, output_dir / f"frame-bps-{i}.ply", write_vertex_colors=True
        )

    # Save diffs.
    if diff_func:
        diffs["ref"] = mesh.diff.summary(
            diff_func(
                bps.render.surface(reference_bps, args.resolution),
                parent,
            )
        )
        io.write_json(diffs, output_dir / "frame-diff.json")

    if args.visualize:
        for m in rendered_meshes:
            mesh.show(m)


def _construct_bps_list_individual(
    _: mesh.submesh.Pair,
    frame_pairs: list[mesh.submesh.Pair],
    args: argparse.Namespace,
) -> list[bps.BlendedPolynomialSurface]:
    bps_list = []
    for pair in frame_pairs:
        frame_bps = mesh.submesh.create_bps_degree_one(
            pair, args.degree, args.scale, args.beta
        )
        bps.cache.onerings(args.submesh_path.name, frame_bps)
        bps_list.append(frame_bps)
    return bps_list


def _construct_bps_list_from_reference(
    reference_pair: mesh.submesh.Pair,
    frame_pairs: list[mesh.submesh.Pair],
    args: argparse.Namespace,
) -> list[bps.BlendedPolynomialSurface]:
    bps_list = []
    reference_bps = mesh.submesh.create_bps_degree_one(
        reference_pair, args.degree, args.scale, args.beta
    )
    for pair in frame_pairs:
        frame_bps = bps.BlendedPolynomialSurface(
            pair.child,
            args.degree,
            args.scale,
            reference_bps.coefficients,
            args.beta,
        )
        bps.cache.onerings(args.submesh_path.name, frame_bps)
        bps_list.append(frame_bps)

    return bps_list


def _construct_bps_list_from_mean(
    reference_pair: mesh.submesh.Pair,
    frame_pairs: list[mesh.submesh.Pair],
    args: argparse.Namespace,
) -> list[bps.BlendedPolynomialSurface]:
    bps_list = []
    reference_bps = mesh.submesh.create_bps_degree_one(
        reference_pair, args.degree, args.scale, args.beta
    )
    sum_coeffs = torch.clone(reference_bps.coefficients)

    # Calculate coefficients from each individual pair.
    for pair in frame_pairs:
        frame_bps = mesh.submesh.create_bps_degree_one(
            pair, args.degree, args.scale, args.beta
        )
        bps.cache.onerings(args.submesh_path.name, frame_bps)
        sum_coeffs += frame_bps.coefficients
        bps_list.append(frame_bps)

    # +1 to account for the reference pair.
    mean_coeffs = sum_coeffs / (len(frame_pairs) + 1)

    # Override coefficients with the mean.
    for bps_ in bps_list:
        bps_.coefficients = mean_coeffs

    return bps_list


def _construct_bps_list_from_weighted_mean(
    reference_pair: mesh.submesh.Pair,
    frame_pairs: list[mesh.submesh.Pair],
    args: argparse.Namespace,
) -> list[bps.BlendedPolynomialSurface]:
    bps_list = []
    reference_bps = mesh.submesh.create_bps_degree_one(
        reference_pair, args.degree, args.scale, args.beta
    )
    # Weight the mean by the local scale at each vertex.
    sum_coeffs = reference_bps.coefficients * reference_bps.vertex_scales[:, None, None]
    weights = reference_bps.vertex_scales[:, None, None].clone()

    # Calculate coefficients from each individual pair.
    for pair in frame_pairs:
        frame_bps = mesh.submesh.create_bps_degree_one(
            pair, args.degree, args.scale, args.beta
        )
        bps.cache.onerings(args.submesh_path.name, frame_bps)
        weight = frame_bps.vertex_scales[:, None, None]
        sum_coeffs += frame_bps.coefficients * weight
        weights += weight
        bps_list.append(frame_bps)

    mean_coeffs = sum_coeffs / weights

    # Override coefficients with the mean.
    for bps_ in bps_list:
        bps_.coefficients = mean_coeffs

    return bps_list


_diff_functions = {
    "vertex-to-vertex": mesh.diff.vertex_to_vertex,
    "vertex-to-mesh": mesh.diff.vertex_to_mesh,
    None: None,
}


def _add_diff_colors(mesh_: mesh.TriangleMesh, diff_: torch.Tensor) -> None:
    colors = torch.ones_like(mesh_.vertices)
    colors -= torch.tensor([[0, 1, 1]]) * diff_[:, None] / diff_.max()
    mesh_.update(vertex_colors=colors)


def plot_diffs(args: argparse.Namespace, output_name: str) -> None:
    """Plot diff data from JSON files."""
    data = [
        io.read_json(io.output_dir() / name / "frame-diff.json")
        for name in args.diff_names
    ]
    fig = plot.diff_comparison(data, args.dataset_names or range(len(data)))
    io.write_figure(fig, io.output_dir() / f"diff-{output_name}.svg")


def screenshot_mesh(args: argparse.Namespace, output_name: str) -> None:
    """Take screenshots of a mesh."""
    camera_view = None
    if cv := args.camera_view:
        camera_view = {"custom": {"front": cv[:3], "up": cv[3:]}}

    output_dir = io.output_dir("screenshots")

    for i, path in enumerate(args.mesh_path):
        mesh_ = io.read_mesh(path)
        screenshot.mesh(mesh_, output_dir, f"{output_name}-{i}", camera_view, args.zoom)


def screenshot_polyline(args: argparse.Namespace, output_name: str) -> None:
    """Take screenshots of a polyline deformation."""
    camera_view = None
    if cv := args.camera_view:
        camera_view = {"custom": {"front": cv[:3], "up": cv[3:]}}

    output_dir = io.output_dir("screenshots") / "polylines" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    polyline = io.read_polyline(args.polyline_path)

    for i in range(args.num_frames):
        mesh_ = bps.render.surface(
            polyline.get_frame(i / (args.num_frames - 1)), args.resolution
        )
        screenshot.mesh(
            mesh_,
            output_dir,
            f"{output_name}-r{args.resolution}-{i}",
            camera_view,
            args.zoom,
        )


def deform_bps(args: argparse.Namespace, output_name: str) -> None:  # noqa: C901 (complexity)
    """Construct a BPS deformation and save the resulting polyline."""
    if (num_frames := len(args.frame_mesh_paths)) < 2:
        raise ValueError(f"expected at least 2 frames but received {num_frames}")
    if args.method == "arap" and num_frames != 2:
        raise ValueError(
            f"must provide exactly 2 frames for arap method but received {num_frames}"
        )

    child = io.read_mesh(args.submesh_path)
    parent = io.read_mesh(args.parent_mesh_path)
    reference_pair = mesh.submesh.Pair(child, parent)

    frame_pairs = [
        mesh.submesh.new_frame(reference_pair, io.read_mesh(p))
        for p in args.frame_mesh_paths
    ]

    # Construct BPSs for each polyline vertex.
    if args.coefficient_transfer_method == "use-reference":
        bps_list = _construct_bps_list_from_reference(reference_pair, frame_pairs, args)
    elif args.coefficient_transfer_method == "mean-simple":
        bps_list = _construct_bps_list_from_mean(reference_pair, frame_pairs, args)
    elif args.coefficient_transfer_method == "mean-weighted":
        bps_list = _construct_bps_list_from_weighted_mean(
            reference_pair, frame_pairs, args
        )
    else:
        bps_list = _construct_bps_list_individual(reference_pair, frame_pairs, args)

    output_path_base = (
        f"{output_name}-{args.method}-{args.coefficient_transfer_method}"
        f"-{args.optimization_algorithm}-r{args.resolution}-f{args.num_frames}"
    )

    if args.method == "linear":
        keyframes = bps_list
        polyline = deform.bps.Polyline(keyframes)
    elif args.method == "arap":
        polyline = deform.bps.optimize_bps_arap(
            bps_list[0],
            bps_list[1],
            args.resolution,
            args.num_frames,
            method=args.optimization_algorithm,
        )
    elif args.method == "arap-alternating":
        polyline_proxy_only = deform.bps.optimize_bps_arap_proxy_only(
            bps_list[0],
            bps_list[1],
            0,
            args.num_frames,
            method=args.optimization_algorithm,
        )
        io.write_polyline(
            polyline_proxy_only,
            io.output_dir("polylines") / f"{output_path_base}-proxy-only.polyline",
        )
        polyline = deform.bps.optimize_bps_arap_coefficients_only(
            bps_list[0],
            bps_list[1],
            args.resolution,
            args.num_frames,
            polyline_proxy_only.get_frame(0.5),
            method=args.optimization_algorithm,
        )
    elif args.method == "progressive":
        polyline = deform.bps.optimize_bps_arap_proxy_only(
            bps_list[0],
            bps_list[1],
            0,
            args.num_frames,
            method=args.optimization_algorithm,
        )
        for resolution in range(args.resolution):
            polyline = deform.bps.optimize_bps_arap_coefficients_only(
                bps_list[0],
                bps_list[1],
                resolution,
                args.num_frames,
                polyline.get_frame(0.5),
                method=args.optimization_algorithm,
            )
            polyline = deform.bps.optimize_bps_arap_proxy_only(
                bps_list[0],
                bps_list[1],
                resolution,
                args.num_frames,
                polyline.get_frame(0.5),
                method=args.optimization_algorithm,
            )

    io.write_polyline(
        polyline,
        io.output_dir("polylines") / f"{output_path_base}.polyline",
    )


def save_polyline_meshes(args: argparse.Namespace, output_name: str) -> None:
    """Save a sequence of meshes corresponding to a polyline."""
    polyline = io.read_polyline(args.polyline_path)

    for i in range(args.num_frames):
        mesh_ = bps.render.surface(
            polyline.get_frame(i / (args.num_frames - 1)), args.resolution
        )
        io.write_mesh(
            mesh_, io.output_dir("polylines/meshes") / f"{output_name}-{i}.ply"
        )


def save_trimesh_polyline_meshes(args: argparse.Namespace, output_name: str) -> None:
    """Save a sequence of meshes corresponding to a trimesh polyline."""
    meshes = [io.read_mesh(path) for path in args.mesh_paths]
    polyline = deform.mesh.Polyline(meshes)

    for i in range(args.num_frames):
        mesh_ = polyline.get_frame(i / (args.num_frames - 1))
        io.write_mesh(
            mesh_,
            io.output_dir("polylines/trimesh-meshes") / f"{output_name}-{i}.ply",
        )


def trimesh_deformation_energy(args: argparse.Namespace, output_name: str) -> None:
    """Calculate the ARAP energy of a trimesh polyline deformation."""
    if len(args.mesh_paths) < 2:
        raise ValueError("must have at least two meshes")

    meshes = [io.read_mesh(path) for path in args.mesh_paths]
    polyline = deform.mesh.Polyline(meshes)
    energy_dist = polyline.symmetric_energy_distribution(
        deform.mesh.energy_function(metrics.arap_regularized), args.num_frames
    )
    torch.save(
        energy_dist,
        io.output_dir("energy-distributions")
        / f"{output_name}-trimesh-{args.num_frames}f.pt",
    )
    print(energy_dist.sum().item())


def deformation_energy(args: argparse.Namespace, output_name: str) -> None:
    """Calculate the energy of a BPS polyline deformation.

    The energy distribution tensors are saved in the `energy-distributions`
    output directory.
    """
    polyline = io.read_polyline(args.polyline_path)

    if args.metric in ["arap", "aiap"]:
        metric = (
            metrics.arap_regularized
            if args.metric == "arap"
            else metrics.aiap_regularized
        )
        energy_dist = polyline.symmetric_energy_distribution(
            deform.bps.energy_function(metric, args.resolution),
            args.num_frames,
        )
    elif args.metric == "surface-area":
        energy_dist = torch.zeros(args.num_frames)
        for i in range(args.num_frames):
            frame = polyline.get_frame(i / (args.num_frames - 1))
            energy_dist[i] = (
                bps.render.surface(frame, args.resolution)
                .open3d_legacy()
                .get_surface_area()
            )

    print(energy_dist.sum().item())
    output_path = (
        io.output_dir("energy-distributions")
        / f"{output_name}-r{args.resolution}-{args.num_frames}f-{args.metric}.pt"
    )
    print(f"Writing energy distribution to {output_path}")
    torch.save(energy_dist, output_path)


def plot_deformation_energy(args: argparse.Namespace, output_name: str) -> None:
    """Plot several deformation energy distributions against each other."""
    distributions = [torch.load(path) for path in args.distribution_paths]
    fig = plot.energy_distributions(
        distributions, args.labels or list(map(str, range(len(distributions))))
    )
    io.write_figure(
        fig,
        io.output_dir("energy-distributions/plots")
        / f"energy-distribution-{output_name}.svg",
    )
    io.write_figure(
        fig,
        io.output_dir("energy-distributions/plots")
        / f"energy-distribution-{output_name}.png",
    )
