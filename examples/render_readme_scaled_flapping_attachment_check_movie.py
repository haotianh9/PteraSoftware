"""Render raw-coordinate wing/wake attachment checks for flapping snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import (  # noqa: E402
    free_flight_readme_scaled_flapping_streamwise as single_case,
)
from examples.render_readme_scaled_flapping_wake_movie import (  # noqa: E402
    _body_states_for_render,
    _build_airplanes,
    _FfmpegRawVideoWriter,
    _infer_global_step_offset,
    _load_snapshot,
    _pair_wake_vertices_by_body_wing,
    _panel_vertex,
    _snapshot_paths,
    _snapshot_render_step,
    _trailing_edge_lines_E,
    _vertices_GP_to_E,
)


def _panel_edges_E(
    airplane: Any,
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
) -> list[np.ndarray]:
    """Return closed panel-edge polylines in Earth axes."""
    panel_edges: list[np.ndarray] = []
    for wing in airplane.wings:
        if wing.panels is None:
            continue
        for panel in np.ravel(wing.panels):
            vertices_GP = np.vstack(
                (
                    _panel_vertex(panel, "Flpp"),
                    _panel_vertex(panel, "Frpp"),
                    _panel_vertex(panel, "Brpp"),
                    _panel_vertex(panel, "Blpp"),
                    _panel_vertex(panel, "Flpp"),
                )
            )
            panel_edges.append(
                _vertices_GP_to_E(
                    vertices_GP_Cg=vertices_GP,
                    position_E_E=position_E_E,
                    R_pas_E_to_BP=R_pas_E_to_BP,
                )
            )
    return panel_edges


def _wake_row_polyline_E(
    wing_wake_vertices_E: np.ndarray, row_index: int
) -> np.ndarray:
    """Return a wake row front-edge polyline from a body/wing wake array."""
    if wing_wake_vertices_E.size == 0 or row_index >= wing_wake_vertices_E.shape[0]:
        return np.empty((0, 3), dtype=float)
    row = wing_wake_vertices_E[row_index]
    points = [row[0, 0]]
    points.extend(row[:, 1])
    return np.asarray(points, dtype=float)


def _attachment_gap_m(
    trailing_edge_E: np.ndarray,
    wake_front_row_E: np.ndarray,
) -> float:
    """Return the largest matched-point distance between TE and newest wake row."""
    if trailing_edge_E.shape != wake_front_row_E.shape or trailing_edge_E.size == 0:
        return float("nan")
    return float(np.max(np.linalg.norm(trailing_edge_E - wake_front_row_E, axis=1)))


def _axis_limits(
    points: list[np.ndarray], components: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded axis limits from a collection of point arrays."""
    finite_points = [
        point_array[:, list(components)]
        for point_array in points
        if point_array.size and np.all(np.isfinite(point_array))
    ]
    if not finite_points:
        return (-0.5, 0.5), (-0.5, 0.5)
    stacked = np.vstack(finite_points)
    mins = stacked.min(axis=0)
    maxes = stacked.max(axis=0)
    span = np.maximum(maxes - mins, 1e-3)
    pad = np.maximum(0.08 * span, np.array([0.02, 0.02]))
    return (float(mins[0] - pad[0]), float(maxes[0] + pad[0])), (
        float(mins[1] - pad[1]),
        float(maxes[1] + pad[1]),
    )


def _draw_body_views(
    axes: tuple[plt.Axes, plt.Axes],
    body_name: str,
    panel_edges_E: list[np.ndarray],
    trailing_edges_E: list[np.ndarray],
    wing_wakes_E: list[np.ndarray],
    color: str,
    wake_rows: int,
) -> None:
    """Draw top and side attachment diagnostics for one body."""
    ax_xy, ax_xz = axes
    plotted_points: list[np.ndarray] = []
    for edge_E in panel_edges_E:
        plotted_points.append(edge_E)
        ax_xy.plot(edge_E[:, 0], edge_E[:, 1], color="0.25", lw=0.7, alpha=0.75)
        ax_xz.plot(edge_E[:, 0], edge_E[:, 2], color="0.25", lw=0.7, alpha=0.75)
    gaps: list[float] = []
    for wing_index, wing_wake_E in enumerate(wing_wakes_E):
        if wing_wake_E.size == 0:
            continue
        max_rows = min(wake_rows, wing_wake_E.shape[0])
        for row_index in range(max_rows - 1, -1, -1):
            row_E = _wake_row_polyline_E(wing_wake_E, row_index)
            if row_E.size == 0:
                continue
            plotted_points.append(row_E)
            alpha = 0.22 if row_index else 1.0
            lw = 0.75 if row_index else 3.2
            ax_xy.plot(row_E[:, 0], row_E[:, 1], color=color, lw=lw, alpha=alpha)
            ax_xz.plot(row_E[:, 0], row_E[:, 2], color=color, lw=lw, alpha=alpha)
        if wing_index < len(trailing_edges_E):
            front_row_E = _wake_row_polyline_E(wing_wake_E, 0)
            gaps.append(_attachment_gap_m(trailing_edges_E[wing_index], front_row_E))
    for trailing_edge_E in trailing_edges_E:
        plotted_points.append(trailing_edge_E)
        ax_xy.plot(trailing_edge_E[:, 0], trailing_edge_E[:, 1], color="black", lw=2.6)
        ax_xz.plot(trailing_edge_E[:, 0], trailing_edge_E[:, 2], color="black", lw=2.6)
        ax_xy.plot(
            trailing_edge_E[:, 0], trailing_edge_E[:, 1], color="#fff6d5", lw=1.5
        )
        ax_xz.plot(
            trailing_edge_E[:, 0], trailing_edge_E[:, 2], color="#fff6d5", lw=1.5
        )

    xy_limits = _axis_limits(plotted_points, (0, 1))
    xz_limits = _axis_limits(plotted_points, (0, 2))
    ax_xy.set_xlim(*xy_limits[0])
    ax_xy.set_ylim(*xy_limits[1])
    ax_xz.set_xlim(*xz_limits[0])
    ax_xz.set_ylim(*xz_limits[1])
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_aspect("equal", adjustable="box")
    finite_gaps = [gap for gap in gaps if np.isfinite(gap)]
    gap_text = (
        f"max TE-new wake gap {max(finite_gaps) * 1000:.1f} mm"
        if finite_gaps
        else "gap unavailable"
    )
    ax_xy.set_title(f"{body_name}: X-Y top view")
    ax_xz.set_title(f"{body_name}: X-Z side view ({gap_text})")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")


def render_attachment_movie(
    run_dir: Path,
    output_path: Path,
    mode: str,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    frame_stride: int,
    wake_rows: int,
    fps: float,
) -> Path:
    """Render a raw-coordinate attachment diagnostic MP4."""
    snapshot_paths = _snapshot_paths(run_dir)[::frame_stride]
    global_offset = _infer_global_step_offset(run_dir)
    max_render_step = max(
        _snapshot_render_step(path, inferred_global_step_offset=global_offset)
        for path in snapshot_paths
    )
    free_steps = max(free_steps, max_render_step - prescribed_steps + 1)
    airplanes = _build_airplanes(
        mode=mode,
        prescribed_steps=prescribed_steps,
        free_steps=free_steps,
        steps_per_flap=steps_per_flap,
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: _FfmpegRawVideoWriter | None = None
    colors = ("#e67e22", "#009ec3")
    try:
        for frame_index, snapshot_path in enumerate(snapshot_paths):
            snapshot = _load_snapshot(snapshot_path)
            local_step = int(snapshot["step"][0])
            render_step = int(
                snapshot.get(
                    "global_step",
                    np.array([global_offset + local_step], dtype=int),
                )[0]
            )
            if mode == "single":
                body_airplanes = [airplanes[render_step]]
                body_positions = [snapshot["position_E_E"]]
                body_rotations = [snapshot["R_pas_E_to_BP1"]]
                body_wakes = _pair_wake_vertices_by_body_wing(
                    snapshot=snapshot,
                    airplanes_at_step=(airplanes[render_step],),
                )
                body_names = ["single"]
            else:
                body_airplanes = list(airplanes[render_step])
                body_positions_E_E, body_R_pas_E_to_BPs = _body_states_for_render(
                    snapshot=snapshot,
                    delta_time=single_case.FLAPPING_PERIOD_S / steps_per_flap,
                )
                body_positions = list(body_positions_E_E)
                body_rotations = list(body_R_pas_E_to_BPs)
                body_wakes = _pair_wake_vertices_by_body_wing(
                    snapshot=snapshot,
                    airplanes_at_step=airplanes[render_step],
                )
                body_names = ["front", "rear"]

            num_bodies = len(body_airplanes)
            fig, axes = plt.subplots(
                num_bodies,
                2,
                figsize=(11.5, 4.5 * num_bodies),
                squeeze=False,
                constrained_layout=True,
            )
            fig.suptitle(
                f"Raw wing/wake attachment check, t={render_step / steps_per_flap:.2f} s, step {render_step}",
                fontsize=13,
            )
            for body_index, airplane in enumerate(body_airplanes):
                panel_edges = _panel_edges_E(
                    airplane=airplane,
                    position_E_E=body_positions[body_index],
                    R_pas_E_to_BP=body_rotations[body_index],
                )
                trailing_edges = _trailing_edge_lines_E(
                    airplane=airplane,
                    position_E_E=body_positions[body_index],
                    R_pas_E_to_BP=body_rotations[body_index],
                )
                _draw_body_views(
                    axes=(axes[body_index, 0], axes[body_index, 1]),
                    body_name=body_names[body_index],
                    panel_edges_E=panel_edges,
                    trailing_edges_E=trailing_edges,
                    wing_wakes_E=body_wakes[body_index],
                    color=colors[body_index % len(colors)],
                    wake_rows=wake_rows,
                )
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = rgba[:, :, :3].copy()
            plt.close(fig)
            if writer is None:
                writer = _FfmpegRawVideoWriter(
                    output_path=output_path,
                    fps=fps,
                    frame_shape=rgb.shape,
                )
            writer.append_data(rgb)
            print(
                f"rendered attachment frame {frame_index + 1}/{len(snapshot_paths)} "
                f"from {snapshot_path.name}"
            )
    finally:
        if writer is not None:
            writer.close()
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "pair"), default="pair")
    parser.add_argument(
        "--prescribed-steps", type=int, default=3 * single_case.DEFAULT_STEPS_PER_FLAP
    )
    parser.add_argument(
        "--free-steps", type=int, default=20 * single_case.DEFAULT_STEPS_PER_FLAP
    )
    parser.add_argument(
        "--steps-per-flap", type=int, default=single_case.DEFAULT_STEPS_PER_FLAP
    )
    parser.add_argument("--x-over-span", type=float, default=2.0)
    parser.add_argument("--y-over-span", type=float, default=0.0)
    parser.add_argument("--z-over-span", type=float, default=0.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--wake-rows", type=int, default=24)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    """Render the requested attachment movie."""
    args = parse_args()
    movie_path = render_attachment_movie(
        run_dir=args.run_dir,
        output_path=args.output_path,
        mode=args.mode,
        prescribed_steps=args.prescribed_steps,
        free_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        x_over_span=args.x_over_span,
        y_over_span=args.y_over_span,
        z_over_span=args.z_over_span,
        frame_stride=args.frame_stride,
        wake_rows=args.wake_rows,
        fps=args.fps,
    )
    print(f"Saved attachment movie to: {movie_path}")


if __name__ == "__main__":
    main()
