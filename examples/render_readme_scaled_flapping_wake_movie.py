"""Render near-body body+wake movies from streamed flapping snapshots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pterasoftware import _transformations

try:
    from examples import free_flight_readme_scaled_flapping_streamwise as single_case
    from examples import (
        free_flight_readme_scaled_flapping_two_flyer_streamwise as pair_case,
    )
except ImportError:
    import free_flight_readme_scaled_flapping_streamwise as single_case
    import free_flight_readme_scaled_flapping_two_flyer_streamwise as pair_case


DEFAULT_SINGLE_RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "readme_scaled_flapping_streamwise"
    / "single_free_x_20p_resolved_wake"
)
DEFAULT_STEPS_PER_FLAP = single_case.DEFAULT_STEPS_PER_FLAP


class _FfmpegRawVideoWriter:
    """Small RGB-frame writer that uses the system ffmpeg binary."""

    def __init__(
        self, output_path: Path, fps: float, frame_shape: tuple[int, int, int]
    ):
        height, width, channels = frame_shape
        if channels != 3:
            raise ValueError(f"Expected RGB frames with 3 channels, got {channels}")
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame_rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self._process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        return_code = self._process.wait()
        if return_code:
            raise RuntimeError(
                "ffmpeg movie encoding failed with return code "
                f"{return_code}:\n{stderr.decode(errors='replace')}"
            )


def _to_pyvista_axes(points_E: np.ndarray) -> np.ndarray:
    """Convert Earth-axis points to PyVista's plotting axes."""
    T_pas_E_to_V = _transformations.generate_rot_T(
        angles=(0.0, 180.0, 0.0),
        passive=True,
        intrinsic=True,
        order="xyz",
    )
    return _transformations.apply_T_to_vectors(T_pas_E_to_V, points_E, has_point=True)


def _vertices_GP_to_E(
    vertices_GP_Cg: np.ndarray,
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
) -> np.ndarray:
    """Transform geometry-axis vertices about the focal CG into Earth axes."""
    T_pas_GP_to_BP = _transformations.generate_rot_T(
        angles=(0.0, 180.0, 0.0),
        passive=True,
        intrinsic=True,
        order="xyz",
    )
    T_pas_E_Cg_to_BP = np.eye(4, dtype=float)
    T_pas_E_Cg_to_BP[:3, :3] = R_pas_E_to_BP
    T_pas_BP_to_E_Cg = _transformations.invert_T_pas(T_pas_E_Cg_to_BP)
    T_pas_E_Cg_to_E = _transformations.generate_trans_T(
        translations=-position_E_E,
        passive=True,
    )
    T_pas_GP_to_E = _transformations.compose_T_pas(
        T_pas_GP_to_BP,
        T_pas_BP_to_E_Cg,
        T_pas_E_Cg_to_E,
    )
    return _transformations.apply_T_to_vectors(
        T_pas_GP_to_E,
        vertices_GP_Cg,
        has_point=True,
    )


def _vertices_GP_to_BP(vertices_GP_Cg: np.ndarray) -> np.ndarray:
    """Transform geometry-axis vertices to body axes about the focal CG."""
    T_pas_GP_to_BP = _transformations.generate_rot_T(
        angles=(0.0, 180.0, 0.0),
        passive=True,
        intrinsic=True,
        order="xyz",
    )
    return _transformations.apply_T_to_vectors(
        T_pas_GP_to_BP,
        vertices_GP_Cg,
        has_point=True,
    )


def _panel_vertex_to_1d(vertex: np.ndarray) -> np.ndarray:
    """Coerce a panel vertex into a flat three-component point."""
    flat = np.asarray(vertex, dtype=float).reshape(-1)
    if flat.size != 3:
        raise ValueError(
            f"Expected a three-component panel vertex, got shape {vertex.shape}"
        )
    return flat


def _panel_vertex(panel: Any, vertex_name: str) -> np.ndarray:
    """Return a panel vertex in the airplane-local geometry frame."""
    vertex = getattr(panel, f"{vertex_name}_G_Cg", None)
    if vertex is None:
        vertex = getattr(panel, f"{vertex_name}_GP1_CgP1")
    return _panel_vertex_to_1d(vertex)


def _panel_surface_free_flight(
    airplane: Any,
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
) -> pv.PolyData:
    """Build a free-flight panel surface robustly for flapping-wing panel arrays."""
    vertices_GP_Cg = np.empty((0, 3), dtype=float)
    faces = np.empty(0, dtype=int)
    panel_num = 0

    for wing in airplane.wings:
        if wing.panels is None:
            continue
        for panel in np.ravel(wing.panels):
            panel_vertices = np.vstack(
                (
                    _panel_vertex(panel, "Flpp"),
                    _panel_vertex(panel, "Frpp"),
                    _panel_vertex(panel, "Brpp"),
                    _panel_vertex(panel, "Blpp"),
                )
            )
            face = np.array(
                [
                    4,
                    4 * panel_num,
                    4 * panel_num + 1,
                    4 * panel_num + 2,
                    4 * panel_num + 3,
                ],
                dtype=int,
            )
            vertices_GP_Cg = np.vstack((vertices_GP_Cg, panel_vertices))
            faces = np.hstack((faces, face))
            panel_num += 1

    vertices_E = _vertices_GP_to_E(
        vertices_GP_Cg=vertices_GP_Cg,
        position_E_E=position_E_E,
        R_pas_E_to_BP=R_pas_E_to_BP,
    )
    return pv.PolyData(_to_pyvista_axes(vertices_E), faces)


def _line_polydata_from_points_E(points_E: np.ndarray) -> pv.PolyData:
    """Build one connected PyVista polyline from Earth-frame points."""
    points_E = np.asarray(points_E, dtype=float)
    if points_E.shape[0] < 2:
        return pv.PolyData()
    points_V = _to_pyvista_axes(points_E)
    lines = np.concatenate(
        (
            np.array([points_V.shape[0]], dtype=int),
            np.arange(points_V.shape[0], dtype=int),
        )
    )
    return pv.PolyData(points_V, lines=lines)


def _trailing_edge_lines_E(
    airplane: Any,
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
) -> list[np.ndarray]:
    """Return one trailing-edge polyline per wing in Earth axes."""
    trailing_edge_lines: list[np.ndarray] = []
    for wing in airplane.wings:
        if wing.panels is None:
            continue
        trailing_edge_panels = [
            panel
            for panel in np.ravel(wing.panels)
            if getattr(panel, "is_trailing_edge", False)
        ]
        trailing_edge_panels.sort(
            key=lambda panel: int(getattr(panel, "local_spanwise_position", 0))
        )
        if not trailing_edge_panels:
            continue
        line_GP_Cg = [_panel_vertex(trailing_edge_panels[0], "Blpp")]
        line_GP_Cg.extend(
            _panel_vertex(panel, "Brpp") for panel in trailing_edge_panels
        )
        trailing_edge_lines.append(
            _vertices_GP_to_E(
                vertices_GP_Cg=np.asarray(line_GP_Cg, dtype=float),
                position_E_E=position_E_E,
                R_pas_E_to_BP=R_pas_E_to_BP,
            )
        )
    return trailing_edge_lines


def _snapshot_paths(run_dir: Path) -> list[Path]:
    """Return streamed heavy snapshot paths in order."""
    paths = sorted((run_dir / "streamed_history").glob("step_*.npz"))
    if not paths:
        paths = sorted(
            (run_dir / "segments").glob("segment_*/streamed_history/step_*.npz")
        )
    if not paths:
        raise FileNotFoundError(f"No step_*.npz files found under {run_dir}")
    return paths


def _snapshot_step(path: Path) -> int:
    """Read a snapshot step without keeping arrays in memory."""
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"][0])


def _infer_global_step_offset(run_dir: Path) -> int:
    """Infer a restart segment's global-step offset from its checkpoint names."""
    checkpoint_paths = sorted(
        (run_dir / "restart_checkpoints").glob("restart_step_*.npz")
    )
    if not checkpoint_paths:
        return 0
    offsets: list[int] = []
    for checkpoint_path in checkpoint_paths:
        try:
            offsets.append(int(checkpoint_path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            with np.load(checkpoint_path, allow_pickle=True) as data:
                if "global_step" in data.files:
                    offsets.append(int(data["global_step"][0]))
    return min(offsets) if offsets else 0


def _snapshot_render_step(
    path: Path,
    inferred_global_step_offset: int,
) -> int:
    """Return the movement index that matches the saved wake/body snapshot."""
    with np.load(path, allow_pickle=True) as data:
        if "global_step" in data.files:
            return int(data["global_step"][0])
        segment_offset = _infer_global_step_offset(path.parent.parent)
        if segment_offset:
            return segment_offset + int(data["step"][0])
        return inferred_global_step_offset + int(data["step"][0])


def _snapshot_render_step_from_snapshot(
    path: Path,
    snapshot: dict[str, np.ndarray],
    inferred_global_step_offset: int,
) -> int:
    """Return the movement index for an already-loaded snapshot."""
    if "global_step" in snapshot:
        return int(snapshot["global_step"][0])
    segment_offset = _infer_global_step_offset(path.parent.parent)
    if segment_offset:
        return segment_offset + int(snapshot["step"][0])
    return inferred_global_step_offset + int(snapshot["step"][0])


def _load_snapshot(path: Path) -> dict[str, np.ndarray]:
    """Load one streamed snapshot into memory."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def _body_states_for_render(
    snapshot: dict[str, np.ndarray],
    delta_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return body poses consistent with the saved current wake state.

    New snapshots store the start-of-step/current pose explicitly. Older snapshots
    only stored the post-MuJoCo pose, so use the streamwise velocity to undo one
    time step. That removes an artificial wake/wing gap of approximately ``U dt`` in
    legacy movies.
    """
    if "current_positions_E_E" in snapshot:
        positions_E_E = snapshot["current_positions_E_E"]
    else:
        positions_E_E = snapshot["positions_E_E"].copy()
        if "velocities_E__E" in snapshot:
            positions_E_E = positions_E_E - snapshot["velocities_E__E"] * delta_time
    if "current_R_pas_E_to_BPs" in snapshot:
        rotations = snapshot["current_R_pas_E_to_BPs"]
    else:
        rotations = snapshot["R_pas_E_to_BPs"]
    return positions_E_E, rotations


def _cropped_wake_surface(
    snapshot: dict[str, np.ndarray],
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
    x_window_m: tuple[float, float],
    y_half_width_m: float,
    z_half_width_m: float,
    wake_ring_stride: int,
) -> pv.PolyData:
    """Build a cropped wake-ring surface near the focal body.

    Multibody streamed wake coordinates are already in the shared Earth-frame solve
    space, despite the legacy internal ``GP1_CgP1`` variable names. Do not transform
    them from body/geometry axes again here.
    """
    vertices_E = np.stack(
        [
            snapshot["wake_fl"],
            snapshot["wake_fr"],
            snapshot["wake_br"],
            snapshot["wake_bl"],
        ],
        axis=1,
    )
    if vertices_E.size == 0:
        return pv.PolyData()

    relative_vertices_E = vertices_E - np.asarray(position_E_E, dtype=float)
    vertices_BP = np.einsum(
        "ij,...j->...i",
        np.asarray(R_pas_E_to_BP, dtype=float),
        relative_vertices_E,
    )
    keep = np.any(
        (
            (vertices_BP[:, :, 0] >= x_window_m[0])
            & (vertices_BP[:, :, 0] <= x_window_m[1])
            & (np.abs(vertices_BP[:, :, 1]) <= y_half_width_m)
            & (np.abs(vertices_BP[:, :, 2]) <= z_half_width_m)
        ),
        axis=1,
    )
    vertices_E = vertices_E[keep]
    vertices_E = vertices_E[::wake_ring_stride]
    if vertices_E.size == 0:
        return pv.PolyData()

    vertices_V = _to_pyvista_axes(vertices_E.reshape(-1, 3))
    num_rings = vertices_E.shape[0]
    faces = np.empty(num_rings * 5, dtype=int)
    faces[0::5] = 4
    base = np.arange(num_rings, dtype=int) * 4
    faces[1::5] = base
    faces[2::5] = base + 1
    faces[3::5] = base + 2
    faces[4::5] = base + 3
    return pv.PolyData(vertices_V, faces)


def _wake_surface_from_vertices_E(vertices_E: np.ndarray) -> pv.PolyData:
    """Build a wake ring surface from Earth-frame ring vertices."""
    if vertices_E.size == 0:
        return pv.PolyData()
    vertices_V = _to_pyvista_axes(vertices_E.reshape(-1, 3))
    num_rings = vertices_E.shape[0]
    faces = np.empty(num_rings * 5, dtype=int)
    faces[0::5] = 4
    base = np.arange(num_rings, dtype=int) * 4
    faces[1::5] = base
    faces[2::5] = base + 1
    faces[3::5] = base + 2
    faces[4::5] = base + 3
    return pv.PolyData(vertices_V, faces)


def _crop_wake_vertices_E(
    vertices_E: np.ndarray,
    position_E_E: np.ndarray,
    R_pas_E_to_BP: np.ndarray,
    x_window_m: tuple[float, float],
    y_half_width_m: float,
    z_half_width_m: float,
    wake_ring_stride: int,
) -> np.ndarray:
    """Return cropped Earth-frame wake-ring vertices near the focal body."""
    if vertices_E.size == 0:
        return vertices_E
    relative_vertices_E = vertices_E - np.asarray(position_E_E, dtype=float)
    vertices_BP = np.einsum(
        "ij,...j->...i",
        np.asarray(R_pas_E_to_BP, dtype=float),
        relative_vertices_E,
    )
    keep = np.any(
        (
            (vertices_BP[:, :, 0] >= x_window_m[0])
            & (vertices_BP[:, :, 0] <= x_window_m[1])
            & (np.abs(vertices_BP[:, :, 1]) <= y_half_width_m)
            & (np.abs(vertices_BP[:, :, 2]) <= z_half_width_m)
        ),
        axis=1,
    )
    return vertices_E[keep][::wake_ring_stride]


def _pair_wake_vertices_by_body(
    snapshot: dict[str, np.ndarray],
    airplanes_at_step: tuple[Any, ...],
    wake_history_rows: int | None,
) -> list[np.ndarray]:
    """Split flattened pair wake rings into one Earth-frame ring array per body."""
    vertices_E = np.stack(
        [
            snapshot["wake_fl"],
            snapshot["wake_fr"],
            snapshot["wake_br"],
            snapshot["wake_bl"],
        ],
        axis=1,
    )
    total_spanwise = sum(
        wing.num_spanwise_panels
        for airplane in airplanes_at_step
        for wing in airplane.wings
    )
    if total_spanwise == 0 or vertices_E.shape[0] == 0:
        return [np.empty((0, 4, 3), dtype=float) for _ in airplanes_at_step]
    if vertices_E.shape[0] % total_spanwise != 0:
        raise ValueError(
            "Snapshot wake-ring count is incompatible with the airplane spanwise "
            "panel counts."
        )
    num_rows = vertices_E.shape[0] // total_spanwise
    if wake_history_rows is None:
        rows_to_keep = num_rows
    else:
        rows_to_keep = min(max(int(wake_history_rows), 1), num_rows)

    body_vertices: list[list[np.ndarray]] = [[] for _ in airplanes_at_step]
    start = 0
    for body_index, airplane in enumerate(airplanes_at_step):
        for wing in airplane.wings:
            num_spanwise = wing.num_spanwise_panels
            wing_count = num_rows * num_spanwise
            wing_vertices = vertices_E[start : start + wing_count].reshape(
                num_rows,
                num_spanwise,
                4,
                3,
            )
            body_vertices[body_index].append(
                wing_vertices[:rows_to_keep].reshape(-1, 4, 3)
            )
            start += wing_count
    return [
        np.concatenate(parts, axis=0) if parts else np.empty((0, 4, 3), dtype=float)
        for parts in body_vertices
    ]


def _pair_wake_vertices_by_body_wing(
    snapshot: dict[str, np.ndarray],
    airplanes_at_step: tuple[Any, ...],
) -> list[list[np.ndarray]]:
    """Split flattened pair wake rings into body/wing row arrays."""
    vertices_E = np.stack(
        [
            snapshot["wake_fl"],
            snapshot["wake_fr"],
            snapshot["wake_br"],
            snapshot["wake_bl"],
        ],
        axis=1,
    )
    total_spanwise = sum(
        wing.num_spanwise_panels
        for airplane in airplanes_at_step
        for wing in airplane.wings
    )
    if total_spanwise == 0 or vertices_E.shape[0] == 0:
        return [[] for _ in airplanes_at_step]
    if vertices_E.shape[0] % total_spanwise != 0:
        raise ValueError(
            "Snapshot wake-ring count is incompatible with the airplane spanwise "
            "panel counts."
        )
    num_rows = vertices_E.shape[0] // total_spanwise
    body_wing_vertices: list[list[np.ndarray]] = [[] for _ in airplanes_at_step]
    start = 0
    for body_index, airplane in enumerate(airplanes_at_step):
        for wing in airplane.wings:
            num_spanwise = wing.num_spanwise_panels
            wing_count = num_rows * num_spanwise
            body_wing_vertices[body_index].append(
                vertices_E[start : start + wing_count].reshape(
                    num_rows,
                    num_spanwise,
                    4,
                    3,
                )
            )
            start += wing_count
    return body_wing_vertices


def _wake_front_row_line_E(wing_wake_vertices_E: np.ndarray) -> np.ndarray:
    """Return the newest wake row's front edge as an Earth-frame polyline."""
    if wing_wake_vertices_E.size == 0:
        return np.empty((0, 3), dtype=float)
    newest_row = wing_wake_vertices_E[0]
    line_points = [newest_row[0, 0]]
    line_points.extend(newest_row[:, 1])
    return np.asarray(line_points, dtype=float)


def _wake_row_spanline_E(
    wing_wake_vertices_E: np.ndarray,
    row_index: int,
) -> np.ndarray:
    """Return one spanwise wake row as an Earth-frame polyline."""
    if wing_wake_vertices_E.size == 0 or row_index >= wing_wake_vertices_E.shape[0]:
        return np.empty((0, 3), dtype=float)
    row = wing_wake_vertices_E[row_index]
    line_points = [row[0, 0]]
    line_points.extend(row[:, 1])
    return np.asarray(line_points, dtype=float)


def _wake_stationline_E(
    wing_wake_vertices_E: np.ndarray,
    station_index: int,
    num_rows: int,
) -> np.ndarray:
    """Return one streamwise wake grid line at a span station."""
    if wing_wake_vertices_E.size == 0:
        return np.empty((0, 3), dtype=float)
    num_rows = min(num_rows, wing_wake_vertices_E.shape[0])
    if station_index == 0:
        return wing_wake_vertices_E[:num_rows, 0, 0]
    return wing_wake_vertices_E[:num_rows, station_index - 1, 1]


def _add_wake_wire_rows(
    plotter: pv.Plotter,
    wing_wake_vertices_E: np.ndarray,
    color: str,
    max_rows: int | None,
    row_stride: int,
    span_station_stride: int = 2,
) -> None:
    """Add a readable age-faded wake wire instead of an opaque wake cloud."""
    if wing_wake_vertices_E.size == 0:
        return
    num_rows = wing_wake_vertices_E.shape[0]
    if max_rows is not None:
        num_rows = min(num_rows, max(1, int(max_rows)))
    row_stride = max(1, int(row_stride))
    span_station_stride = max(1, int(span_station_stride))

    # Draw old rows first and newest rows last, so the attachment region stays visible.
    row_indices = list(range(0, num_rows, row_stride))
    for draw_order, row_index in enumerate(reversed(row_indices)):
        line_E = _wake_row_spanline_E(wing_wake_vertices_E, row_index)
        line = _line_polydata_from_points_E(line_E)
        if line.n_points == 0:
            continue
        age_fraction = row_index / max(num_rows - 1, 1)
        opacity = float(0.18 + 0.72 * (1.0 - age_fraction) ** 1.6)
        line_width = float(1.1 + 2.4 * (1.0 - age_fraction) ** 1.4)
        if draw_order == len(row_indices) - 1:
            line_width = max(line_width, 4.0)
            opacity = 1.0
        plotter.add_mesh(
            line,
            color=color,
            opacity=opacity,
            line_width=line_width,
            render_lines_as_tubes=True,
        )

    num_spanwise = wing_wake_vertices_E.shape[1]
    station_indices = list(range(0, num_spanwise + 1, span_station_stride))
    if station_indices[-1] != num_spanwise:
        station_indices.append(num_spanwise)
    for station_index in station_indices:
        line_E = _wake_stationline_E(
            wing_wake_vertices_E=wing_wake_vertices_E,
            station_index=station_index,
            num_rows=num_rows,
        )
        line = _line_polydata_from_points_E(line_E[::row_stride])
        if line.n_points == 0:
            continue
        plotter.add_mesh(
            line,
            color=color,
            opacity=0.34,
            line_width=1.15,
            render_lines_as_tubes=True,
        )


def _camera_direction_and_up(
    camera_view: str,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return a stable PyVista camera direction/up pair."""
    if camera_view == "top":
        return np.array([0.0, 0.0, 1.0], dtype=float), (0.0, 1.0, 0.0)
    if camera_view == "side":
        return np.array([-1.0, 0.0, 0.18], dtype=float), (0.0, 0.0, 1.0)
    if camera_view == "rear":
        return np.array([-0.45, -1.0, 0.32], dtype=float), (0.0, 0.0, 1.0)
    return np.array([-1.0, -0.65, 1.25], dtype=float), (0.0, 0.0, 1.0)


def _add_wing_surface(
    plotter: pv.Plotter,
    panel_surface: pv.PolyData,
    color: str,
) -> None:
    """Add an intentionally high-contrast wing surface and outline."""
    plotter.add_mesh(
        panel_surface,
        show_edges=True,
        edge_color="#111111",
        color=color,
        opacity=1.0,
        smooth_shading=False,
        ambient=0.75,
        diffuse=0.35,
        line_width=1.5,
    )
    edges = panel_surface.extract_all_edges()
    if edges.n_points > 0:
        plotter.add_mesh(
            edges,
            color="#fff6d5",
            line_width=5.0,
            render_lines_as_tubes=True,
        )


def _build_airplanes(
    mode: str,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
) -> Any:
    """Build the matching movement airplanes for visualization panels."""
    if mode == "single":
        coupled_problem, _, _ = single_case.build_problem(
            prescribed_num_steps=prescribed_steps,
            free_num_steps=free_steps,
            steps_per_flap=steps_per_flap,
        )
        return coupled_problem.coupled_movement.airplanes
    coupled_problem, _, _ = pair_case.build_problem(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        prescribed_num_steps=prescribed_steps,
        free_num_steps=free_steps,
        steps_per_flap=steps_per_flap,
    )
    return coupled_problem.coupled_movement.airplanes


def render_movie(
    run_dir: Path,
    output_path: Path,
    mode: str,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    fps: float,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    focal_body_index: int,
    x_wake_behind_span: float,
    x_wake_ahead_span: float,
    y_half_width_span: float,
    z_half_width_span: float,
    parallel_scale_span: float,
    frame_stride: int,
    wake_ring_stride: int,
    wake_history_rows: int | None,
    highlight_attachment: bool,
    wake_style: str,
    camera_view: str,
    camera_target: str,
) -> Path:
    """Render a near-body MP4 from streamed snapshots."""
    snapshot_paths = _snapshot_paths(run_dir)
    inferred_global_step_offset = _infer_global_step_offset(run_dir)
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    if wake_ring_stride < 1:
        raise ValueError(f"wake_ring_stride must be positive, got {wake_ring_stride}")
    snapshot_paths = snapshot_paths[::frame_stride]
    max_render_step = max(
        _snapshot_render_step(
            path,
            inferred_global_step_offset=inferred_global_step_offset,
        )
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
    plotter = pv.Plotter(window_size=(1280, 900), off_screen=True, lighting=None)
    plotter.enable_parallel_projection()  # type: ignore[call-arg]
    plotter.set_background("black")  # type: ignore[call-arg]

    camera_direction, camera_up = _camera_direction_and_up(camera_view)
    camera_direction /= np.linalg.norm(camera_direction)
    camera_distance = 5.0 * single_case.FULL_SPAN_M
    x_window_m = (
        -x_wake_behind_span * single_case.FULL_SPAN_M,
        x_wake_ahead_span * single_case.FULL_SPAN_M,
    )
    y_half_width_m = y_half_width_span * single_case.FULL_SPAN_M
    z_half_width_m = z_half_width_span * single_case.FULL_SPAN_M

    try:
        for frame_index, snapshot_path in enumerate(snapshot_paths):
            snapshot = _load_snapshot(snapshot_path)
            local_step = int(snapshot["step"][0])
            render_step = _snapshot_render_step_from_snapshot(
                snapshot_path,
                snapshot,
                inferred_global_step_offset=inferred_global_step_offset,
            )
            if render_step >= len(airplanes):
                continue
            plotter.clear()

            if mode == "single":
                position_E_E = snapshot["position_E_E"]
                R_pas_E_to_BP = snapshot["R_pas_E_to_BP1"]
                panel_surfaces = [
                    _panel_surface_free_flight(
                        airplanes[render_step],
                        position_E_E,
                        R_pas_E_to_BP,
                    )
                ]
            else:
                body_positions_E_E, body_R_pas_E_to_BPs = _body_states_for_render(
                    snapshot=snapshot,
                    delta_time=single_case.FLAPPING_PERIOD_S / steps_per_flap,
                )
                position_E_E = body_positions_E_E[focal_body_index]
                R_pas_E_to_BP = body_R_pas_E_to_BPs[focal_body_index]
                panel_surfaces = [
                    _panel_surface_free_flight(
                        airplanes[render_step][body_index],
                        body_positions_E_E[body_index],
                        body_R_pas_E_to_BPs[body_index],
                    )
                    for body_index in range(len(airplanes[render_step]))
                ]
            camera_target_E_E = position_E_E
            if mode == "pair" and camera_target == "formation":
                camera_target_E_E = np.mean(body_positions_E_E, axis=0)

            wake_colors = ("#ff9f1c", "#2ec4ff", "#ff5d73", "#b7ff4a")
            if wake_style == "wire":
                if mode == "single":
                    body_wing_wakes = _pair_wake_vertices_by_body_wing(
                        snapshot=snapshot,
                        airplanes_at_step=(airplanes[render_step],),
                    )
                else:
                    body_wing_wakes = _pair_wake_vertices_by_body_wing(
                        snapshot=snapshot,
                        airplanes_at_step=airplanes[render_step],
                    )
                for body_index, body_wakes in enumerate(body_wing_wakes):
                    for wing_wake_vertices_E in body_wakes:
                        _add_wake_wire_rows(
                            plotter=plotter,
                            wing_wake_vertices_E=wing_wake_vertices_E,
                            color=wake_colors[body_index % len(wake_colors)],
                            max_rows=wake_history_rows,
                            row_stride=wake_ring_stride,
                        )
            else:
                if mode == "single":
                    wake_surfaces = [
                        _cropped_wake_surface(
                            snapshot=snapshot,
                            position_E_E=position_E_E,
                            R_pas_E_to_BP=R_pas_E_to_BP,
                            x_window_m=x_window_m,
                            y_half_width_m=y_half_width_m,
                            z_half_width_m=z_half_width_m,
                            wake_ring_stride=wake_ring_stride,
                        )
                    ]
                else:
                    wake_surfaces = [
                        _wake_surface_from_vertices_E(
                            _crop_wake_vertices_E(
                                vertices_E=body_wake_vertices,
                                position_E_E=position_E_E,
                                R_pas_E_to_BP=R_pas_E_to_BP,
                                x_window_m=x_window_m,
                                y_half_width_m=y_half_width_m,
                                z_half_width_m=z_half_width_m,
                                wake_ring_stride=wake_ring_stride,
                            )
                        )
                        for body_wake_vertices in _pair_wake_vertices_by_body(
                            snapshot=snapshot,
                            airplanes_at_step=airplanes[render_step],
                            wake_history_rows=wake_history_rows,
                        )
                    ]
                for body_index, wake_surface in enumerate(wake_surfaces):
                    if wake_surface.n_points > 0:
                        wake_edges = wake_surface.extract_all_edges()
                        plotter.add_mesh(
                            wake_edges,
                            smooth_shading=False,
                            color=wake_colors[body_index % len(wake_colors)],
                            opacity=0.42,
                            line_width=1.05,
                        )
            wing_colors = ("#ffb000", "#00d7ff", "#ff5d73", "#b7ff4a")
            for body_index, panel_surface in enumerate(panel_surfaces):
                _add_wing_surface(
                    plotter=plotter,
                    panel_surface=panel_surface,
                    color=wing_colors[body_index % len(wing_colors)],
                )
            if highlight_attachment:
                if mode == "single":
                    body_wing_wakes = _pair_wake_vertices_by_body_wing(
                        snapshot=snapshot,
                        airplanes_at_step=(airplanes[render_step],),
                    )
                    body_airplanes = [airplanes[render_step]]
                    body_positions = [position_E_E]
                    body_rotations = [R_pas_E_to_BP]
                else:
                    body_wing_wakes = _pair_wake_vertices_by_body_wing(
                        snapshot=snapshot,
                        airplanes_at_step=airplanes[render_step],
                    )
                    body_airplanes = list(airplanes[render_step])
                    body_positions = list(body_positions_E_E)
                    body_rotations = list(body_R_pas_E_to_BPs)
                for body_index, airplane_at_step in enumerate(body_airplanes):
                    color = wing_colors[body_index % len(wing_colors)]
                    for trailing_edge_line_E in _trailing_edge_lines_E(
                        airplane=airplane_at_step,
                        position_E_E=body_positions[body_index],
                        R_pas_E_to_BP=body_rotations[body_index],
                    ):
                        trailing_edge_poly = _line_polydata_from_points_E(
                            trailing_edge_line_E
                        )
                        if trailing_edge_poly.n_points > 0:
                            plotter.add_mesh(
                                trailing_edge_poly,
                                color="#fff6d5",
                                line_width=8.0,
                                render_lines_as_tubes=True,
                            )
                    for wing_wake_vertices_E in body_wing_wakes[body_index]:
                        wake_front_row_line_E = _wake_front_row_line_E(
                            wing_wake_vertices_E
                        )
                        wake_front_row_poly = _line_polydata_from_points_E(
                            wake_front_row_line_E
                        )
                        if wake_front_row_poly.n_points > 0:
                            plotter.add_mesh(
                                wake_front_row_poly,
                                color=color,
                                line_width=7.0,
                                render_lines_as_tubes=True,
                            )

            focal_point_V = _to_pyvista_axes(camera_target_E_E)
            plotter.camera.position = tuple(
                focal_point_V + camera_distance * camera_direction
            )
            plotter.camera.focal_point = tuple(focal_point_V)
            plotter.camera.up = camera_up
            plotter.camera.parallel_scale = (
                parallel_scale_span * single_case.FULL_SPAN_M
            )
            plotter.reset_camera_clipping_range()
            plotter.add_text(
                f"t = {render_step / steps_per_flap:.2f} s, step {render_step}",
                position="upper_left",
                font_size=12,
                color="#d7d7d7",
            )
            frame = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)
            frame_rgb = frame[:, :, :3]
            if writer is None:
                writer = _FfmpegRawVideoWriter(
                    output_path=output_path,
                    fps=fps,
                    frame_shape=frame_rgb.shape,
                )
            writer.append_data(frame_rgb)
            print(
                f"rendered frame {frame_index + 1}/{len(snapshot_paths)} "
                f"from {snapshot_path.name}"
            )
    finally:
        if writer is not None:
            writer.close()
        pv.close_all()
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SINGLE_RUN_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--mode", choices=("single", "pair"), default="single")
    parser.add_argument(
        "--prescribed-steps", type=int, default=3 * DEFAULT_STEPS_PER_FLAP
    )
    parser.add_argument("--free-steps", type=int, default=20 * DEFAULT_STEPS_PER_FLAP)
    parser.add_argument("--steps-per-flap", type=int, default=DEFAULT_STEPS_PER_FLAP)
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--x-over-span", type=float, default=0.5)
    parser.add_argument("--y-over-span", type=float, default=0.25)
    parser.add_argument("--z-over-span", type=float, default=0.0)
    parser.add_argument("--focal-body-index", type=int, default=1)
    parser.add_argument("--x-wake-behind-span", type=float, default=3.0)
    parser.add_argument("--x-wake-ahead-span", type=float, default=0.75)
    parser.add_argument("--y-half-width-span", type=float, default=1.35)
    parser.add_argument("--z-half-width-span", type=float, default=1.1)
    parser.add_argument("--parallel-scale-span", type=float, default=1.6)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth saved snapshot; useful for quick preview movies.",
    )
    parser.add_argument(
        "--wake-ring-stride",
        type=int,
        default=1,
        help="Render every Nth cropped wake ring to keep the wake from hiding the body.",
    )
    parser.add_argument(
        "--wake-history-rows",
        type=int,
        default=48,
        help="For pair movies, render only the newest N wake rows per body.",
    )
    parser.add_argument(
        "--no-highlight-attachment",
        action="store_true",
        help="Disable the thick trailing-edge/newest-wake-row attachment overlay.",
    )
    parser.add_argument(
        "--wake-style",
        choices=("wire", "surface"),
        default="wire",
        help="Render an age-faded wake wire for clarity, or the older surface-edge style.",
    )
    parser.add_argument(
        "--camera-view",
        choices=("oblique", "top", "side", "rear"),
        default="oblique",
        help="Camera view for the presentation movie.",
    )
    parser.add_argument(
        "--camera-target",
        choices=("focal", "formation"),
        default="focal",
        help="For pair movies, focus the camera on one body or the two-body midpoint.",
    )
    return parser.parse_args()


def main() -> None:
    """Render the requested movie."""
    args = parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = args.run_dir / "AnimateFreeFlight_near_body_wake.mp4"
    movie_path = render_movie(
        run_dir=args.run_dir,
        output_path=output_path,
        mode=args.mode,
        prescribed_steps=args.prescribed_steps,
        free_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        fps=args.fps,
        x_over_span=args.x_over_span,
        y_over_span=args.y_over_span,
        z_over_span=args.z_over_span,
        focal_body_index=args.focal_body_index,
        x_wake_behind_span=args.x_wake_behind_span,
        x_wake_ahead_span=args.x_wake_ahead_span,
        y_half_width_span=args.y_half_width_span,
        z_half_width_span=args.z_half_width_span,
        parallel_scale_span=args.parallel_scale_span,
        frame_stride=args.frame_stride,
        wake_ring_stride=args.wake_ring_stride,
        wake_history_rows=args.wake_history_rows,
        highlight_attachment=not args.no_highlight_attachment,
        wake_style=args.wake_style,
        camera_view=args.camera_view,
        camera_target=args.camera_target,
    )
    print(f"Saved movie to: {movie_path}")


if __name__ == "__main__":
    main()
