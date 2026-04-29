"""Run two flapping-forward free-flight bodies inline for a gap sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from pterasoftware import _transformations

try:
    from examples import free_flight_case_utils as ff_utils
    from examples import free_flight_flapping_forward as flap_case
except ImportError:
    import free_flight_case_utils as ff_utils
    import free_flight_flapping_forward as flap_case


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "phase1_multibody_validation"
    / "two_flapping_forward_inline_gap_sweep"
)
DEFAULT_GAPS_M = (0.5, 1.0, 2.0)
DEFAULT_GAP_AXIS = "x"
DEFAULT_CLAMP_YAW_DEG = 0.0
DEFAULT_STEPS_PER_FLAP = flap_case.DEFAULT_STEPS_PER_FLAP
DEFAULT_PRESCRIBED_STEPS = 4 * DEFAULT_STEPS_PER_FLAP
DEFAULT_FREE_STEPS = 20 * DEFAULT_STEPS_PER_FLAP


def parse_gap_values_m(gaps_text: str) -> tuple[float, ...]:
    """Parse comma-separated positive gap values."""
    if not isinstance(gaps_text, str):
        raise TypeError("gaps_text must be a string.")

    cleaned = [token.strip() for token in gaps_text.split(",") if token.strip()]
    if not cleaned:
        raise ValueError("At least one gap value must be provided.")

    values_m: list[float] = []
    for token in cleaned:
        value_m = float(token)
        if value_m <= 0.0:
            raise ValueError("All gap values must be greater than zero.")
        values_m.append(value_m)
    return tuple(values_m)


def gap_label(gap_m: float) -> str:
    """Build a stable directory label for one gap value."""
    return f"gap_{gap_m:.3f}m".replace(".", "p")


def build_gap_offset_E_m(gap_m: float, gap_axis: str) -> np.ndarray:
    """Build body-2 offset for an inline or lateral initial gap."""
    if gap_axis == "x":
        return np.array([-gap_m, 0.0, 0.0], dtype=float)
    if gap_axis == "y":
        return np.array([0.0, -gap_m, 0.0], dtype=float)
    if gap_axis == "z":
        return np.array([0.0, 0.0, -gap_m], dtype=float)
    raise ValueError('gap_axis must be one of "x", "y", or "z".')


def clamp_multibody_attitude(
    mujoco_model: object,
    target_pitch_deg: float,
    target_yaw_deg: float | None = None,
) -> None:
    """Clamp multibody rigid-body attitude angles after each dynamics step."""
    for body_index, body_id in enumerate(mujoco_model.body_ids):
        R_pas_BP_to_E = mujoco_model.data.xmat[body_id].reshape(3, 3)
        current_angles_deg = ff_utils.extract_izyx_angles_deg(R_pas_BP_to_E.T)
        clamped_angles_deg = current_angles_deg.copy()
        clamped_angles_deg[1] = target_pitch_deg
        if target_yaw_deg is not None:
            clamped_angles_deg[2] = target_yaw_deg

        clamped_T_pas_E_to_BP = _transformations.generate_rot_T(
            angles=clamped_angles_deg,
            passive=True,
            intrinsic=True,
            order="zyx",
        )
        clamped_R_pas_BP_to_E = clamped_T_pas_E_to_BP[:3, :3].T
        clamped_quat_act_E_to_BP_wxyz = _transformations.R_to_quat_wxyz(
            clamped_R_pas_BP_to_E
        )

        qpos_adr = int(mujoco_model.body_qposadrs[body_index])
        qvel_adr = int(mujoco_model.body_qveladrs[body_index])
        mujoco_model.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
            clamped_quat_act_E_to_BP_wxyz
        )
        mujoco_model.data.qvel[qvel_adr + 4] = 0.0
        if target_yaw_deg is not None:
            mujoco_model.data.qvel[qvel_adr + 5] = 0.0

    mujoco.mj_forward(mujoco_model.model, mujoco_model.data)


def install_multibody_fixed_attitude_rotation(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
    target_pitch_deg: float,
    target_yaw_deg: float | None = None,
) -> None:
    """Patch MuJoCo stepping so selected attitude angles are clamped each step."""
    mujoco_model = coupled_problem.mujoco_model
    clamp_multibody_attitude(
        mujoco_model=mujoco_model,
        target_pitch_deg=target_pitch_deg,
        target_yaw_deg=target_yaw_deg,
    )
    original_step = mujoco_model.step

    def step_with_fixed_attitude() -> None:
        original_step()
        clamp_multibody_attitude(
            mujoco_model=mujoco_model,
            target_pitch_deg=target_pitch_deg,
            target_yaw_deg=target_yaw_deg,
        )

    mujoco_model.step = step_with_fixed_attitude


def multibody_rigid_body_drag_model(
    body_index: int,
    coupled_operating_point: ps.operating_point.CoupledOperatingPoint,
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tuned body drag surrogate for one multibody flapping aircraft."""
    del body_index, airplane

    speed_mps = coupled_operating_point.vCg__E
    rho = coupled_operating_point.rho

    linear_drag_n = flap_case.BODY_DRAG_LINEAR_N_PER_MPS * speed_mps
    quadratic_drag_n = 0.5 * rho * flap_case.BODY_DRAG_CD_AREA_M2 * speed_mps**2
    total_drag_n = linear_drag_n + quadratic_drag_n

    return (
        np.array([-total_drag_n, 0.0, 0.0], dtype=float),
        np.zeros(3, dtype=float),
    )


def build_problem(
    inline_gap_m: float,
    gap_axis: str,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    clamp_yaw_deg: float | None,
) -> tuple[
    ps.problems.MultiBodyCoupledUnsteadyProblem,
    ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Build a two-flapping-body multibody free-flight problem."""
    delta_time = flap_case.FLAPPING_PERIOD_S / steps_per_flap

    airplane_1 = flap_case.build_airplane()
    airplane_2 = flap_case.build_airplane()
    airplane_movement_1 = flap_case.build_airplane_movement(airplane_1)
    airplane_movement_2 = flap_case.build_airplane_movement(airplane_2)

    operating_point_kwargs = dict(
        rho=flap_case.AIR_DENSITY,
        vCg__E=flap_case.INITIAL_SPEED_MPS,
        alpha=flap_case.INITIAL_ALPHA_DEG,
        beta=0.0,
        angles_E_to_BP1_izyx=(
            0.0,
            flap_case.INITIAL_ALPHA_DEG,
            flap_case.INITIAL_YAW_DEG,
        ),
        externalFX_W=0.0,
        nu=flap_case.KINEMATIC_VISCOSITY,
        g_E=flap_case.GRAVITY_E,
    )
    coupled_operating_point_1 = ps.operating_point.CoupledOperatingPoint(
        **operating_point_kwargs
    )
    coupled_operating_point_2 = ps.operating_point.CoupledOperatingPoint(
        **operating_point_kwargs
    )

    coupled_movement = ps.movements.movement.MultiBodyCoupledMovement(
        airplane_movements=[airplane_movement_1, airplane_movement_2],
        initial_coupled_operating_points=[
            coupled_operating_point_1,
            coupled_operating_point_2,
        ],
        initial_positions_E_E=[
            np.array([0.0, 0.0, 0.0], dtype=float),
            build_gap_offset_E_m(gap_m=inline_gap_m, gap_axis=gap_axis),
        ],
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    coupled_problem = ps.problems.MultiBodyCoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1s=[flap_case.INERTIA_BP1_CGP1, flap_case.INERTIA_BP1_CGP1],
        external_forces_fn=multibody_rigid_body_drag_model,
    )
    coupled_solver = ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_problem
    )
    install_multibody_fixed_attitude_rotation(
        coupled_problem=coupled_problem,
        target_pitch_deg=flap_case.FIXED_PITCH_DEG,
        target_yaw_deg=clamp_yaw_deg,
    )
    return coupled_problem, coupled_solver


def get_multibody_history_arrays(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract time, position, velocity, alpha, and Euler-angle histories."""
    coupled_operating_points_by_step = (
        coupled_problem.coupled_movement.coupled_operating_points
    )
    positions_by_step = coupled_problem.coupled_movement.positions_E_E
    num_steps = len(coupled_operating_points_by_step)
    num_bodies = len(coupled_operating_points_by_step[0])

    times_s = np.arange(num_steps, dtype=float) * coupled_problem.delta_time
    positions_E_E = np.zeros((num_steps, num_bodies, 3), dtype=float)
    velocities_E__E = np.zeros((num_steps, num_bodies, 3), dtype=float)
    alphas_deg = np.zeros((num_steps, num_bodies), dtype=float)
    euler_angles_deg = np.zeros((num_steps, num_bodies, 3), dtype=float)

    for step_index, (coupled_operating_points, positions_step) in enumerate(
        zip(coupled_operating_points_by_step, positions_by_step, strict=True)
    ):
        for body_index, (coupled_operating_point, position_E_E) in enumerate(
            zip(coupled_operating_points, positions_step, strict=True)
        ):
            positions_E_E[step_index, body_index] = position_E_E
            velocities_E__E[step_index, body_index] = coupled_operating_point.vCg_E__E
            alphas_deg[step_index, body_index] = coupled_operating_point.alpha
            euler_angles_deg[step_index, body_index] = (
                coupled_operating_point.angles_E_to_BP1_izyx
            )

    return times_s, positions_E_E, velocities_E__E, alphas_deg, euler_angles_deg


def save_gap_plots(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
) -> dict[str, str]:
    """Save per-body histories and inter-body separation history plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = times_s / flap_case.FLAPPING_PERIOD_S

    plot_paths: dict[str, str] = {}
    for body_index in range(velocities_E__E.shape[1]):
        body_velocities_E__E = velocities_E__E[:, body_index, :]
        body_speeds_mps = np.linalg.norm(body_velocities_E__E, axis=1)
        body_plot_path = ff_utils.save_velocity_history_plot(
            x_values=x_values,
            velocities_E__E=body_velocities_E__E,
            speeds_mps=body_speeds_mps,
            alphas_deg=alphas_deg[:, body_index],
            euler_angles_deg=euler_angles_deg[:, body_index, :],
            save_path=output_dir / f"body_{body_index}_velocity_history.png",
            x_label="Time / Flapping Period",
            title=f"Two Flapping Bodies Inline (Body {body_index})",
        )
        plot_paths[f"body_{body_index}_velocity_history"] = str(body_plot_path)

    interbody_offset_E_E = positions_E_E[:, 1, :] - positions_E_E[:, 0, :]
    interbody_offset_mag_m = np.linalg.norm(interbody_offset_E_E, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x_values, interbody_offset_E_E[:, 0], label="dx")
    axes[0].plot(x_values, interbody_offset_E_E[:, 1], label="dy")
    axes[0].plot(x_values, interbody_offset_E_E[:, 2], label="dz")
    axes[0].set_ylabel("Offset (m)")
    axes[0].set_title("Inter-Body Offset Components")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, interbody_offset_mag_m, color="black")
    axes[1].set_ylabel("|Offset| (m)")
    axes[1].set_xlabel("Time / Flapping Period")
    axes[1].set_title("Inter-Body Separation Magnitude")
    axes[1].grid(True)

    fig.tight_layout()
    offset_plot_path = output_dir / "interbody_offset_history.png"
    fig.savefig(offset_plot_path, dpi=150)
    plt.close(fig)
    plot_paths["interbody_offset_history"] = str(offset_plot_path)
    return plot_paths


def save_interbody_trajectory_movie(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    frame_stride: int | None = None,
) -> Path:
    """Save a simple two-view trajectory movie for the two-body evolution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(times_s)
    if num_steps < 2:
        raise ValueError("Need at least 2 time points to render a trajectory movie.")

    if frame_stride is None:
        frame_stride = max(1, num_steps // 600)
    frame_indices = list(range(0, num_steps, frame_stride))
    if frame_indices[-1] != num_steps - 1:
        frame_indices.append(num_steps - 1)

    x_values = positions_E_E[:, :, 0]
    y_values = positions_E_E[:, :, 1]
    z_values = positions_E_E[:, :, 2]

    def _limits(values: np.ndarray) -> tuple[float, float]:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        pad = max(0.05, 0.1 * max(1e-9, v_max - v_min))
        return v_min - pad, v_max + pad

    x_min, x_max = _limits(x_values)
    y_min, y_max = _limits(y_values)
    z_min, z_max = _limits(z_values)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_xy, ax_xz = axes
    ax_xy.set_title("Top View (X-Y)")
    ax_xz.set_title("Side View (X-Z)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xz.set_xlim(x_min, x_max)
    ax_xz.set_ylim(z_min, z_max)
    ax_xy.grid(True)
    ax_xz.grid(True)

    colors = ("tab:blue", "tab:orange")
    trail_xy = []
    trail_xz = []
    points_xy = []
    points_xz = []
    for body_index, color in enumerate(colors):
        (trail_xy_line,) = ax_xy.plot([], [], "-", color=color, linewidth=1.5)
        (trail_xz_line,) = ax_xz.plot([], [], "-", color=color, linewidth=1.5)
        (point_xy,) = ax_xy.plot([], [], "o", color=color, markersize=6)
        (point_xz,) = ax_xz.plot([], [], "o", color=color, markersize=6)
        trail_xy.append(trail_xy_line)
        trail_xz.append(trail_xz_line)
        points_xy.append(point_xy)
        points_xz.append(point_xz)

    fig.tight_layout()
    fps = max(
        8,
        min(
            30,
            int(round(1.0 / ((times_s[1] - times_s[0]) * frame_stride))),
        ),
    )
    frames: list[np.ndarray] = []
    for step in frame_indices:
        for body_index in range(2):
            trail_xy[body_index].set_data(
                x_values[: step + 1, body_index], y_values[: step + 1, body_index]
            )
            trail_xz[body_index].set_data(
                x_values[: step + 1, body_index], z_values[: step + 1, body_index]
            )
            points_xy[body_index].set_data(
                [x_values[step, body_index]], [y_values[step, body_index]]
            )
            points_xz[body_index].set_data(
                [x_values[step, body_index]], [z_values[step, body_index]]
            )

        current_period = times_s[step] / flap_case.FLAPPING_PERIOD_S
        fig.suptitle(f"Two Flapping Bodies Inline: t/T = {current_period:.2f}")
        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame_rgba[:, :, :3].copy())

    plt.close(fig)
    movie_path = output_dir / "trajectory_render.mp4"
    try:
        ff_utils._write_mp4_with_ffmpeg(
            rgb_frames=frames,
            fps=fps,
            save_path=movie_path,
        )
    except Exception:
        imageio.mimsave(movie_path, frames, fps=fps)
    return movie_path


def _get_multibody_wake_ring_vortex_surfaces(
    coupled_solver: ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
    step: int,
) -> pv.PolyData:
    """Build wake-ring surfaces for one multibody step in standard coordinates."""
    num_wake_ring_vortices = coupled_solver.list_num_wake_vortices[step]
    stack_fr = coupled_solver.listStackFrwrvp_GP1_CgP1[step]
    stack_fl = coupled_solver.listStackFlwrvp_GP1_CgP1[step]
    stack_bl = coupled_solver.listStackBlwrvp_GP1_CgP1[step]
    stack_br = coupled_solver.listStackBrwrvp_GP1_CgP1[step]

    if stack_fr is None or stack_fl is None or stack_bl is None or stack_br is None:
        return pv.PolyData(np.zeros((0, 3), dtype=float), np.zeros(0, dtype=int))

    wake_vertices = np.zeros((0, 3), dtype=float)
    wake_faces = np.zeros(0, dtype=int)
    for wake_index in range(num_wake_ring_vortices):
        fr = stack_fr[wake_index]
        fl = stack_fl[wake_index]
        bl = stack_bl[wake_index]
        br = stack_br[wake_index]
        wake_vertices_to_add = np.vstack((fl, fr, br, bl))
        wake_face_to_add = np.array(
            [
                4,
                wake_index * 4,
                wake_index * 4 + 1,
                wake_index * 4 + 2,
                wake_index * 4 + 3,
            ],
            dtype=int,
        )
        wake_vertices = np.vstack((wake_vertices, wake_vertices_to_add))
        wake_faces = np.hstack((wake_faces, wake_face_to_add))

    return pv.PolyData(wake_vertices, wake_faces)


def save_standard_wake_movie(
    coupled_solver: ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
    output_dir: Path,
    filename_stem: str,
    follow_body_index: int = 0,
    show_wake_vortices: bool = True,
) -> Path:
    """Render a standard aerodynamic movie with wake surfaces for a multibody run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    step_problems = coupled_solver.multi_body_coupled_steady_problems
    num_steps = len(step_problems)
    if num_steps < 2:
        raise ValueError("Need at least 2 steps to render a standard wake movie.")
    if follow_body_index < 0 or follow_body_index >= coupled_solver.num_bodies:
        raise ValueError(
            "follow_body_index must be within [0, num_bodies - 1] for this case."
        )

    first_problem = step_problems[0]
    if first_problem is None:
        raise RuntimeError("First multibody step history is unavailable for rendering.")
    first_panel_surfaces = ps.output._get_panel_surfaces(first_problem.airplanes)
    bounds = np.array(first_panel_surfaces.bounds, dtype=float)
    panel_diagonal = float(np.linalg.norm(bounds[1::2] - bounds[::2]))
    follow_distance = max(4.0 * panel_diagonal, 2.0)
    follow_parallel_scale = max(1.8 * panel_diagonal, 0.8)
    camera_direction = np.array([-1.0, -1.0, 0.9], dtype=float)
    camera_direction /= np.linalg.norm(camera_direction)

    requested_fps = 1.0 / coupled_solver.delta_time
    fps = max(8, min(30, int(round(requested_fps))))

    original_plotter = pv.Plotter

    def offscreen_plotter(*args: Any, **kwargs: Any) -> pv.Plotter:
        kwargs.setdefault("off_screen", True)
        return original_plotter(*args, **kwargs)

    frame_width = 1280
    frame_height = 720
    mp4_path = output_dir / f"{filename_stem}.mp4"

    ffmpeg_process = subprocess.Popen(
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
            f"{frame_width}x{frame_height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert ffmpeg_process.stdin is not None
    assert ffmpeg_process.stderr is not None

    pv.Plotter = offscreen_plotter
    plotter = pv.Plotter(window_size=(frame_width, frame_height), lighting=None)
    plotter.enable_parallel_projection()  # type: ignore[call-arg]
    plotter.set_background(color="black")  # type: ignore[call-arg]
    try:
        for step_index in range(num_steps):
            this_problem = step_problems[step_index]
            if this_problem is None:
                break

            plotter.clear()
            panel_surfaces = ps.output._get_panel_surfaces(this_problem.airplanes)

            if show_wake_vortices:
                wake_surfaces = _get_multibody_wake_ring_vortex_surfaces(
                    coupled_solver, step_index
                )
                if wake_surfaces.n_points > 0:
                    plotter.add_mesh(
                        wake_surfaces,
                        show_edges=True,
                        smooth_shading=False,
                        color="white",
                    )

            plotter.add_mesh(
                panel_surfaces,
                show_edges=True,
                smooth_shading=False,
                color="chartreuse",
            )

            this_position = coupled_solver.stackPositions_E_E[
                step_index, follow_body_index
            ]
            focal_point = tuple(this_position.tolist())
            camera_position = tuple(
                (this_position + follow_distance * camera_direction).tolist()
            )
            plotter.camera.position = camera_position
            plotter.camera.focal_point = focal_point
            plotter.camera.up = (0.0, 0.0, 1.0)
            plotter.camera.parallel_scale = follow_parallel_scale
            plotter.reset_camera_clipping_range()

            current_period = (
                step_index * coupled_solver.delta_time / flap_case.FLAPPING_PERIOD_S
            )
            plotter.add_text(
                f"Two Flapping Bodies | t/T = {current_period:.2f}",
                position="upper_left",
                font_size=10,
                color="#cfcfcf",
            )

            frame_rgb = np.asarray(
                plotter.screenshot(
                    filename=None,
                    transparent_background=False,
                    return_img=True,
                )
            )[:, :, :3]
            ffmpeg_process.stdin.write(np.asarray(frame_rgb, dtype=np.uint8).tobytes())
    finally:
        pv.Plotter = original_plotter
        pv.close_all()
        if ffmpeg_process.stdin is not None and not ffmpeg_process.stdin.closed:
            ffmpeg_process.stdin.close()

    stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
    return_code = ffmpeg_process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg MP4 conversion failed:\n{stderr}")

    return mp4_path


def summarize_gap_run(
    inline_gap_m: float,
    gap_axis: str,
    clamp_yaw_deg: float | None,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
    run_status: str,
    error_message: str | None,
) -> dict[str, Any]:
    """Build a machine-readable summary for one inline-gap simulation."""
    interbody_velocity_delta_mps = np.max(
        np.linalg.norm(velocities_E__E[:, 0, :] - velocities_E__E[:, 1, :], axis=1)
    )
    interbody_alpha_delta_deg = float(
        np.max(np.abs(alphas_deg[:, 0] - alphas_deg[:, 1]))
    )
    interbody_euler_delta_deg = float(
        np.max(np.abs(euler_angles_deg[:, 0, :] - euler_angles_deg[:, 1, :]))
    )
    offset_history_E_E = positions_E_E[:, 1, :] - positions_E_E[:, 0, :]
    reference_offset_E_E = offset_history_E_E[0]
    interbody_offset_drift_m = float(
        np.max(
            np.linalg.norm(offset_history_E_E - reference_offset_E_E[None, :], axis=1)
        )
    )

    block_size = min(len(velocities_E__E), steps_per_flap)
    last_block_mean_velocity_body_0 = velocities_E__E[-block_size:, 0, :].mean(axis=0)
    last_block_mean_velocity_body_1 = velocities_E__E[-block_size:, 1, :].mean(axis=0)

    return {
        "case": "phase1_multibody_two_flapping_forward_inline_gap_sweep",
        "inline_gap_m": inline_gap_m,
        "gap_axis": gap_axis,
        "body_2_initial_offset_E_m": build_gap_offset_E_m(
            gap_m=inline_gap_m, gap_axis=gap_axis
        ).tolist(),
        "flapping_frequency_hz": flap_case.FLAPPING_FREQUENCY_HZ,
        "flapping_period_s": flap_case.FLAPPING_PERIOD_S,
        "steps_per_flap": steps_per_flap,
        "fixed_pitch_deg": flap_case.FIXED_PITCH_DEG,
        "fixed_yaw_deg": clamp_yaw_deg,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "periods_total_requested": (prescribed_num_steps + free_num_steps)
        / steps_per_flap,
        "periods_total_completed": float(times_s[-1] / flap_case.FLAPPING_PERIOD_S),
        "time_range_s": [float(times_s[0]), float(times_s[-1])],
        "run_status": run_status,
        "error_message": error_message,
        "body_0_final_position_E_m": positions_E_E[-1, 0, :].tolist(),
        "body_1_final_position_E_m": positions_E_E[-1, 1, :].tolist(),
        "body_0_final_velocity_E_mps": velocities_E__E[-1, 0, :].tolist(),
        "body_1_final_velocity_E_mps": velocities_E__E[-1, 1, :].tolist(),
        "body_0_last_flap_mean_velocity_E_mps": last_block_mean_velocity_body_0.tolist(),
        "body_1_last_flap_mean_velocity_E_mps": last_block_mean_velocity_body_1.tolist(),
        "max_interbody_velocity_delta_mps": float(interbody_velocity_delta_mps),
        "max_interbody_alpha_delta_deg": interbody_alpha_delta_deg,
        "max_interbody_euler_delta_deg": interbody_euler_delta_deg,
        "max_interbody_offset_drift_m": interbody_offset_drift_m,
        "passes_body_symmetry_check": bool(
            interbody_velocity_delta_mps < 1e-3
            and interbody_alpha_delta_deg < 1e-3
            and interbody_euler_delta_deg < 1e-3
            and interbody_offset_drift_m < 1e-5
        ),
    }


def run_gap_case(
    output_dir: Path,
    inline_gap_m: float,
    gap_axis: str,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    show_progress: bool,
    prescribed_wake: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    history_save_dir: Path | None,
    standard_render_wake: bool,
    standard_render_follow_body_index: int,
    clamp_yaw_deg: float | None,
) -> dict[str, Any]:
    """Run one inline-gap two-flapping-body case and save outputs."""
    coupled_problem, coupled_solver = build_problem(
        inline_gap_m=inline_gap_m,
        gap_axis=gap_axis,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        clamp_yaw_deg=clamp_yaw_deg,
    )
    run_status = "ok"
    error_message: str | None = None
    try:
        coupled_solver.run(
            prescribed_wake=prescribed_wake,
            show_progress=show_progress,
            history_stride=history_stride,
            save_every_n_steps=save_every_n_steps,
            history_save_dir=history_save_dir,
        )
    except Exception as exc:
        run_status = "failed"
        error_message = repr(exc)
        print(
            f"Run failed for gap={inline_gap_m:.3f} m at partial history; "
            f"saving partial diagnostics. Error: {error_message}"
        )

    times_s, positions_E_E, velocities_E__E, alphas_deg, euler_angles_deg = (
        get_multibody_history_arrays(coupled_problem)
    )
    summary = summarize_gap_run(
        inline_gap_m=inline_gap_m,
        gap_axis=gap_axis,
        clamp_yaw_deg=clamp_yaw_deg,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        run_status=run_status,
        error_message=error_message,
    )
    summary["diagnostic_plots"] = save_gap_plots(
        output_dir=output_dir,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
    )
    if len(times_s) >= 2:
        try:
            movie_path = save_interbody_trajectory_movie(
                output_dir=output_dir,
                times_s=times_s,
                positions_E_E=positions_E_E,
            )
            summary["trajectory_movie_mp4"] = str(movie_path)
        except Exception as exc:
            summary["trajectory_movie_error"] = repr(exc)
            print(
                "Trajectory render export failed; diagnostics plots and summary were "
                f"still saved. Error: {exc!r}"
            )
    if len(times_s) >= 2 and standard_render_wake:
        try:
            standard_movie_path = save_standard_wake_movie(
                coupled_solver=coupled_solver,
                output_dir=output_dir,
                filename_stem="AnimateFreeFlight_multibody_standard_wake",
                follow_body_index=standard_render_follow_body_index,
                show_wake_vortices=True,
            )
            summary["standard_wake_movie_mp4"] = str(standard_movie_path)
        except Exception as exc:
            summary["standard_wake_movie_error"] = repr(exc)
            print(
                "Standard wake render export failed; diagnostics plots and summary were "
                f"still saved. Error: {exc!r}"
            )
    ff_utils.write_json(output_dir / "summary.json", summary)
    return summary


def run_gap_sweep(
    output_root: Path,
    gaps_m: tuple[float, ...],
    gap_axis: str,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    show_progress: bool,
    prescribed_wake: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    history_save_dir: Path | None,
    standard_render_wake: bool,
    standard_render_follow_body_index: int,
    clamp_yaw_deg: float | None,
) -> dict[str, Any]:
    """Run the requested initial-gap sweep and save per-gap summaries."""
    output_root.mkdir(parents=True, exist_ok=True)

    gap_summaries: list[dict[str, Any]] = []
    for gap_m in gaps_m:
        this_output_dir = output_root / gap_label(gap_m)
        this_history_save_dir: Path | None = None
        if history_save_dir is not None:
            this_history_save_dir = history_save_dir / gap_label(gap_m)
        summary = run_gap_case(
            output_dir=this_output_dir,
            inline_gap_m=gap_m,
            gap_axis=gap_axis,
            prescribed_num_steps=prescribed_num_steps,
            free_num_steps=free_num_steps,
            steps_per_flap=steps_per_flap,
            show_progress=show_progress,
            prescribed_wake=prescribed_wake,
            history_stride=history_stride,
            save_every_n_steps=save_every_n_steps,
            history_save_dir=this_history_save_dir,
            standard_render_wake=standard_render_wake,
            standard_render_follow_body_index=standard_render_follow_body_index,
            clamp_yaw_deg=clamp_yaw_deg,
        )
        gap_summaries.append(summary)
        print(
            f"Saved summary to: {this_output_dir / 'summary.json'} "
            f"(status={summary['run_status']})"
        )

    sweep_summary = {
        "case": "phase1_multibody_two_flapping_forward_inline_gap_sweep",
        "gaps_m": list(gaps_m),
        "gap_axis": gap_axis,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "steps_per_flap": steps_per_flap,
        "fixed_pitch_deg": flap_case.FIXED_PITCH_DEG,
        "fixed_yaw_deg": clamp_yaw_deg,
        "flapping_period_s": flap_case.FLAPPING_PERIOD_S,
        "gap_results": gap_summaries,
    }
    ff_utils.write_json(output_root / "gap_sweep_summary.json", sweep_summary)
    print(f"Saved sweep summary to: {output_root / 'gap_sweep_summary.json'}")
    return sweep_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run two flapping-forward multibody cases inline for a set of initial gaps."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where per-gap outputs should be written.",
    )
    parser.add_argument(
        "--gaps-m",
        type=str,
        default=",".join(str(gap_m) for gap_m in DEFAULT_GAPS_M),
        help="Comma-separated initial inline gap values in meters.",
    )
    parser.add_argument(
        "--gap-axis",
        choices=("x", "y", "z"),
        default=DEFAULT_GAP_AXIS,
        help=(
            "Earth-axis direction for body 2 offset from body 1. "
            "Use x for true streamwise inline spacing in the current forward case."
        ),
    )
    parser.add_argument(
        "--prescribed-steps",
        type=int,
        default=DEFAULT_PRESCRIBED_STEPS,
        help="Number of prescribed wake-building steps.",
    )
    parser.add_argument(
        "--free-steps",
        type=int,
        default=DEFAULT_FREE_STEPS,
        help="Number of free-flight steps after the prescribed phase.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Temporal resolution in steps per flapping period.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the progress bar while solving each gap case.",
    )
    parser.add_argument(
        "--prescribed-wake",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a prescribed wake instead of the default free wake.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help=(
            "Retain full in-memory history every N steps. Use 1 to keep all history "
            "(default). Values >1 reduce memory usage."
        ),
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=None,
        help=(
            "If set, write compressed per-step snapshots to disk every N steps to "
            "reduce reliance on RAM history."
        ),
    )
    parser.add_argument(
        "--history-save-dir",
        type=Path,
        default=None,
        help="Directory where streamed history snapshots should be written.",
    )
    parser.add_argument(
        "--standard-render-wake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a standard wake-visible aerodynamic movie for each gap case.",
    )
    parser.add_argument(
        "--standard-render-follow-body-index",
        type=int,
        default=0,
        help="Body index to follow in the standard wake render camera.",
    )
    parser.add_argument(
        "--clamp-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp yaw after each MuJoCo step for long-run numerical stability.",
    )
    parser.add_argument(
        "--target-yaw-deg",
        type=float,
        default=DEFAULT_CLAMP_YAW_DEG,
        help="Target yaw angle in degrees when yaw clamping is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the inline-gap sweep."""
    args = parse_args()
    clamp_yaw_deg = args.target_yaw_deg if args.clamp_yaw else None
    run_gap_sweep(
        output_root=args.output_root,
        gaps_m=parse_gap_values_m(args.gaps_m),
        gap_axis=args.gap_axis,
        prescribed_num_steps=args.prescribed_steps,
        free_num_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        show_progress=args.show_progress,
        prescribed_wake=args.prescribed_wake,
        history_stride=args.history_stride,
        save_every_n_steps=args.save_every_n_steps,
        history_save_dir=args.history_save_dir,
        standard_render_wake=args.standard_render_wake,
        standard_render_follow_body_index=args.standard_render_follow_body_index,
        clamp_yaw_deg=clamp_yaw_deg,
    )


if __name__ == "__main__":
    main()
