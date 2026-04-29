"""Shared helpers for branch-local free-flight example scripts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyvista as pv
from PIL import Image, ImageSequence

import pterasoftware as ps
from pterasoftware import _transformations


def extract_izyx_angles_deg(R_pas_E_to_BP1: np.ndarray) -> np.ndarray:
    """Extract intrinsic z-y'-x'' Euler angles from a passive rotation matrix."""
    sin_angle_y = np.clip(-R_pas_E_to_BP1[0, 2], -1.0, 1.0)
    angle_y = np.rad2deg(np.arcsin(sin_angle_y))

    if np.abs(sin_angle_y) > 0.99999:
        angle_x = 0.0
        angle_z = np.rad2deg(np.arctan2(-R_pas_E_to_BP1[1, 0], R_pas_E_to_BP1[1, 1]))
    else:
        angle_x = np.rad2deg(np.arctan2(R_pas_E_to_BP1[1, 2], R_pas_E_to_BP1[2, 2]))
        angle_z = np.rad2deg(np.arctan2(R_pas_E_to_BP1[0, 1], R_pas_E_to_BP1[0, 0]))

    return np.array([angle_x, angle_y, angle_z], dtype=float)


def clamp_mujoco_pitch(mujoco_model: object, target_pitch_deg: float) -> None:
    """Clamp the MuJoCo rigid-body state back to a fixed pitch angle."""
    R_pas_BP1_to_E = mujoco_model.data.xmat[mujoco_model.body_id].reshape(3, 3)
    current_angles_deg = extract_izyx_angles_deg(R_pas_BP1_to_E.T)
    clamped_angles_deg = current_angles_deg.copy()
    clamped_angles_deg[1] = target_pitch_deg

    clamped_T_pas_E_to_BP1 = _transformations.generate_rot_T(
        angles=clamped_angles_deg,
        passive=True,
        intrinsic=True,
        order="zyx",
    )
    clamped_R_pas_BP1_to_E = clamped_T_pas_E_to_BP1[:3, :3].T
    clamped_quat_act_E_to_BP1_wxyz = _transformations.R_to_quat_wxyz(
        clamped_R_pas_BP1_to_E
    )

    mujoco_model.data.qpos[3:7] = clamped_quat_act_E_to_BP1_wxyz
    mujoco_model.data.qvel[4] = 0.0
    mujoco.mj_forward(mujoco_model.model, mujoco_model.data)


def install_fixed_pitch_rotation(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    target_pitch_deg: float,
) -> None:
    """Patch the example's MuJoCo step so pitch is clamped after each update."""
    mujoco_model = coupled_problem.mujoco_model
    clamp_mujoco_pitch(mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg)
    original_step = mujoco_model.step

    def step_with_fixed_pitch() -> None:
        original_step()
        clamp_mujoco_pitch(mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg)

    mujoco_model.step = step_with_fixed_pitch


def get_history_arrays(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract time, velocity, speed, alpha, and Euler-angle histories."""
    operating_points = coupled_problem.coupled_movement.coupled_operating_points
    num_steps = len(operating_points)
    times_s = np.arange(num_steps) * coupled_problem.delta_time
    velocities_E__E = np.array(
        [operating_point.vCg_E__E for operating_point in operating_points],
        dtype=float,
    )
    speeds_mps = np.array(
        [operating_point.vCg__E for operating_point in operating_points],
        dtype=float,
    )
    alphas_deg = np.array(
        [operating_point.alpha for operating_point in operating_points],
        dtype=float,
    )
    euler_angles_deg = np.array(
        [operating_point.angles_E_to_BP1_izyx for operating_point in operating_points],
        dtype=float,
    )
    return times_s, velocities_E__E, speeds_mps, alphas_deg, euler_angles_deg


def save_velocity_history_plot(
    x_values: np.ndarray,
    velocities_E__E: np.ndarray,
    speeds_mps: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
    save_path: Path,
    x_label: str,
    title: str,
) -> Path:
    """Save the velocity and attitude histories as a PNG."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(x_values, velocities_E__E[:, 0], label="Vx")
    axes[0].plot(x_values, velocities_E__E[:, 1], label="Vy")
    axes[0].plot(x_values, velocities_E__E[:, 2], label="Vz")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("CG Velocity Components in Earth Axes")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, speeds_mps, color="black")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("CG Speed Magnitude")
    axes[1].grid(True)

    axes[2].plot(x_values, alphas_deg, color="tab:green")
    axes[2].set_ylabel("Alpha (deg)")
    axes[2].set_title("Angle of Attack")
    axes[2].grid(True)

    axes[3].plot(x_values, euler_angles_deg[:, 0], label="Roll")
    axes[3].plot(x_values, euler_angles_deg[:, 1], label="Pitch")
    axes[3].plot(x_values, euler_angles_deg[:, 2], label="Yaw")
    axes[3].set_ylabel("Euler (deg)")
    axes[3].set_xlabel(x_label)
    axes[3].set_title("Euler Angles")
    axes[3].grid(True)
    axes[3].legend()

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def compute_energy_histories(
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    weight_n: float,
    gravity_E: np.ndarray | tuple[float, float, float],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return mass, kinetic, gravitational potential, and total energy histories."""
    gravity_E = np.asarray(gravity_E, dtype=float)
    gravity_magnitude = float(np.linalg.norm(gravity_E))
    if gravity_magnitude <= 0.0:
        raise ValueError("gravity_E must have non-zero magnitude.")

    mass_kg = float(weight_n / gravity_magnitude)
    kinetic_energy_j = 0.5 * mass_kg * np.sum(velocities_E__E**2, axis=1)
    gravity_direction_E = gravity_E / gravity_magnitude
    potential_energy_j = -weight_n * (positions_E_E @ gravity_direction_E)
    total_energy_j = kinetic_energy_j + potential_energy_j
    return mass_kg, kinetic_energy_j, potential_energy_j, total_energy_j


def save_energy_history_plot(
    x_values: np.ndarray,
    kinetic_energy_j: np.ndarray,
    potential_energy_j: np.ndarray,
    total_energy_j: np.ndarray,
    save_path: Path,
    x_label: str,
    title: str,
) -> Path:
    """Save the energy histories as a PNG."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x_values, kinetic_energy_j, label="Kinetic")
    ax.plot(x_values, potential_energy_j, label="Potential")
    ax.plot(x_values, total_energy_j, label="Total", color="black", linewidth=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (J)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    """Return the mean of the final block of samples."""
    return values[-block_size:].mean(axis=0)


def add_block_summaries(
    summary: dict[str, Any],
    velocities_E__E: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
    steps_per_block: int,
    block_name: str,
    block_counts: tuple[int, ...] = (1, 2, 5, 10),
) -> None:
    """Append late-time block summaries to an output dictionary."""
    for block_count in block_counts:
        block_size = block_count * steps_per_block
        if len(velocities_E__E) < block_size:
            continue

        if block_count == 1:
            velocity_key = f"last_{block_name}_mean_velocity_E_mps"
            alpha_key = f"last_{block_name}_mean_alpha_deg"
        else:
            velocity_key = f"last_{block_count}_{block_name}_mean_velocity_E_mps"
            alpha_key = f"last_{block_count}_{block_name}_mean_alpha_deg"

        summary[velocity_key] = _block_mean(velocities_E__E, block_size).tolist()
        summary[alpha_key] = float(_block_mean(alphas_deg, block_size))

    summary["pitch_span_deg"] = float(np.ptp(euler_angles_deg[:, 1]))


def write_json(save_path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file with stable formatting."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))
    return save_path


def _write_mp4_with_ffmpeg(
    rgb_frames: list[np.ndarray],
    fps: float,
    save_path: Path,
) -> None:
    """Encode RGB frames to an MP4 using the system ffmpeg binary."""
    height, width = rgb_frames[0].shape[:2]
    process = subprocess.Popen(
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
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(save_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    assert process.stdin is not None
    for rgb_frame in rgb_frames:
        process.stdin.write(np.asarray(rgb_frame, dtype=np.uint8).tobytes())
    process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg MP4 conversion failed:\n{stderr}")


def save_animation_bundle(
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    output_dir: Path,
    filename_stem: str,
    scalar_type: str = "lift",
    show_wake_vortices: bool = True,
) -> tuple[Path, Path]:
    """Render a body-following free-flight animation and save WebP and MP4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    original_plotter = pv.Plotter

    def offscreen_plotter(*args: Any, **kwargs: Any) -> pv.Plotter:
        kwargs.setdefault("off_screen", True)
        return original_plotter(*args, **kwargs)

    os.chdir(output_dir)
    pv.Plotter = offscreen_plotter
    try:
        ps.output.animate_free_flight(
            coupled_solver=coupled_solver,
            scalar_type=scalar_type,
            show_wake_vortices=show_wake_vortices,
            save=True,
            testing=True,
            camera_mode="follow_body",
        )
    finally:
        pv.Plotter = original_plotter
        os.chdir(original_cwd)

    source_webp_path = output_dir / "AnimateFreeFlight.webp"
    if not source_webp_path.exists():
        raise FileNotFoundError(source_webp_path)

    webp_path = output_dir / f"{filename_stem}.webp"
    if webp_path.exists():
        webp_path.unlink()
    source_webp_path.rename(webp_path)

    image = Image.open(webp_path)
    rgb_frames: list[np.ndarray] = []
    durations_ms: list[int] = []
    for frame in ImageSequence.Iterator(image):
        rgb_frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        durations_ms.append(frame.info.get("duration", image.info.get("duration", 40)))

    fps = 1000.0 / (sum(durations_ms) / len(durations_ms)) if durations_ms else 25.0
    mp4_path = output_dir / f"{filename_stem}.mp4"
    try:
        with imageio.get_writer(
            mp4_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        ) as writer:
            for rgb_frame in rgb_frames:
                writer.append_data(rgb_frame)
    except Exception:
        _write_mp4_with_ffmpeg(rgb_frames=rgb_frames, fps=fps, save_path=mp4_path)

    return webp_path, mp4_path
