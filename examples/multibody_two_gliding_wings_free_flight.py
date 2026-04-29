"""Validate two far-separated gliders in true multibody free flight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from pterasoftware import _transformations

try:
    from examples import free_flight_case_utils as ff_utils
    from examples import free_flight_gliding_wing as glider_case
except ImportError:
    import free_flight_case_utils as ff_utils
    import free_flight_gliding_wing as glider_case


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "phase1_multibody_validation"
    / "two_gliding_wings_free_flight_far_apart"
)
DEFAULT_LATERAL_SEPARATION_M = 5.0
DEFAULT_STEPS_PER_REFERENCE_PERIOD = 12
DEFAULT_PRESCRIBED_STEPS = 2 * DEFAULT_STEPS_PER_REFERENCE_PERIOD
DEFAULT_FREE_STEPS = 4 * DEFAULT_STEPS_PER_REFERENCE_PERIOD


def build_separation_vector_E_m(
    separation_distance_m: float,
    separation_axis: str,
) -> np.ndarray:
    """Build a one-axis separation vector from distance and axis label."""
    if separation_axis not in {"x", "y", "z"}:
        raise ValueError("separation_axis must be one of: x, y, z.")
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    separation_vector_E_m = np.zeros(3, dtype=float)
    separation_vector_E_m[axis_to_index[separation_axis]] = separation_distance_m
    return separation_vector_E_m


def build_multibody_problem(
    separation_vector_E_m: np.ndarray,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_reference_period: int,
    fix_pitch_rotation: bool,
    fixed_pitch_deg: float,
) -> tuple[
    ps.problems.MultiBodyCoupledUnsteadyProblem,
    ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Build a true multibody free-flight problem with two far-separated gliders."""
    delta_time = glider_case.REFERENCE_PERIOD_S / steps_per_reference_period

    airplane_1 = glider_case.build_airplane()
    airplane_2 = glider_case.build_airplane()
    airplane_movement_1 = glider_case.build_airplane_movement(airplane_1)
    airplane_movement_2 = glider_case.build_airplane_movement(airplane_2)

    operating_point_kwargs = dict(
        rho=glider_case.AIR_DENSITY,
        vCg__E=glider_case.DEFAULT_INITIAL_SPEED_MPS,
        alpha=glider_case.DEFAULT_INITIAL_ALPHA_DEG,
        beta=0.0,
        angles_E_to_BP1_izyx=(
            0.0,
            glider_case.DEFAULT_INITIAL_ALPHA_DEG,
            glider_case.INITIAL_YAW_DEG,
        ),
        externalFX_W=0.0,
        nu=glider_case.KINEMATIC_VISCOSITY,
        g_E=glider_case.GRAVITY_E,
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
            separation_vector_E_m.astype(float, copy=True),
        ],
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    coupled_problem = ps.problems.MultiBodyCoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1s=[glider_case.INERTIA_BP1_CGP1, glider_case.INERTIA_BP1_CGP1],
    )
    coupled_solver = ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_problem
    )
    if fix_pitch_rotation:
        install_multibody_fixed_pitch_rotation(
            coupled_problem=coupled_problem,
            target_pitch_deg=fixed_pitch_deg,
        )
    return coupled_problem, coupled_solver


def clamp_multibody_pitch(mujoco_model: object, target_pitch_deg: float) -> None:
    """Clamp every multibody rigid-body pitch angle after each dynamics step."""
    for body_index, body_id in enumerate(mujoco_model.body_ids):
        R_pas_BP_to_E = mujoco_model.data.xmat[body_id].reshape(3, 3)
        current_angles_deg = ff_utils.extract_izyx_angles_deg(R_pas_BP_to_E.T)
        clamped_angles_deg = current_angles_deg.copy()
        clamped_angles_deg[1] = target_pitch_deg

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

    mujoco.mj_forward(mujoco_model.model, mujoco_model.data)


def install_multibody_fixed_pitch_rotation(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
    target_pitch_deg: float,
) -> None:
    """Patch multibody MuJoCo stepping so pitch is clamped after each step."""
    mujoco_model = coupled_problem.mujoco_model
    clamp_multibody_pitch(mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg)
    original_step = mujoco_model.step

    def step_with_fixed_pitch() -> None:
        original_step()
        clamp_multibody_pitch(
            mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg
        )

    mujoco_model.step = step_with_fixed_pitch


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


def summarize_free_flight_validation(
    single_problem: ps.problems.CoupledUnsteadyProblem,
    single_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    pair_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
    pair_solver: ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
    separation_vector_E_m: np.ndarray,
    steps_per_reference_period: int,
) -> dict[str, Any]:
    """Compare the far-separated two-body histories against the single-body case."""
    single_times_s, single_velocities_E__E, _, single_alphas_deg, single_eulers_deg = (
        ff_utils.get_history_arrays(single_problem)
    )
    (
        pair_times_s,
        pair_positions_E_E,
        pair_velocities_E__E,
        pair_alphas_deg,
        pair_eulers_deg,
    ) = get_multibody_history_arrays(pair_problem)

    num_single_steps = len(single_solver.stackPosition_E_E)
    num_pair_steps = len(pair_solver.stackPositions_E_E)
    num_comparison_steps = min(
        num_single_steps,
        num_pair_steps,
        len(single_times_s),
        len(pair_times_s),
    )

    single_times_s = single_times_s[:num_comparison_steps]
    single_velocities_E__E = single_velocities_E__E[:num_comparison_steps]
    single_alphas_deg = single_alphas_deg[:num_comparison_steps]
    single_eulers_deg = single_eulers_deg[:num_comparison_steps]
    pair_times_s = pair_times_s[:num_comparison_steps]
    pair_positions_E_E = pair_positions_E_E[:num_comparison_steps]
    pair_velocities_E__E = pair_velocities_E__E[:num_comparison_steps]
    pair_alphas_deg = pair_alphas_deg[:num_comparison_steps]
    pair_eulers_deg = pair_eulers_deg[:num_comparison_steps]

    single_positions_E_E = single_solver.stackPosition_E_E[:num_comparison_steps]
    reference_position_offsets_E_E = single_positions_E_E - single_positions_E_E[0:1, :]

    pair_body_summaries: list[dict[str, Any]] = []
    max_velocity_error_mps = 0.0
    max_alpha_error_deg = 0.0
    max_euler_error_deg = 0.0
    max_position_error_m = 0.0

    for body_index in range(pair_velocities_E__E.shape[1]):
        body_position_offsets_E_E = (
            pair_positions_E_E[:, body_index, :]
            - pair_positions_E_E[0:1, body_index, :]
        )
        velocity_error = np.linalg.norm(
            pair_velocities_E__E[:, body_index, :] - single_velocities_E__E, axis=1
        )
        alpha_error = np.abs(pair_alphas_deg[:, body_index] - single_alphas_deg)
        euler_error = np.max(
            np.abs(pair_eulers_deg[:, body_index, :] - single_eulers_deg), axis=1
        )
        position_error = np.linalg.norm(
            body_position_offsets_E_E - reference_position_offsets_E_E, axis=1
        )

        max_velocity_error_mps = max(
            max_velocity_error_mps, float(np.max(velocity_error))
        )
        max_alpha_error_deg = max(max_alpha_error_deg, float(np.max(alpha_error)))
        max_euler_error_deg = max(max_euler_error_deg, float(np.max(euler_error)))
        max_position_error_m = max(max_position_error_m, float(np.max(position_error)))

        pair_body_summaries.append(
            {
                "body_index": body_index,
                "initial_position_E_m": pair_positions_E_E[0, body_index].tolist(),
                "final_position_E_m": pair_positions_E_E[-1, body_index].tolist(),
                "final_velocity_E_mps": pair_velocities_E__E[-1, body_index].tolist(),
                "max_velocity_error_vs_single_mps": float(np.max(velocity_error)),
                "max_alpha_error_vs_single_deg": float(np.max(alpha_error)),
                "max_euler_component_error_vs_single_deg": float(np.max(euler_error)),
                "max_offset_position_error_vs_single_m": float(np.max(position_error)),
            }
        )

    interbody_velocity_delta_mps = np.max(
        np.linalg.norm(
            pair_velocities_E__E[:, 0, :] - pair_velocities_E__E[:, 1, :], axis=1
        )
    )
    interbody_alpha_delta_deg = float(
        np.max(np.abs(pair_alphas_deg[:, 0] - pair_alphas_deg[:, 1]))
    )
    interbody_euler_delta_deg = float(
        np.max(np.abs(pair_eulers_deg[:, 0, :] - pair_eulers_deg[:, 1, :]))
    )
    pair_offset_history_E_E = pair_positions_E_E[:, 1, :] - pair_positions_E_E[:, 0, :]
    reference_offset_E_E = pair_offset_history_E_E[0]
    interbody_offset_drift_m = float(
        np.max(
            np.linalg.norm(
                pair_offset_history_E_E - reference_offset_E_E[None, :], axis=1
            )
        )
    )

    return {
        "case": "phase1_multibody_free_wake_two_gliding_wings",
        "description": (
            "Two identical gliding wings are advanced together with a true free wake "
            "in the new multibody solver. The summary reports both inter-body symmetry "
            "and the current gap versus the legacy single-body solver."
        ),
        "body_2_initial_offset_E_m": separation_vector_E_m.tolist(),
        "body_2_initial_offset_norm_m": float(np.linalg.norm(separation_vector_E_m)),
        "steps_per_reference_period": steps_per_reference_period,
        "prescribed_steps": pair_problem.coupled_movement.prescribed_num_steps,
        "free_steps": pair_problem.coupled_movement.free_num_steps,
        "single_case_final_position_E_m": single_solver.stackPosition_E_E[-1].tolist(),
        "single_case_final_velocity_E_mps": single_velocities_E__E[-1].tolist(),
        "single_case_times_s": [float(single_times_s[0]), float(single_times_s[-1])],
        "pair_case_times_s": [float(pair_times_s[0]), float(pair_times_s[-1])],
        "pair_bodies": pair_body_summaries,
        "max_velocity_error_vs_single_mps": max_velocity_error_mps,
        "max_alpha_error_vs_single_deg": max_alpha_error_deg,
        "max_euler_component_error_vs_single_deg": max_euler_error_deg,
        "max_offset_position_error_vs_single_m": max_position_error_m,
        "passes_single_case_equivalence_check": bool(
            max_velocity_error_mps < 5e-4
            and max_alpha_error_deg < 5e-4
            and max_euler_error_deg < 5e-4
            and max_position_error_m < 5e-4
        ),
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


def save_pair_diagnostic_plots(
    output_dir: Path,
    pair_times_s: np.ndarray,
    pair_positions_E_E: np.ndarray,
    pair_velocities_E__E: np.ndarray,
    pair_alphas_deg: np.ndarray,
    pair_eulers_deg: np.ndarray,
) -> dict[str, str]:
    """Save per-body history plots plus inter-body separation history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = pair_times_s / glider_case.REFERENCE_PERIOD_S

    plot_paths: dict[str, str] = {}
    for body_index in range(pair_velocities_E__E.shape[1]):
        body_velocities_E__E = pair_velocities_E__E[:, body_index, :]
        body_speeds_mps = np.linalg.norm(body_velocities_E__E, axis=1)
        body_alphas_deg = pair_alphas_deg[:, body_index]
        body_eulers_deg = pair_eulers_deg[:, body_index, :]
        body_plot_path = ff_utils.save_velocity_history_plot(
            x_values=x_values,
            velocities_E__E=body_velocities_E__E,
            speeds_mps=body_speeds_mps,
            alphas_deg=body_alphas_deg,
            euler_angles_deg=body_eulers_deg,
            save_path=output_dir / f"body_{body_index}_velocity_history.png",
            x_label="Time / Reference Period",
            title=f"Multibody Gliding Wing History (Body {body_index})",
        )
        plot_paths[f"body_{body_index}_velocity_history"] = str(body_plot_path)

    interbody_offset_E_E = pair_positions_E_E[:, 1, :] - pair_positions_E_E[:, 0, :]
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
    axes[1].set_xlabel("Time / Reference Period")
    axes[1].set_title("Inter-Body Separation Magnitude")
    axes[1].grid(True)

    fig.tight_layout()
    offset_plot_path = output_dir / "interbody_offset_history.png"
    fig.savefig(offset_plot_path, dpi=150)
    plt.close(fig)
    plot_paths["interbody_offset_history"] = str(offset_plot_path)
    return plot_paths


def run_validation(
    output_dir: Path,
    lateral_separation_m: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_reference_period: int,
    separation_axis: str = "y",
    pitch_mode: str = "free",
    history_stride: int = 1,
    save_every_n_steps: int | None = None,
    history_save_dir: Path | None = None,
) -> dict[str, Any]:
    """Run single- and two-body free-flight solves and save a comparison summary."""
    fix_pitch_rotation = pitch_mode == "fixed"
    fixed_pitch_deg = glider_case.FIXED_PITCH_DEG
    separation_vector_E_m = build_separation_vector_E_m(
        separation_distance_m=lateral_separation_m,
        separation_axis=separation_axis,
    )

    single_problem, single_solver = glider_case.build_problem(
        initial_speed_mps=glider_case.DEFAULT_INITIAL_SPEED_MPS,
        initial_alpha_deg=glider_case.DEFAULT_INITIAL_ALPHA_DEG,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_reference_period=steps_per_reference_period,
        fix_pitch_rotation=fix_pitch_rotation,
        fixed_pitch_deg=fixed_pitch_deg,
    )
    pair_problem, pair_solver = build_multibody_problem(
        separation_vector_E_m=separation_vector_E_m,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_reference_period=steps_per_reference_period,
        fix_pitch_rotation=fix_pitch_rotation,
        fixed_pitch_deg=fixed_pitch_deg,
    )

    single_history_dir = (
        None if history_save_dir is None else history_save_dir / "single_history"
    )
    pair_history_dir = (
        None if history_save_dir is None else history_save_dir / "pair_history"
    )
    single_solver.run(
        prescribed_wake=False,
        show_progress=False,
        history_stride=history_stride,
        save_every_n_steps=save_every_n_steps,
        history_save_dir=single_history_dir,
    )
    pair_solver.run(
        prescribed_wake=False,
        show_progress=False,
        history_stride=history_stride,
        save_every_n_steps=save_every_n_steps,
        history_save_dir=pair_history_dir,
    )

    summary = summarize_free_flight_validation(
        single_problem=single_problem,
        single_solver=single_solver,
        pair_problem=pair_problem,
        pair_solver=pair_solver,
        separation_vector_E_m=separation_vector_E_m,
        steps_per_reference_period=steps_per_reference_period,
    )
    (
        pair_times_s,
        pair_positions_E_E,
        pair_velocities_E__E,
        pair_alphas_deg,
        pair_eulers_deg,
    ) = get_multibody_history_arrays(pair_problem)
    summary["diagnostic_plots"] = save_pair_diagnostic_plots(
        output_dir=output_dir,
        pair_times_s=pair_times_s,
        pair_positions_E_E=pair_positions_E_E,
        pair_velocities_E__E=pair_velocities_E__E,
        pair_alphas_deg=pair_alphas_deg,
        pair_eulers_deg=pair_eulers_deg,
    )
    ff_utils.write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that two far-separated gliders in multibody free flight "
            "reproduce the matching single-glider case."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the summary JSON should be saved.",
    )
    parser.add_argument(
        "--lateral-separation-m",
        type=float,
        default=DEFAULT_LATERAL_SEPARATION_M,
        help="Initial body-to-body separation magnitude in meters.",
    )
    parser.add_argument(
        "--separation-axis",
        choices=("x", "y", "z"),
        default="y",
        help=(
            "Earth-axis direction for body 2 offset from body 1. "
            "Use y for the existing far-apart validation baseline."
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
        help="Number of coupled free-flight steps after the prescribed phase.",
    )
    parser.add_argument(
        "--steps-per-reference-period",
        type=int,
        default=DEFAULT_STEPS_PER_REFERENCE_PERIOD,
        help="Temporal resolution relative to the reference gliding period.",
    )
    parser.add_argument(
        "--pitch-mode",
        choices=("free", "fixed"),
        default="free",
        help="Use free pitch dynamics or clamp pitch for both bodies.",
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
    return parser.parse_args()


def main() -> None:
    """Run the validation script."""
    args = parse_args()
    summary = run_validation(
        output_dir=args.output_dir,
        lateral_separation_m=args.lateral_separation_m,
        prescribed_num_steps=args.prescribed_steps,
        free_num_steps=args.free_steps,
        steps_per_reference_period=args.steps_per_reference_period,
        separation_axis=args.separation_axis,
        pitch_mode=args.pitch_mode,
        history_stride=args.history_stride,
        save_every_n_steps=args.save_every_n_steps,
        history_save_dir=args.history_save_dir,
    )
    print(f"Saved summary to: {args.output_dir / 'summary.json'}")
    print(
        "Far-field free-wake symmetry:",
        summary["passes_body_symmetry_check"],
    )


if __name__ == "__main__":
    main()
