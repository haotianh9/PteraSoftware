"""Run a single-wing fixed-trim steady-time check with free wake."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    from examples import free_flight_gliding_wing as glide_case
    from examples import multibody_streamwise_stability_energy_sweep as sweep_case
except ImportError:
    import free_flight_case_utils as ff_utils
    import free_flight_gliding_wing as glide_case
    import multibody_streamwise_stability_energy_sweep as sweep_case


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "single_wing_streamwise_steady_check"
)

DEFAULT_STREAMWISE_SPEED_MPS = 1.0
DEFAULT_ANGLE_OF_ATTACK_DEG = 5.0
DEFAULT_REFERENCE_TIME_S = sweep_case.REFERENCE_TIME_S
DEFAULT_STEPS_PER_REFERENCE_TIME = 24
DEFAULT_TIME_STEP_S = DEFAULT_REFERENCE_TIME_S / DEFAULT_STEPS_PER_REFERENCE_TIME


def _quat_from_izyx_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Return a MuJoCo-compatible quaternion for intrinsic z-y-x Euler angles."""
    clamped_T_pas_E_to_BP1 = _transformations.generate_rot_T(
        angles=angles_deg,
        passive=True,
        intrinsic=True,
        order="zyx",
    )
    clamped_R_pas_BP1_to_E = clamped_T_pas_E_to_BP1[:3, :3].T
    return _transformations.R_to_quat_wxyz(clamped_R_pas_BP1_to_E)


def _stack_history(history: list[np.ndarray], width: int) -> np.ndarray:
    """Stack a per-step history list into a dense array."""
    if not history:
        return np.zeros((0, width), dtype=float)
    return np.stack(history, axis=0)


@dataclass
class SingleBodyFixedTrimClampDiagnostics:
    """Clamp one rigid body to a translating fixed-trim path."""

    mujoco_model: object
    target_position_E_m: np.ndarray
    target_angles_deg: np.ndarray
    prescribed_streamwise_speed_mps: float = DEFAULT_STREAMWISE_SPEED_MPS
    delta_time_s: float = DEFAULT_TIME_STEP_S
    max_abs_speed_mps: float = 50.0

    def __post_init__(self) -> None:
        self.target_position_E_m = np.asarray(self.target_position_E_m, dtype=float)
        self.target_angles_deg = np.asarray(self.target_angles_deg, dtype=float)
        if self.target_position_E_m.shape != (3,):
            raise ValueError("target_position_E_m must have shape (3,).")
        if self.target_angles_deg.shape != (3,):
            raise ValueError("target_angles_deg must have shape (3,).")
        if self.prescribed_streamwise_speed_mps <= 0.0:
            raise ValueError("prescribed_streamwise_speed_mps must be positive.")
        if self.delta_time_s <= 0.0:
            raise ValueError("delta_time_s must be positive.")
        if self.max_abs_speed_mps <= 0.0:
            raise ValueError("max_abs_speed_mps must be positive.")

        self.initial_target_position_E_m = self.target_position_E_m.copy()
        self.state_step_index = 0
        self.raw_forces_E: list[np.ndarray] = []
        self.raw_moments_E_Cg: list[np.ndarray] = []
        self.projected_forces_E: list[np.ndarray] = []
        self.projected_moments_E_Cg: list[np.ndarray] = []
        self.preclamp_velocities_E: list[np.ndarray] = []
        self.preclamp_angular_rates_rad_s: list[np.ndarray] = []

    def install(self) -> None:
        """Patch MuJoCo hooks to enforce fixed-trim state each step."""
        self.enforce_state(record=False)
        original_apply_loads = self.mujoco_model.apply_loads
        original_step = self.mujoco_model.step

        def apply_fixed_trim_loads(
            forces_E: np.ndarray,
            moments_E_Cg: np.ndarray,
        ) -> None:
            raw_forces_E = np.asarray(forces_E, dtype=float).copy()
            raw_moments_E_Cg = np.asarray(moments_E_Cg, dtype=float).copy()
            projected_forces_E = np.zeros_like(raw_forces_E)
            projected_moments_E_Cg = np.zeros_like(raw_moments_E_Cg)

            self.raw_forces_E.append(raw_forces_E)
            self.raw_moments_E_Cg.append(raw_moments_E_Cg)
            self.projected_forces_E.append(projected_forces_E.copy())
            self.projected_moments_E_Cg.append(projected_moments_E_Cg.copy())
            original_apply_loads(projected_forces_E, projected_moments_E_Cg)

        def step_fixed_trim() -> None:
            original_step()
            self.state_step_index += 1
            self.enforce_state(record=True)

        self.mujoco_model.apply_loads = apply_fixed_trim_loads
        self.mujoco_model.step = step_fixed_trim

    def _current_target_position_E_m(self) -> np.ndarray:
        """Return the translating target position for the current state step."""
        position = self.initial_target_position_E_m.copy()
        position[0] += (
            self.prescribed_streamwise_speed_mps
            * self.delta_time_s
            * self.state_step_index
        )
        return position

    def enforce_state(self, record: bool) -> None:
        """Clamp all rigid-body state components and record pre-clamp rates."""
        velocity_E = np.copy(self.mujoco_model.data.qvel[0:3])
        angular_rate_rad_s = np.copy(self.mujoco_model.data.qvel[3:6])

        target_quat_wxyz = _quat_from_izyx_angles_deg(self.target_angles_deg)
        self.mujoco_model.data.qpos[0:3] = self._current_target_position_E_m()
        self.mujoco_model.data.qpos[3:7] = target_quat_wxyz
        self.mujoco_model.data.qvel[0] = self.prescribed_streamwise_speed_mps
        self.mujoco_model.data.qvel[1:6] = 0.0

        mujoco.mj_forward(self.mujoco_model.model, self.mujoco_model.data)
        if record:
            self.preclamp_velocities_E.append(velocity_E)
            self.preclamp_angular_rates_rad_s.append(angular_rate_rad_s)
            self.raise_if_diverged()

    def raise_if_diverged(self) -> None:
        """Stop a case early if the fixed-trim state becomes non-finite."""
        speed_abs_mps = float(abs(self.mujoco_model.data.qvel[0]))
        if not np.isfinite(speed_abs_mps):
            raise RuntimeError(
                "Single-wing fixed-trim case diverged with non-finite Ux."
            )
        if speed_abs_mps > self.max_abs_speed_mps:
            raise RuntimeError(
                "Single-wing fixed-trim case exceeded speed guard: "
                f"|Ux|={speed_abs_mps:.3g} m/s."
            )

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return recorded clamp/load histories as arrays."""
        raw_forces_E = _stack_history(self.raw_forces_E, 3)
        raw_moments_E_Cg = _stack_history(self.raw_moments_E_Cg, 3)
        projected_forces_E = _stack_history(self.projected_forces_E, 3)
        projected_moments_E_Cg = _stack_history(self.projected_moments_E_Cg, 3)
        preclamp_velocities_E = _stack_history(self.preclamp_velocities_E, 3)
        preclamp_angular_rates_rad_s = _stack_history(
            self.preclamp_angular_rates_rad_s, 3
        )

        num_steps = min(
            len(raw_forces_E),
            len(raw_moments_E_Cg),
            len(preclamp_velocities_E),
            len(preclamp_angular_rates_rad_s),
        )
        raw_forces_E = raw_forces_E[:num_steps]
        raw_moments_E_Cg = raw_moments_E_Cg[:num_steps]
        projected_forces_E = projected_forces_E[:num_steps]
        projected_moments_E_Cg = projected_moments_E_Cg[:num_steps]
        preclamp_velocities_E = preclamp_velocities_E[:num_steps]
        preclamp_angular_rates_rad_s = preclamp_angular_rates_rad_s[:num_steps]

        clamp_forces_E = -raw_forces_E
        clamp_moments_E_Cg = -raw_moments_E_Cg
        clamp_power_proxy_W = np.sum(
            np.abs(clamp_forces_E * preclamp_velocities_E), axis=1
        ) + np.sum(
            np.abs(clamp_moments_E_Cg * preclamp_angular_rates_rad_s),
            axis=1,
        )

        return {
            "raw_forces_E_N": raw_forces_E,
            "raw_moments_E_Cg_Nm": raw_moments_E_Cg,
            "projected_forces_E_N": projected_forces_E,
            "projected_moments_E_Cg_Nm": projected_moments_E_Cg,
            "preclamp_velocities_E_mps": preclamp_velocities_E,
            "preclamp_angular_rates_rad_s": preclamp_angular_rates_rad_s,
            "clamp_forces_E_N": clamp_forces_E,
            "clamp_moments_E_Cg_Nm": clamp_moments_E_Cg,
            "clamp_power_proxy_W": clamp_power_proxy_W,
        }


def build_problem(
    angle_of_attack_deg: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    time_step_s: float,
    prescribed_streamwise_speed_mps: float,
    max_abs_speed_mps: float,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    SingleBodyFixedTrimClampDiagnostics,
]:
    """Build one fixed-trim single-wing problem."""
    airplane = sweep_case.build_rectangular_fixed_wing_airplane(
        name="Single Fixed Wing"
    )
    airplane_movement = glide_case.build_airplane_movement(airplane)
    coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
        rho=glide_case.AIR_DENSITY,
        vCg__E=prescribed_streamwise_speed_mps,
        alpha=angle_of_attack_deg,
        beta=0.0,
        angles_E_to_BP1_izyx=(0.0, angle_of_attack_deg, 0.0),
        externalFX_W=0.0,
        nu=glide_case.KINEMATIC_VISCOSITY,
        g_E=glide_case.GRAVITY_E,
    )
    coupled_movement = ps.movements.movement.CoupledMovement(
        airplane_movement=airplane_movement,
        initial_coupled_operating_point=coupled_operating_point,
        delta_time=time_step_s,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    coupled_problem = ps.problems.CoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1=glide_case.INERTIA_BP1_CGP1,
    )
    coupled_solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_unsteady_problem=coupled_problem
    )
    clamp_diagnostics = SingleBodyFixedTrimClampDiagnostics(
        mujoco_model=coupled_problem.mujoco_model,
        target_position_E_m=np.array([0.0, 0.0, 0.0], dtype=float),
        target_angles_deg=np.array([0.0, angle_of_attack_deg, 0.0], dtype=float),
        prescribed_streamwise_speed_mps=prescribed_streamwise_speed_mps,
        delta_time_s=time_step_s,
        max_abs_speed_mps=max_abs_speed_mps,
    )
    clamp_diagnostics.install()
    return coupled_problem, coupled_solver, clamp_diagnostics


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing moving average."""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def estimate_steady_time_s(
    times_s: np.ndarray,
    thrust_required_N: np.ndarray,
    window_s: float = 1.0,
    rel_tol: float = 0.02,
    abs_tol_N: float = 1.0e-8,
    reference_tail_s: float = 2.0,
    min_time_s: float = 0.0,
) -> tuple[float | None, dict[str, float]]:
    """Estimate the first time after which thrust stays close to tail steady value."""
    if len(times_s) < 5:
        return None, {}

    delta_time_s = float(np.median(np.diff(times_s)))
    if delta_time_s <= 0.0:
        return None, {}

    window_steps = max(3, int(round(window_s / delta_time_s)))
    tail_steps = max(window_steps, int(round(reference_tail_s / delta_time_s)))
    if len(thrust_required_N) < tail_steps + 2:
        return None, {}

    tail_mean = float(np.mean(thrust_required_N[-tail_steps:]))
    ma = moving_average(thrust_required_N, window_steps)
    ma_times = times_s[window_steps - 1 :]
    tol = max(abs_tol_N, rel_tol * max(1.0e-8, abs(tail_mean)))
    within_tol = np.abs(ma - tail_mean) <= tol

    steady_time_s: float | None = None
    for idx in range(len(within_tol)):
        if float(ma_times[idx]) < min_time_s:
            continue
        if bool(np.all(within_tol[idx:])):
            steady_time_s = float(ma_times[idx])
            break

    diagnostics = {
        "delta_time_s": delta_time_s,
        "window_s": float(window_steps * delta_time_s),
        "reference_tail_s": float(tail_steps * delta_time_s),
        "tail_mean_required_thrust_N": tail_mean,
        "tolerance_N": float(tol),
    }
    return steady_time_s, diagnostics


def save_thrust_history_plot(
    times_s: np.ndarray,
    thrust_required_N: np.ndarray,
    steady_time_s: float | None,
    steady_diagnostics: dict[str, float],
    save_path: Path,
) -> None:
    """Save a thrust-convergence plot."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(times_s) > 3:
        delta_time_s = float(np.median(np.diff(times_s)))
        window_s = steady_diagnostics.get("window_s", 1.0)
        window_steps = max(3, int(round(window_s / delta_time_s)))
        ma = moving_average(thrust_required_N, window_steps)
        ma_times = times_s[window_steps - 1 :]
    else:
        ma = thrust_required_N.copy()
        ma_times = times_s.copy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        times_s, thrust_required_N, color="tab:blue", alpha=0.45, label="Instantaneous"
    )
    axes[0].plot(ma_times, ma, color="black", linewidth=2.0, label="Moving mean")

    tail_mean = steady_diagnostics.get("tail_mean_required_thrust_N")
    tol = steady_diagnostics.get("tolerance_N")
    if tail_mean is not None:
        axes[0].axhline(
            float(tail_mean), color="tab:green", linestyle="--", label="Tail mean"
        )
    if tail_mean is not None and tol is not None:
        axes[0].axhspan(
            float(tail_mean) - float(tol),
            float(tail_mean) + float(tol),
            color="tab:green",
            alpha=0.15,
            label="Steady tolerance band",
        )
    if steady_time_s is not None:
        axes[0].axvline(
            steady_time_s,
            color="tab:red",
            linestyle="--",
            label="Estimated steady time",
        )

    axes[0].set_ylabel("Required Thrust (N)")
    axes[0].set_title("Single-Wing Streamwise Clamp Thrust Convergence")
    axes[0].grid(True)
    axes[0].legend()

    if len(ma_times) > 1:
        d_ma_dt = np.gradient(ma, ma_times)
        axes[1].plot(ma_times, d_ma_dt, color="tab:purple")
    axes[1].set_ylabel("d(Mean Thrust)/dt (N/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Moving-Mean Slope")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-wing fixed-trim free-wake check and estimate the "
            "physical time needed to reach thrust steady state."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--angle-of-attack-deg", type=float, default=DEFAULT_ANGLE_OF_ATTACK_DEG
    )
    parser.add_argument(
        "--prescribed-streamwise-speed-mps",
        type=float,
        default=DEFAULT_STREAMWISE_SPEED_MPS,
    )
    parser.add_argument("--total-time-s", type=float, default=20.0)
    parser.add_argument("--prescribed-time-s", type=float, default=1.0)
    parser.add_argument("--time-step-s", type=float, default=DEFAULT_TIME_STEP_S)
    parser.add_argument("--history-stride", type=int, default=24)
    parser.add_argument("--save-every-n-steps", type=int, default=24)
    parser.add_argument(
        "--show-progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-abs-speed-mps", type=float, default=50.0)
    parser.add_argument("--steady-window-s", type=float, default=1.0)
    parser.add_argument("--steady-tail-s", type=float, default=2.0)
    parser.add_argument("--steady-rel-tol", type=float, default=0.02)
    parser.add_argument("--steady-abs-tol-n", type=float, default=1.0e-8)
    parser.add_argument("--steady-min-time-s", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    """Run one single-wing fixed-trim case and summarize steady-time convergence."""
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    prescribed_num_steps = int(round(args.prescribed_time_s / args.time_step_s))
    total_num_steps = int(round(args.total_time_s / args.time_step_s))
    free_num_steps = max(1, total_num_steps - prescribed_num_steps)
    history_save_dir = output_root / "streamed_history"

    coupled_problem, coupled_solver, clamp_diagnostics = build_problem(
        angle_of_attack_deg=float(args.angle_of_attack_deg),
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        time_step_s=float(args.time_step_s),
        prescribed_streamwise_speed_mps=float(args.prescribed_streamwise_speed_mps),
        max_abs_speed_mps=float(args.max_abs_speed_mps),
    )

    run_status = "completed"
    try:
        coupled_solver.run(
            prescribed_wake=False,
            show_progress=bool(args.show_progress),
            history_stride=int(args.history_stride),
            save_every_n_steps=int(args.save_every_n_steps),
            history_save_dir=history_save_dir,
        )
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("Received KeyboardInterrupt, saving partial histories.")

    times_s, velocities_E__E, _, alphas_deg, euler_angles_deg = (
        ff_utils.get_history_arrays(coupled_problem)
    )
    clamp_arrays = clamp_diagnostics.history_arrays()

    n = min(len(times_s), len(velocities_E__E), len(clamp_arrays["clamp_forces_E_N"]))
    times_s = times_s[:n]
    velocities_E__E = velocities_E__E[:n]
    alphas_deg = alphas_deg[:n]
    euler_angles_deg = euler_angles_deg[:n]
    required_thrust_N = clamp_arrays["clamp_forces_E_N"][:n, 0]

    steady_time_s, steady_diagnostics = estimate_steady_time_s(
        times_s=times_s,
        thrust_required_N=required_thrust_N,
        window_s=float(args.steady_window_s),
        rel_tol=float(args.steady_rel_tol),
        abs_tol_N=float(args.steady_abs_tol_n),
        reference_tail_s=float(args.steady_tail_s),
        min_time_s=float(args.steady_min_time_s),
    )

    save_thrust_history_plot(
        times_s=times_s,
        thrust_required_N=required_thrust_N,
        steady_time_s=steady_time_s,
        steady_diagnostics=steady_diagnostics,
        save_path=output_root / "single_wing_required_thrust_history.png",
    )

    np.savez_compressed(
        output_root / "single_wing_history.npz",
        time_s=times_s,
        velocity_E_mps=velocities_E__E,
        alpha_deg=alphas_deg,
        euler_deg=euler_angles_deg,
        required_thrust_N=required_thrust_N,
        clamp_force_E_N=clamp_arrays["clamp_forces_E_N"][:n],
        clamp_moment_E_Cg_Nm=clamp_arrays["clamp_moments_E_Cg_Nm"][:n],
    )

    summary = {
        "case": "single_wing_fixed_trim_steady_check",
        "run_status": run_status,
        "wake_model": "free",
        "prescribed_wake": False,
        "constraint_mode": "translating_fixed_trim_path_and_attitude_clamped_prescribed_streamwise_speed",
        "angle_of_attack_deg": float(args.angle_of_attack_deg),
        "prescribed_streamwise_speed_mps": float(args.prescribed_streamwise_speed_mps),
        "time_step_s": float(args.time_step_s),
        "prescribed_steps": int(prescribed_num_steps),
        "free_steps": int(free_num_steps),
        "time_total_requested_s": float(args.total_time_s),
        "time_total_completed_s": float(times_s[-1] if len(times_s) else 0.0),
        "steady_time_estimate_s": (
            None if steady_time_s is None else float(steady_time_s)
        ),
        "steady_rel_tol": float(args.steady_rel_tol),
        "steady_abs_tol_n": float(args.steady_abs_tol_n),
        "steady_min_time_s": float(args.steady_min_time_s),
        "steady_detection": steady_diagnostics,
        "final_required_thrust_N": (
            float(required_thrust_N[-1]) if len(required_thrust_N) else None
        ),
        "tail_mean_required_thrust_N": (
            None
            if not steady_diagnostics
            else float(steady_diagnostics["tail_mean_required_thrust_N"])
        ),
        "required_thrust_range_N": (
            None if len(required_thrust_N) == 0 else float(np.ptp(required_thrust_N))
        ),
        "final_mean_velocity_E_mps": (
            np.mean(
                velocities_E__E[-max(1, int(round(1.0 / args.time_step_s))) :], axis=0
            ).tolist()
            if len(velocities_E__E)
            else None
        ),
    }
    ff_utils.write_json(output_root / "summary.json", summary)
    print(f"Saved single-wing steady check summary to: {output_root / 'summary.json'}")
    print(
        "Estimated steady time (s):",
        "None" if steady_time_s is None else f"{steady_time_s:.3f}",
    )


if __name__ == "__main__":
    main()
