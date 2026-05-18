"""Run two README-scaled flapping flyers with independent streamwise motion free."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from pterasoftware import _transformations
from pterasoftware.multibody_coupled_unsteady_ring_vortex_lattice_method import (
    MultiBodyRestartState,
)

try:
    from examples import free_flight_readme_scaled_flapping_streamwise as single_case
except ImportError:
    import free_flight_readme_scaled_flapping_streamwise as single_case


if not hasattr(ps.problems, "MultiBodyCoupledUnsteadyProblem"):
    raise RuntimeError("This example requires the multibody free-flight API.")


FULL_SPAN_M = single_case.FULL_SPAN_M
DEFAULT_STEPS_PER_FLAP = single_case.DEFAULT_STEPS_PER_FLAP
DEFAULT_PRESCRIBED_STEPS = 3 * DEFAULT_STEPS_PER_FLAP
DEFAULT_FREE_STEPS = 20 * DEFAULT_STEPS_PER_FLAP
DEFAULT_INITIAL_SPEED_MPS = 0.843615696
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "readme_scaled_flapping_streamwise"
    / "two_flyer_free_x_grid_20p_resolved_wake"
)
GRID_X_OVER_SPAN = (0.5, 1.0, 2.0, 3.0, 5.0)
GRID_Y_OVER_SPAN = (0.25, 0.5, 0.75, 1.0)
GRID_Z_OVER_SPAN = (0.0,)


def format_label(prefix: str, value: float) -> str:
    """Return a filesystem-safe normalized-coordinate label."""
    sign = "m" if value < 0.0 else "p"
    return f"{prefix}_{sign}{abs(value):.3f}".replace(".", "p")


def run_label(x_over_span: float, y_over_span: float, z_over_span: float) -> str:
    """Return the directory label for one two-flyer case."""
    return "_".join(
        (
            format_label("xB", x_over_span),
            format_label("yB", y_over_span),
            format_label("zB", z_over_span),
        )
    )


def load_restart_state(path: Path) -> MultiBodyRestartState:
    """Load a restart-compatible multibody checkpoint."""
    path = Path(path)
    with np.load(path) as data:
        if "restart_compatible" not in data or not bool(data["restart_compatible"][0]):
            raise ValueError(f"{path} is not a restart-compatible checkpoint.")
        return MultiBodyRestartState(
            global_step=int(data["global_step"][0]),
            positions_E_E=np.asarray(data["positions_E_E"], dtype=float),
            R_pas_E_to_BPs=np.asarray(data["R_pas_E_to_BPs"], dtype=float),
            velocities_E__E=np.asarray(data["velocities_E__E"], dtype=float),
            omegas_BPs__E=np.asarray(data["omegas_BPs__E"], dtype=float),
            wake_strengths=np.asarray(data["wake_strengths"], dtype=float),
            wake_ages=np.asarray(data["wake_ages"], dtype=float),
            wake_rc0s=np.asarray(data["wake_rc0s"], dtype=float),
            wake_br=np.asarray(data["wake_br"], dtype=float),
            wake_fr=np.asarray(data["wake_fr"], dtype=float),
            wake_fl=np.asarray(data["wake_fl"], dtype=float),
            wake_bl=np.asarray(data["wake_bl"], dtype=float),
            previous_body_positions_E_E=np.asarray(
                data["previous_body_positions_E_E"], dtype=float
            ),
            previous_body_R_pas_GP_to_Es=np.asarray(
                data["previous_body_R_pas_GP_to_Es"], dtype=float
            ),
            previous_bound_vortex_strengths=np.asarray(
                data["previous_bound_vortex_strengths"], dtype=float
            ),
            previous_stack_cpp=np.asarray(data["previous_stack_cpp"], dtype=float),
            previous_stack_cblvpr=np.asarray(
                data["previous_stack_cblvpr"], dtype=float
            ),
            previous_stack_cblvpf=np.asarray(
                data["previous_stack_cblvpf"], dtype=float
            ),
            previous_stack_cblvpl=np.asarray(
                data["previous_stack_cblvpl"], dtype=float
            ),
            previous_stack_cblvpb=np.asarray(
                data["previous_stack_cblvpb"], dtype=float
            ),
            previous_panel_blpp=np.asarray(data["previous_panel_blpp"], dtype=float),
            previous_panel_brpp=np.asarray(data["previous_panel_brpp"], dtype=float),
        )


def latest_restart_checkpoint(case_dir: Path) -> Path | None:
    """Return the latest restart-compatible checkpoint under a case directory."""
    candidates = sorted(
        Path(case_dir).glob("**/restart_checkpoints/restart_step_*.npz")
    )
    restartable: list[tuple[int, float, Path]] = []
    for candidate in candidates:
        try:
            with np.load(candidate) as data:
                if "restart_compatible" in data and bool(data["restart_compatible"][0]):
                    global_step = int(data["global_step"][0])
                    restartable.append(
                        (global_step, candidate.stat().st_mtime, candidate)
                    )
        except Exception:
            continue
    if not restartable:
        return None
    restartable.sort()
    return restartable[-1][2]


def flapping_phase_offset_deg(global_step: int, steps_per_flap: int) -> float:
    """Return the flapping phase offset needed for a restarted segment."""
    raw_phase_deg = 360.0 * (
        (int(global_step) % int(steps_per_flap)) / float(steps_per_flap)
    )
    wrapped_phase_deg = ((raw_phase_deg + 180.0) % 360.0) - 180.0
    if wrapped_phase_deg <= -180.0:
        return 180.0
    return wrapped_phase_deg


def body_positions_from_offsets(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    span_m: float = FULL_SPAN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return front and rear initial body positions.

    Body 1 is always the front flyer, body 2 is always the rear flyer. The requested
    offsets are DeltaX = x_front - x_rear, DeltaY = y_rear - y_front, and
    DeltaZ = z_rear - z_front. With the rear flyer at the origin, the front flyer is
    therefore [X, -Y, -Z].
    """
    front_position_E_m = span_m * np.array(
        [x_over_span, -y_over_span, -z_over_span],
        dtype=float,
    )
    rear_position_E_m = np.zeros(3, dtype=float)
    return front_position_E_m, rear_position_E_m


def initial_planforms_overlap(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    span_m: float = FULL_SPAN_M,
    chord_m: float = single_case.ROOT_CHORD_M,
    vertical_clearance_m: float = 1.0e-6,
) -> bool:
    """Return whether the two rectangular planform envelopes overlap initially."""
    x_separation_m = abs(x_over_span * span_m)
    y_separation_m = abs(y_over_span * span_m)
    z_separation_m = abs(z_over_span * span_m)
    return (
        x_separation_m < chord_m
        and y_separation_m < span_m
        and z_separation_m < vertical_clearance_m
    )


def initial_condition_is_collision_free(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
) -> bool:
    """Return whether a requested initial condition is safe to run."""
    return not initial_planforms_overlap(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
    )


def _quat_from_izyx_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Return a MuJoCo wxyz quaternion for intrinsic z-y-x Euler angles."""
    clamped_T_pas_E_to_BP = _transformations.generate_rot_T(
        angles=angles_deg,
        passive=True,
        intrinsic=True,
        order="zyx",
    )
    clamped_R_pas_BP_to_E = clamped_T_pas_E_to_BP[:3, :3].T
    return _transformations.R_to_quat_wxyz(clamped_R_pas_BP_to_E)


def _stack_history(
    history: list[np.ndarray], num_bodies: int, width: int
) -> np.ndarray:
    """Stack per-step, per-body history arrays."""
    if not history:
        return np.zeros((0, num_bodies, width), dtype=float)
    return np.stack(history, axis=0)


@dataclass
class MultiBodyStreamwiseFreeClampDiagnostics:
    """Clamp y/z/attitude while preserving independent x and Ux for each body."""

    mujoco_model: object
    target_positions_E_m: np.ndarray
    target_angles_deg: np.ndarray
    initial_streamwise_speed_mps: float | np.ndarray = DEFAULT_INITIAL_SPEED_MPS
    max_abs_speed_mps: float = 50.0
    max_abs_x_over_span: float = 40.0
    span_m: float = FULL_SPAN_M
    forward_fn: Any = mujoco.mj_forward

    def __post_init__(self) -> None:
        self.target_positions_E_m = np.asarray(
            self.target_positions_E_m, dtype=float
        ).copy()
        self.target_angles_deg = np.asarray(self.target_angles_deg, dtype=float).copy()
        if self.target_positions_E_m.shape != (self.mujoco_model.num_bodies, 3):
            raise ValueError("target_positions_E_m has wrong shape.")
        if self.target_angles_deg.shape != (3,):
            raise ValueError("target_angles_deg must have shape (3,).")
        self.initial_streamwise_speed_mps = np.broadcast_to(
            np.asarray(self.initial_streamwise_speed_mps, dtype=float),
            (self.mujoco_model.num_bodies,),
        ).copy()
        self.raw_forces_E: list[np.ndarray] = []
        self.raw_moments_E_Cg: list[np.ndarray] = []
        self.projected_forces_E: list[np.ndarray] = []
        self.projected_moments_E_Cg: list[np.ndarray] = []
        self.preclamp_velocities_E: list[np.ndarray] = []
        self.preclamp_angular_rates_rad_s: list[np.ndarray] = []
        self.dense_diagnostics_csv_path: Path | None = None
        self.dense_diagnostics_every_n_steps: int | None = None
        self.streamed_load_history_dir: Path | None = None
        self.streamed_load_save_every_n_steps: int | None = None
        self.global_step_offset = 0
        self.delta_time_s = 0.0

    def configure_streaming(
        self,
        history_save_dir: Path | None,
        save_every_n_steps: int | None,
        dense_diagnostics_every_n_steps: int | None,
        global_step_offset: int,
        delta_time_s: float,
    ) -> None:
        """Configure lightweight load snapshots and dense live CSV diagnostics."""
        if history_save_dir is None:
            return
        history_save_dir = Path(history_save_dir)
        history_save_dir.mkdir(parents=True, exist_ok=True)
        self.global_step_offset = int(global_step_offset)
        self.delta_time_s = float(delta_time_s)
        if save_every_n_steps is not None:
            self.streamed_load_history_dir = history_save_dir / "clamp_loads"
            self.streamed_load_history_dir.mkdir(parents=True, exist_ok=True)
            self.streamed_load_save_every_n_steps = int(save_every_n_steps)
        if dense_diagnostics_every_n_steps is not None:
            self.dense_diagnostics_every_n_steps = int(dense_diagnostics_every_n_steps)
            self.dense_diagnostics_csv_path = (
                history_save_dir / "dense_speed_force_history.csv"
            )
            self._initialize_dense_csv()

    def _initialize_dense_csv(self) -> None:
        """Create the per-step live-readable dense CSV."""
        if self.dense_diagnostics_csv_path is None:
            return
        fieldnames = [
            "local_step",
            "global_step",
            "time_s",
            "time_over_flap_period",
            "front_x_m",
            "rear_x_m",
            "x_over_span",
            "front_ux_mps",
            "rear_ux_mps",
            "relative_ux_mps",
        ]
        for body_name in ("front", "rear"):
            fieldnames.extend(
                [
                    f"{body_name}_raw_fx_N",
                    f"{body_name}_raw_fy_N",
                    f"{body_name}_raw_fz_N",
                    f"{body_name}_projected_fx_N",
                    f"{body_name}_clamp_fy_N",
                    f"{body_name}_clamp_fz_N",
                    f"{body_name}_raw_mx_Nm",
                    f"{body_name}_raw_my_Nm",
                    f"{body_name}_raw_mz_Nm",
                    f"{body_name}_clamp_mx_Nm",
                    f"{body_name}_clamp_my_Nm",
                    f"{body_name}_clamp_mz_Nm",
                ]
            )
        with self.dense_diagnostics_csv_path.open("w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=fieldnames).writeheader()

    def install(self) -> None:
        """Install load projection and post-step streamwise-free clamping."""
        self.enforce_state(record=False)
        original_apply_loads = self.mujoco_model.apply_loads
        original_step = self.mujoco_model.step

        def apply_streamwise_free_loads(
            forces_E: np.ndarray,
            moments_E_Cg: np.ndarray,
        ) -> None:
            raw_forces_E = np.asarray(forces_E, dtype=float).copy()
            raw_moments_E_Cg = np.asarray(moments_E_Cg, dtype=float).copy()
            projected_forces_E = np.zeros_like(raw_forces_E)
            projected_forces_E[:, 0] = raw_forces_E[:, 0]
            projected_moments_E_Cg = np.zeros_like(raw_moments_E_Cg)

            self.raw_forces_E.append(raw_forces_E)
            self.raw_moments_E_Cg.append(raw_moments_E_Cg)
            self.projected_forces_E.append(projected_forces_E.copy())
            self.projected_moments_E_Cg.append(projected_moments_E_Cg.copy())
            self._save_load_snapshot_if_requested(
                step=len(self.raw_forces_E) - 1,
                raw_forces_E=raw_forces_E,
                raw_moments_E_Cg=raw_moments_E_Cg,
                projected_forces_E=projected_forces_E,
                projected_moments_E_Cg=projected_moments_E_Cg,
            )
            original_apply_loads(projected_forces_E, projected_moments_E_Cg)

        def step_streamwise_free() -> None:
            original_step()
            self.enforce_state(record=True)

        self.mujoco_model.apply_loads = apply_streamwise_free_loads
        self.mujoco_model.step = step_streamwise_free

    def _save_load_snapshot_if_requested(
        self,
        step: int,
        raw_forces_E: np.ndarray,
        raw_moments_E_Cg: np.ndarray,
        projected_forces_E: np.ndarray,
        projected_moments_E_Cg: np.ndarray,
    ) -> None:
        """Save sparse load snapshots for quick live checks."""
        if (
            self.streamed_load_history_dir is None
            or self.streamed_load_save_every_n_steps is None
        ):
            return
        if step % self.streamed_load_save_every_n_steps != 0:
            return
        clamp_forces_E = np.zeros_like(raw_forces_E)
        clamp_forces_E[:, 1:3] = -raw_forces_E[:, 1:3]
        np.savez_compressed(
            self.streamed_load_history_dir / f"step_{step:06d}.npz",
            step=np.array([step], dtype=int),
            raw_forces_E_N=raw_forces_E,
            raw_moments_E_Cg_Nm=raw_moments_E_Cg,
            projected_forces_E_N=projected_forces_E,
            projected_moments_E_Cg_Nm=projected_moments_E_Cg,
            clamp_forces_E_N=clamp_forces_E,
            clamp_moments_E_Cg_Nm=-raw_moments_E_Cg,
        )

    def _save_dense_row_if_requested(self) -> None:
        """Append one dense CSV row for real-time monitoring."""
        if (
            self.dense_diagnostics_csv_path is None
            or self.dense_diagnostics_every_n_steps is None
            or not self.raw_forces_E
        ):
            return
        local_step = len(self.raw_forces_E) - 1
        if local_step % self.dense_diagnostics_every_n_steps != 0:
            return

        positions_x = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        velocities_x = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        for body_index, qpos_adr in enumerate(self.mujoco_model.body_qposadrs):
            qvel_adr = int(self.mujoco_model.body_qveladrs[body_index])
            positions_x[body_index] = self.mujoco_model.data.qpos[qpos_adr]
            velocities_x[body_index] = self.mujoco_model.data.qvel[qvel_adr]

        raw_forces_E = self.raw_forces_E[-1]
        raw_moments_E_Cg = self.raw_moments_E_Cg[-1]
        projected_forces_E = self.projected_forces_E[-1]
        clamp_forces_E = np.zeros_like(raw_forces_E)
        clamp_forces_E[:, 1:3] = -raw_forces_E[:, 1:3]
        clamp_moments_E_Cg = -raw_moments_E_Cg
        global_step = self.global_step_offset + local_step
        row: dict[str, float | int] = {
            "local_step": local_step,
            "global_step": global_step,
            "time_s": global_step * self.delta_time_s,
            "time_over_flap_period": global_step
            * self.delta_time_s
            / single_case.FLAPPING_PERIOD_S,
            "front_x_m": float(positions_x[0]),
            "rear_x_m": float(positions_x[1]),
            "x_over_span": float((positions_x[0] - positions_x[1]) / self.span_m),
            "front_ux_mps": float(velocities_x[0]),
            "rear_ux_mps": float(velocities_x[1]),
            "relative_ux_mps": float(velocities_x[0] - velocities_x[1]),
        }
        for body_index, body_name in enumerate(("front", "rear")):
            row.update(
                {
                    f"{body_name}_raw_fx_N": float(raw_forces_E[body_index, 0]),
                    f"{body_name}_raw_fy_N": float(raw_forces_E[body_index, 1]),
                    f"{body_name}_raw_fz_N": float(raw_forces_E[body_index, 2]),
                    f"{body_name}_projected_fx_N": float(
                        projected_forces_E[body_index, 0]
                    ),
                    f"{body_name}_clamp_fy_N": float(clamp_forces_E[body_index, 1]),
                    f"{body_name}_clamp_fz_N": float(clamp_forces_E[body_index, 2]),
                    f"{body_name}_raw_mx_Nm": float(raw_moments_E_Cg[body_index, 0]),
                    f"{body_name}_raw_my_Nm": float(raw_moments_E_Cg[body_index, 1]),
                    f"{body_name}_raw_mz_Nm": float(raw_moments_E_Cg[body_index, 2]),
                    f"{body_name}_clamp_mx_Nm": float(
                        clamp_moments_E_Cg[body_index, 0]
                    ),
                    f"{body_name}_clamp_my_Nm": float(
                        clamp_moments_E_Cg[body_index, 1]
                    ),
                    f"{body_name}_clamp_mz_Nm": float(
                        clamp_moments_E_Cg[body_index, 2]
                    ),
                }
            )
        with self.dense_diagnostics_csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            writer.writerow(row)

    def enforce_state(self, record: bool) -> None:
        """Restore constrained directions while preserving each body's x and Ux."""
        target_quat_wxyz = _quat_from_izyx_angles_deg(self.target_angles_deg)
        velocities_E = np.zeros((self.mujoco_model.num_bodies, 3), dtype=float)
        angular_rates_rad_s = np.zeros((self.mujoco_model.num_bodies, 3), dtype=float)

        for body_index, qpos_adr in enumerate(self.mujoco_model.body_qposadrs):
            qvel_adr = int(self.mujoco_model.body_qveladrs[body_index])
            velocities_E[body_index] = self.mujoco_model.data.qvel[
                qvel_adr : qvel_adr + 3
            ]
            angular_rates_rad_s[body_index] = self.mujoco_model.data.qvel[
                qvel_adr + 3 : qvel_adr + 6
            ]
            if record:
                x_position_m = float(self.mujoco_model.data.qpos[qpos_adr])
                x_velocity_mps = float(self.mujoco_model.data.qvel[qvel_adr])
            else:
                x_position_m = float(self.target_positions_E_m[body_index, 0])
                x_velocity_mps = float(self.initial_streamwise_speed_mps[body_index])

            self.mujoco_model.data.qpos[qpos_adr] = x_position_m
            self.mujoco_model.data.qpos[qpos_adr + 1 : qpos_adr + 3] = (
                self.target_positions_E_m[body_index, 1:3]
            )
            self.mujoco_model.data.qpos[qpos_adr + 3 : qpos_adr + 7] = target_quat_wxyz
            self.mujoco_model.data.qvel[qvel_adr] = x_velocity_mps
            self.mujoco_model.data.qvel[qvel_adr + 1 : qvel_adr + 6] = 0.0

        self.forward_fn(self.mujoco_model.model, self.mujoco_model.data)
        if record:
            self.preclamp_velocities_E.append(velocities_E)
            self.preclamp_angular_rates_rad_s.append(angular_rates_rad_s)
            self._save_dense_row_if_requested()
            self.raise_if_diverged()

    def raise_if_diverged(self) -> None:
        """Stop if streamwise-free dynamics clearly diverged."""
        positions_x = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        speeds_x = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        for body_index, qpos_adr in enumerate(self.mujoco_model.body_qposadrs):
            qvel_adr = int(self.mujoco_model.body_qveladrs[body_index])
            positions_x[body_index] = self.mujoco_model.data.qpos[qpos_adr]
            speeds_x[body_index] = self.mujoco_model.data.qvel[qvel_adr]
        if not np.all(np.isfinite(positions_x)) or not np.all(np.isfinite(speeds_x)):
            raise RuntimeError("Two-flyer streamwise-free case diverged.")
        if np.max(np.abs(speeds_x)) > self.max_abs_speed_mps:
            raise RuntimeError("Two-flyer streamwise speed exceeded guard.")
        if np.max(np.abs(positions_x / self.span_m)) > self.max_abs_x_over_span:
            raise RuntimeError("Two-flyer streamwise position exceeded guard.")

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return recorded load histories."""
        raw_forces_E = _stack_history(
            self.raw_forces_E, self.mujoco_model.num_bodies, 3
        )
        raw_moments_E_Cg = _stack_history(
            self.raw_moments_E_Cg, self.mujoco_model.num_bodies, 3
        )
        projected_forces_E = _stack_history(
            self.projected_forces_E, self.mujoco_model.num_bodies, 3
        )
        projected_moments_E_Cg = _stack_history(
            self.projected_moments_E_Cg, self.mujoco_model.num_bodies, 3
        )
        n = min(len(raw_forces_E), len(raw_moments_E_Cg), len(projected_forces_E))
        raw_forces_E = raw_forces_E[:n]
        raw_moments_E_Cg = raw_moments_E_Cg[:n]
        projected_forces_E = projected_forces_E[:n]
        projected_moments_E_Cg = projected_moments_E_Cg[:n]
        clamp_forces_E = np.zeros_like(raw_forces_E)
        clamp_forces_E[:, :, 1:3] = -raw_forces_E[:, :, 1:3]
        return {
            "raw_forces_E_N": raw_forces_E,
            "raw_moments_E_Cg_Nm": raw_moments_E_Cg,
            "projected_forces_E_N": projected_forces_E,
            "projected_moments_E_Cg_Nm": projected_moments_E_Cg,
            "clamp_forces_E_N": clamp_forces_E,
            "clamp_moments_E_Cg_Nm": -raw_moments_E_Cg,
        }


def build_problem(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    initial_speed_mps: float | np.ndarray = DEFAULT_INITIAL_SPEED_MPS,
    weight_n: float = single_case.DEFAULT_WEIGHT_N,
    initial_positions_E_m: np.ndarray | None = None,
    phase_offset_deg: float = 0.0,
) -> tuple[
    ps.problems.MultiBodyCoupledUnsteadyProblem,
    ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
    MultiBodyStreamwiseFreeClampDiagnostics,
]:
    """Build one two-flyer streamwise-free flapping problem."""
    delta_time = single_case.FLAPPING_PERIOD_S / steps_per_flap
    front_airplane = single_case.build_airplane(weight_n=weight_n)
    rear_airplane = single_case.build_airplane(weight_n=weight_n)
    front_movement = single_case.build_airplane_movement(
        front_airplane,
        phase_offset_deg=phase_offset_deg,
    )
    rear_movement = single_case.build_airplane_movement(
        rear_airplane,
        phase_offset_deg=phase_offset_deg,
    )
    initial_speeds = np.broadcast_to(
        np.asarray(initial_speed_mps, dtype=float),
        (2,),
    ).copy()
    if initial_positions_E_m is None:
        initial_positions = body_positions_from_offsets(
            x_over_span=x_over_span,
            y_over_span=y_over_span,
            z_over_span=z_over_span,
        )
    else:
        checked_positions = np.asarray(initial_positions_E_m, dtype=float)
        if checked_positions.shape != (2, 3):
            raise ValueError("initial_positions_E_m must have shape (2, 3).")
        initial_positions = (checked_positions[0].copy(), checked_positions[1].copy())
    initial_operating_points = [
        ps.operating_point.CoupledOperatingPoint(
            rho=single_case.AIR_DENSITY,
            vCg__E=float(initial_speeds[body_index]),
            alpha=0.0,
            beta=0.0,
            angles_E_to_BP1_izyx=(0.0, 0.0, 0.0),
            externalFX_W=0.0,
            nu=single_case.KINEMATIC_VISCOSITY,
            g_E=single_case.GRAVITY_E,
        )
        for body_index in range(2)
    ]
    coupled_movement = ps.movements.movement.MultiBodyCoupledMovement(
        airplane_movements=[front_movement, rear_movement],
        initial_coupled_operating_points=initial_operating_points,
        initial_positions_E_E=initial_positions,
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    inertia = single_case.inertia_from_weight(weight_n)
    coupled_problem = ps.problems.MultiBodyCoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1s=[inertia.copy(), inertia.copy()],
    )
    coupled_solver = ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_problem
    )
    diagnostics = MultiBodyStreamwiseFreeClampDiagnostics(
        mujoco_model=coupled_problem.mujoco_model,
        target_positions_E_m=np.vstack(initial_positions),
        target_angles_deg=np.zeros(3, dtype=float),
        initial_streamwise_speed_mps=initial_speeds,
    )
    diagnostics.install()
    return coupled_problem, coupled_solver, diagnostics


def get_history_arrays(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, positions, and velocities from the multibody movement."""
    positions = np.asarray(coupled_problem.coupled_movement.positions_E_E, dtype=float)
    operating_points_by_step = coupled_problem.coupled_movement.coupled_operating_points
    velocities = np.array(
        [
            [op.vCg_E__E for op in operating_points]
            for operating_points in operating_points_by_step
        ],
        dtype=float,
    )
    times = np.arange(len(positions), dtype=float) * coupled_problem.delta_time
    return times, positions, velocities


def distance_convergence_from_csv(
    csv_path: Path,
    steps_per_flap: int,
    slope_tol_xb_per_period: float,
    window_periods: int = 5,
    min_global_step: int | None = None,
) -> dict[str, Any]:
    """Return cycle-mean X/B convergence diagnostics from a dense CSV."""
    if not Path(csv_path).exists():
        return {"converged": False, "reason": "missing_csv"}
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    x_values: list[float] = []
    for row in rows:
        if min_global_step is not None:
            try:
                if int(row["global_step"]) < min_global_step:
                    continue
            except (KeyError, ValueError):
                pass
        try:
            x_values.append(float(row["x_over_span"]))
        except (KeyError, ValueError):
            continue
    num_cycles = len(x_values) // steps_per_flap
    if num_cycles < window_periods:
        return {
            "converged": False,
            "num_cycles": num_cycles,
            "min_global_step": min_global_step,
            "reason": "not_enough_cycles",
        }
    trimmed = np.asarray(x_values[: num_cycles * steps_per_flap], dtype=float)
    cycle_means = trimmed.reshape((num_cycles, steps_per_flap)).mean(axis=1)
    window = cycle_means[-window_periods:]
    cycle_ids = np.arange(num_cycles - window_periods + 1, num_cycles + 1, dtype=float)
    slope = float(np.polyfit(cycle_ids, window, 1)[0])
    return {
        "converged": bool(abs(slope) <= slope_tol_xb_per_period),
        "num_cycles": int(num_cycles),
        "min_global_step": min_global_step,
        "cycle_mean_x_over_span": cycle_means.tolist(),
        "final_window_slope_xb_per_period": slope,
        "distance_slope_tol_xb_per_period": float(slope_tol_xb_per_period),
        "final_5_cycle_mean_x_over_span": float(np.mean(window)),
    }


def save_case_plots(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    load_history: dict[str, np.ndarray],
) -> dict[str, str]:
    """Save quick-look velocity, separation, and load plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_over_span = (positions_E_E[:, 0, 0] - positions_E_E[:, 1, 0]) / FULL_SPAN_M

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times_s, x_over_span)
    axes[0].set_ylabel("X/B")
    axes[0].set_title("Streamwise separation")
    axes[0].grid(True)
    axes[1].plot(times_s, velocities_E__E[:, 0, 0], label="Front Ux")
    axes[1].plot(times_s, velocities_E__E[:, 1, 0], label="Rear Ux")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Ux (m/s)")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    separation_path = output_dir / "separation_velocity_history.png"
    fig.savefig(separation_path, dpi=160)
    plt.close(fig)

    raw_forces = load_history["raw_forces_E_N"]
    clamp_forces = load_history["clamp_forces_E_N"]
    n = min(len(times_s), len(raw_forces))
    load_path = output_dir / "force_clamp_history.png"
    if n > 0:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        t = times_s[-n:]
        axes[0].plot(t, raw_forces[:n, 0, 0], label="Front raw Fx")
        axes[0].plot(t, raw_forces[:n, 1, 0], label="Rear raw Fx")
        axes[0].set_ylabel("Fx (N)")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(t, clamp_forces[:n, 0, 1], label="Front clamp Fy")
        axes[1].plot(t, clamp_forces[:n, 0, 2], label="Front clamp Fz")
        axes[1].plot(t, clamp_forces[:n, 1, 1], label="Rear clamp Fy")
        axes[1].plot(t, clamp_forces[:n, 1, 2], label="Rear clamp Fz")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Clamp force (N)")
        axes[1].grid(True)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(load_path, dpi=160)
        plt.close(fig)

    return {
        "separation_velocity_plot": str(separation_path),
        "force_clamp_plot": str(load_path),
    }


def write_summary(
    output_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    initial_speed_mps: float,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    load_history: dict[str, np.ndarray],
    plot_paths: dict[str, str],
    restart_checkpoint_dir: Path | None = None,
    convergence: dict[str, Any] | None = None,
) -> Path:
    """Write summary.json for a completed case."""
    free_start = min(prescribed_num_steps, len(times_s) - 1)
    final_period_steps = min(steps_per_flap, len(times_s) - free_start)
    final_slice = slice(len(times_s) - final_period_steps, len(times_s))
    final_x_over_span = (
        positions_E_E[final_slice, 0, 0] - positions_E_E[final_slice, 1, 0]
    ) / FULL_SPAN_M
    summary: dict[str, Any] = {
        "case": "readme_scaled_flapping_two_flyer_streamwise_free_x",
        "body_convention": "body_1_front_body_2_rear",
        "prescribed_wake": False,
        "wake_model": "free",
        "clamp_mode": "independent_x_and_ux_free_yz_and_all_rotations_clamped",
        "x_over_span_initial": x_over_span,
        "y_over_span_initial": y_over_span,
        "z_over_span_initial": z_over_span,
        "initial_positions_E_m": [
            value.tolist()
            for value in body_positions_from_offsets(
                x_over_span, y_over_span, z_over_span
            )
        ],
        "initial_speed_mps": np.asarray(initial_speed_mps, dtype=float).tolist(),
        "steps_per_flap": steps_per_flap,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "periods_free": free_num_steps / steps_per_flap,
        "periods_total": (prescribed_num_steps + free_num_steps) / steps_per_flap,
        "final_mean_x_over_span_last_period": float(np.mean(final_x_over_span)),
        "final_mean_front_ux_mps_last_period": float(
            np.mean(velocities_E__E[final_slice, 0, 0])
        ),
        "final_mean_rear_ux_mps_last_period": float(
            np.mean(velocities_E__E[final_slice, 1, 0])
        ),
        "final_mean_relative_ux_mps_last_period": float(
            np.mean(
                velocities_E__E[final_slice, 0, 0] - velocities_E__E[final_slice, 1, 0]
            )
        ),
        "plot_paths": plot_paths,
    }
    if restart_checkpoint_dir is not None:
        summary["restart_checkpoint_dir"] = str(restart_checkpoint_dir)
        latest_checkpoint = latest_restart_checkpoint(output_dir)
        summary["latest_restart_checkpoint"] = (
            None if latest_checkpoint is None else str(latest_checkpoint)
        )
    if convergence is not None:
        summary["distance_convergence"] = convergence
    raw_forces = load_history["raw_forces_E_N"]
    if len(raw_forces) >= final_period_steps:
        force_slice = slice(len(raw_forces) - final_period_steps, len(raw_forces))
        summary["final_mean_raw_fx_N_last_period"] = np.mean(
            raw_forces[force_slice, :, 0], axis=0
        ).tolist()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def run_case(
    output_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    initial_speed_mps: float | np.ndarray,
    save_every_n_steps: int,
    dense_diagnostics_every_n_steps: int,
    show_progress: bool,
    restart_state: MultiBodyRestartState | None = None,
    restart_global_step_offset: int | None = None,
    restart_checkpoint_every_n_steps: int | None = None,
    distance_slope_tol_xb_per_period: float | None = None,
) -> Path:
    """Run one two-flyer streamwise-free flapping case."""
    if not initial_condition_is_collision_free(x_over_span, y_over_span, z_over_span):
        raise ValueError(
            f"Initial condition collides: X/B={x_over_span}, "
            f"Y/B={y_over_span}, Z/B={z_over_span}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "streamed_history"
    restart_checkpoint_dir = output_dir / "restart_checkpoints"
    initial_positions_E_m = None
    phase_offset_deg = 0.0
    if restart_state is not None:
        initial_positions_E_m = restart_state.positions_E_E
        initial_speed_mps = restart_state.velocities_E__E[:, 0]
        phase_offset_deg = flapping_phase_offset_deg(
            restart_state.global_step,
            steps_per_flap,
        )
        if restart_global_step_offset is None:
            restart_global_step_offset = restart_state.global_step
    if restart_global_step_offset is None:
        restart_global_step_offset = 0
    coupled_problem, coupled_solver, diagnostics = build_problem(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        initial_speed_mps=initial_speed_mps,
        initial_positions_E_m=initial_positions_E_m,
        phase_offset_deg=phase_offset_deg,
    )
    diagnostics.configure_streaming(
        history_save_dir=history_dir,
        save_every_n_steps=save_every_n_steps,
        dense_diagnostics_every_n_steps=dense_diagnostics_every_n_steps,
        global_step_offset=restart_global_step_offset,
        delta_time_s=coupled_problem.delta_time,
    )
    coupled_solver.run(
        prescribed_wake=False,
        show_progress=show_progress,
        history_stride=steps_per_flap,
        save_every_n_steps=save_every_n_steps,
        history_save_dir=history_dir,
        restart_state=restart_state,
        restart_checkpoint_dir=restart_checkpoint_dir,
        restart_checkpoint_every_n_steps=(
            restart_checkpoint_every_n_steps or steps_per_flap
        ),
        restart_global_step_offset=restart_global_step_offset,
    )
    times_s, positions_E_E, velocities_E__E = get_history_arrays(coupled_problem)
    load_history = diagnostics.history_arrays()
    np.savez_compressed(
        output_dir / "history.npz",
        times_s=times_s,
        positions_E_m=positions_E_E,
        velocities_E_mps=velocities_E__E,
        **load_history,
    )
    plot_paths = save_case_plots(
        output_dir=output_dir,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        load_history=load_history,
    )
    dense_csv_path = history_dir / "dense_speed_force_history.csv"
    convergence = None
    if distance_slope_tol_xb_per_period is not None:
        convergence = distance_convergence_from_csv(
            dense_csv_path,
            steps_per_flap=steps_per_flap,
            slope_tol_xb_per_period=distance_slope_tol_xb_per_period,
            min_global_step=restart_global_step_offset + prescribed_num_steps,
        )
    return write_summary(
        output_dir=output_dir,
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        initial_speed_mps=initial_speed_mps,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        load_history=load_history,
        plot_paths=plot_paths,
        restart_checkpoint_dir=restart_checkpoint_dir,
        convergence=convergence,
    )


def combine_segment_dense_csvs(case_dir: Path) -> Path | None:
    """Combine segment dense CSV files into a top-level live-readable CSV."""
    segment_csvs = sorted(
        Path(case_dir).glob(
            "segments/segment_*/streamed_history/dense_speed_force_history.csv"
        )
    )
    if not segment_csvs:
        return None
    output_csv = Path(case_dir) / "combined_dense_speed_force_history.csv"
    header: list[str] | None = None
    rows_by_global_step: dict[int, list[str]] = {}
    with output_csv.open("w", newline="") as output_file:
        for csv_path in segment_csvs:
            with csv_path.open(newline="") as input_file:
                reader = csv.reader(input_file)
                try:
                    this_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = this_header
                if this_header != header:
                    raise ValueError(
                        f"Dense CSV header mismatch while combining {csv_path}."
                    )
                global_step_index = header.index("global_step")
                for row in reader:
                    if not row:
                        continue
                    global_step = int(float(row[global_step_index]))
                    # Later segment files replace earlier duplicate steps. This keeps
                    # resumed cases coherent when a preliminary or failed extension
                    # wrote a few rows from the same checkpoint.
                    rows_by_global_step[global_step] = row
        if header is None:
            return None
        writer = csv.writer(output_file)
        writer.writerow(header)
        for global_step in sorted(rows_by_global_step):
            writer.writerow(rows_by_global_step[global_step])
    return output_csv


def write_failed_case_summary(
    case_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    error: BaseException,
) -> Path:
    """Write a top-level summary for a case that stopped on an exception."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combine_segment_dense_csvs(case_dir)
    latest_checkpoint = latest_restart_checkpoint(case_dir)
    payload: dict[str, Any] = {
        "case": "readme_scaled_flapping_two_flyer_streamwise_free_x_restartable",
        "status": "failed",
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "body_convention": "body_1_front_body_2_rear",
        "x_over_span_initial": x_over_span,
        "y_over_span_initial": y_over_span,
        "z_over_span_initial": z_over_span,
        "error_type": type(error).__name__,
        "error": str(error),
        "combined_dense_csv": None if combined_csv is None else str(combined_csv),
        "latest_restart_checkpoint": (
            None if latest_checkpoint is None else str(latest_checkpoint)
        ),
        "distance_convergence": {
            "converged": False,
            "reason": "case_failed",
        },
    }
    summary_path = case_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary_path


def run_case_until_distance_converged(
    case_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    initial_free_steps: int,
    extension_steps: int,
    max_free_steps: int,
    steps_per_flap: int,
    initial_speed_mps: float,
    save_every_n_steps: int,
    dense_diagnostics_every_n_steps: int,
    show_progress: bool,
    slope_tol_xb_per_period: float,
    restart_from: Path | None = None,
) -> Path:
    """Run or extend one case in restartable segments until X/B converges."""
    case_dir.mkdir(parents=True, exist_ok=True)
    completed_free_steps = 0
    remaining_prescribed_steps = prescribed_num_steps
    existing_segment_indices = []
    for segment_dir in (case_dir / "segments").glob("segment_*"):
        try:
            existing_segment_indices.append(int(segment_dir.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    segment_index = max(existing_segment_indices) + 1 if existing_segment_indices else 0
    restart_state: MultiBodyRestartState | None = None
    if restart_from is None:
        restart_from = latest_restart_checkpoint(case_dir)
    if restart_from is not None:
        restart_state = load_restart_state(restart_from)
        restart_global_step = int(restart_state.global_step)
        remaining_prescribed_steps = max(0, prescribed_num_steps - restart_global_step)
        completed_free_steps = max(0, restart_global_step - prescribed_num_steps)

    summary_path = case_dir / "summary.json"
    while completed_free_steps < max_free_steps:
        this_segment_dir = case_dir / "segments" / f"segment_{segment_index:03d}"
        this_prescribed_steps = (
            prescribed_num_steps
            if restart_state is None
            else remaining_prescribed_steps
        )
        this_free_steps = (
            initial_free_steps if completed_free_steps == 0 else extension_steps
        )
        remaining = max_free_steps - completed_free_steps
        this_free_steps = min(this_free_steps, remaining)
        summary_path = run_case(
            output_dir=this_segment_dir,
            x_over_span=x_over_span,
            y_over_span=y_over_span,
            z_over_span=z_over_span,
            prescribed_num_steps=this_prescribed_steps,
            free_num_steps=this_free_steps,
            steps_per_flap=steps_per_flap,
            initial_speed_mps=initial_speed_mps,
            save_every_n_steps=save_every_n_steps,
            dense_diagnostics_every_n_steps=dense_diagnostics_every_n_steps,
            show_progress=show_progress,
            restart_state=restart_state,
            restart_global_step_offset=(
                0 if restart_state is None else restart_state.global_step
            ),
            restart_checkpoint_every_n_steps=steps_per_flap,
            distance_slope_tol_xb_per_period=slope_tol_xb_per_period,
        )
        completed_free_steps += this_free_steps
        remaining_prescribed_steps = 0
        combined_csv = combine_segment_dense_csvs(case_dir)
        convergence = (
            {"converged": False, "reason": "missing_combined_csv"}
            if combined_csv is None
            else distance_convergence_from_csv(
                combined_csv,
                steps_per_flap=steps_per_flap,
                slope_tol_xb_per_period=slope_tol_xb_per_period,
                min_global_step=prescribed_num_steps,
            )
        )
        latest_checkpoint = latest_restart_checkpoint(case_dir)
        top_summary = {
            "case": "readme_scaled_flapping_two_flyer_streamwise_free_x_restartable",
            "status": (
                "converged"
                if convergence.get("converged", False)
                else "running_or_unconverged"
            ),
            "body_convention": "body_1_front_body_2_rear",
            "x_over_span_initial": x_over_span,
            "y_over_span_initial": y_over_span,
            "z_over_span_initial": z_over_span,
            "segments_completed": segment_index + 1,
            "completed_free_steps": completed_free_steps,
            "completed_free_periods": completed_free_steps / steps_per_flap,
            "max_free_periods": max_free_steps / steps_per_flap,
            "combined_dense_csv": None if combined_csv is None else str(combined_csv),
            "latest_restart_checkpoint": (
                None if latest_checkpoint is None else str(latest_checkpoint)
            ),
            "distance_convergence": convergence,
            "last_segment_summary": str(summary_path),
        }
        summary_path = case_dir / "summary.json"
        summary_path.write_text(json.dumps(top_summary, indent=2) + "\n")
        if convergence.get("converged", False):
            return summary_path
        if latest_checkpoint is None:
            raise RuntimeError(f"No restart checkpoint found under {case_dir}.")
        restart_state = load_restart_state(latest_checkpoint)
        segment_index += 1
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        summary = {}
    summary["status"] = "max_free_periods_reached"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run README-scaled two-flyer flapping streamwise-free cases."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--x-over-span", type=float, default=None)
    parser.add_argument("--y-over-span", type=float, default=None)
    parser.add_argument("--z-over-span", type=float, default=0.0)
    parser.add_argument("--run-grid", action="store_true")
    parser.add_argument(
        "--prescribed-steps", type=int, default=DEFAULT_PRESCRIBED_STEPS
    )
    parser.add_argument("--free-steps", type=int, default=DEFAULT_FREE_STEPS)
    parser.add_argument("--steps-per-flap", type=int, default=DEFAULT_STEPS_PER_FLAP)
    parser.add_argument(
        "--initial-speed", type=float, default=DEFAULT_INITIAL_SPEED_MPS
    )
    parser.add_argument("--save-every-n-steps", type=int, default=2)
    parser.add_argument("--dense-diagnostics-every-n-steps", type=int, default=1)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--restart-from", type=Path, default=None)
    parser.add_argument("--restart-from-latest", type=Path, default=None)
    parser.add_argument("--extend-until-distance-converged", action="store_true")
    parser.add_argument("--extension-periods", type=int, default=10)
    parser.add_argument("--max-free-periods", type=int, default=80)
    parser.add_argument("--distance-slope-tol-xb-per-period", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    """Run one case or the default 20-case grid sequentially."""
    args = parse_args()
    cases: list[tuple[float, float, float]]
    if args.run_grid:
        cases = [
            (x, y, z)
            for x in GRID_X_OVER_SPAN
            for y in GRID_Y_OVER_SPAN
            for z in GRID_Z_OVER_SPAN
        ]
    else:
        if args.x_over_span is None or args.y_over_span is None:
            raise ValueError(
                "--x-over-span and --y-over-span are required without --run-grid."
            )
        cases = [(args.x_over_span, args.y_over_span, args.z_over_span)]

    for x_over_span, y_over_span, z_over_span in cases:
        case_dir = args.output_root / run_label(x_over_span, y_over_span, z_over_span)
        restart_from = args.restart_from
        if args.restart_from_latest is not None:
            restart_from = latest_restart_checkpoint(args.restart_from_latest)
            if restart_from is None:
                raise RuntimeError(
                    f"No restart-compatible checkpoint under {args.restart_from_latest}."
                )
        try:
            if args.extend_until_distance_converged:
                summary_path = run_case_until_distance_converged(
                    case_dir=case_dir,
                    x_over_span=x_over_span,
                    y_over_span=y_over_span,
                    z_over_span=z_over_span,
                    prescribed_num_steps=args.prescribed_steps,
                    initial_free_steps=args.free_steps,
                    extension_steps=args.extension_periods * args.steps_per_flap,
                    max_free_steps=args.max_free_periods * args.steps_per_flap,
                    steps_per_flap=args.steps_per_flap,
                    initial_speed_mps=args.initial_speed,
                    save_every_n_steps=args.save_every_n_steps,
                    dense_diagnostics_every_n_steps=args.dense_diagnostics_every_n_steps,
                    show_progress=args.show_progress,
                    slope_tol_xb_per_period=args.distance_slope_tol_xb_per_period,
                    restart_from=restart_from,
                )
            else:
                restart_state = (
                    None if restart_from is None else load_restart_state(restart_from)
                )
                summary_path = run_case(
                    output_dir=case_dir,
                    x_over_span=x_over_span,
                    y_over_span=y_over_span,
                    z_over_span=z_over_span,
                    prescribed_num_steps=args.prescribed_steps,
                    free_num_steps=args.free_steps,
                    steps_per_flap=args.steps_per_flap,
                    initial_speed_mps=args.initial_speed,
                    save_every_n_steps=args.save_every_n_steps,
                    dense_diagnostics_every_n_steps=args.dense_diagnostics_every_n_steps,
                    show_progress=args.show_progress,
                    restart_state=restart_state,
                    distance_slope_tol_xb_per_period=(
                        args.distance_slope_tol_xb_per_period
                    ),
                )
        except Exception as error:
            summary_path = write_failed_case_summary(
                case_dir=case_dir,
                x_over_span=x_over_span,
                y_over_span=y_over_span,
                z_over_span=z_over_span,
                error=error,
            )
            print(f"Saved failed summary to: {summary_path}")
            raise
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
