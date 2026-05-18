"""Run a README-scaled flapping wing with only streamwise motion free."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
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
except ImportError:
    import free_flight_case_utils as ff_utils

if not hasattr(ps.operating_point, "CoupledOperatingPoint"):
    raise RuntimeError(
        "This example requires the coupled free-flight API from "
        "`origin/feature/free_flight`."
    )


AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY = 15.06e-6
GRAVITY_E = (0.0, 0.0, 9.80665)

AIRFOIL_NAME = "naca0012"
FULL_SPAN_M = 1.0
SEMI_SPAN_M = FULL_SPAN_M / 2.0
ROOT_CHORD_M = 0.11666666666666667
TIP_CHORD_M = 0.08333333333333333
MEAN_CHORD_M = 0.5 * (ROOT_CHORD_M + TIP_CHORD_M)
REFERENCE_AREA_M2 = FULL_SPAN_M * MEAN_CHORD_M
ROOT_LE_Y_FROM_SYMMETRY_M = SEMI_SPAN_M * 0.5 / (0.5 + 7.0)
TIP_Y_FROM_ROOT_M = SEMI_SPAN_M * 7.0 / (0.5 + 7.0)
TIP_X_FROM_ROOT_M = 0.75 / 7.0 * TIP_Y_FROM_ROOT_M
TIP_Z_FROM_ROOT_M = 0.5 / 7.0 * TIP_Y_FROM_ROOT_M

WING_INCIDENCE_DEG = 30.0
TIP_TWIST_DEG = 5.0
LEFT_TIP_REFERENCE_INCIDENCE_DEG = WING_INCIDENCE_DEG + TIP_TWIST_DEG
INITIAL_YAW_DEG = 0.0
INITIAL_PITCH_DEG = 0.0
INITIAL_ROLL_DEG = 0.0
INITIAL_SPEED_MPS = 1.0

FLAPPING_AMPLITUDE_DEG = 15.0
FLAPPING_FREQUENCY_HZ = 1.0
FLAPPING_PERIOD_S = 1.0 / FLAPPING_FREQUENCY_HZ
DEFAULT_STEPS_PER_FLAP = 48

NUM_CHORDWISE_PANELS = 6
NUM_SPANWISE_PANELS_PER_HALF = 12

SMALL_FREE_FLIGHT_WING_LOADING_N_M2 = 0.0235 / (0.22 * 0.035)
DEFAULT_WEIGHT_N = SMALL_FREE_FLIGHT_WING_LOADING_N_M2 * REFERENCE_AREA_M2

DEFAULT_PRESCRIBED_STEPS = 3 * DEFAULT_STEPS_PER_FLAP
DEFAULT_FREE_STEPS = 30 * DEFAULT_STEPS_PER_FLAP
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "readme_scaled_flapping_streamwise"
    / "free_x"
)


def inertia_from_weight(weight_n: float) -> np.ndarray:
    """Return a simple positive wing-like inertia matrix."""
    mass_kg = weight_n / np.linalg.norm(GRAVITY_E)
    chord_m = MEAN_CHORD_M
    span_m = FULL_SPAN_M
    return np.diag(
        [
            mass_kg * span_m**2 / 12.0,
            mass_kg * chord_m**2 / 12.0,
            mass_kg * (span_m**2 + chord_m**2) / 12.0,
        ]
    )


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
    """Stack one-body vector histories into a dense array."""
    if not history:
        return np.zeros((0, width), dtype=float)
    return np.stack(history, axis=0)


@dataclass
class StreamwiseFreeClampDiagnostics:
    """Patch a single-body MuJoCo model so only x and Ux remain free."""

    mujoco_model: object
    target_position_E_m: np.ndarray
    target_angles_deg: np.ndarray
    initial_streamwise_speed_mps: float = INITIAL_SPEED_MPS
    max_abs_speed_mps: float = 50.0
    max_abs_x_m: float = 200.0
    forward_fn: Callable[[Any, Any], None] = mujoco.mj_forward

    def __post_init__(self) -> None:
        self.target_position_E_m = np.asarray(self.target_position_E_m, dtype=float)
        self.target_angles_deg = np.asarray(self.target_angles_deg, dtype=float)
        if self.target_position_E_m.shape != (3,):
            raise ValueError("target_position_E_m must have shape (3,).")
        if self.target_angles_deg.shape != (3,):
            raise ValueError("target_angles_deg must have shape (3,).")
        if self.max_abs_speed_mps <= 0.0:
            raise ValueError("max_abs_speed_mps must be positive.")
        if self.max_abs_x_m <= 0.0:
            raise ValueError("max_abs_x_m must be positive.")

        self.raw_forces_E: list[np.ndarray] = []
        self.raw_moments_E_Cg: list[np.ndarray] = []
        self.projected_forces_E: list[np.ndarray] = []
        self.projected_moments_E_Cg: list[np.ndarray] = []
        self.preclamp_velocities_E: list[np.ndarray] = []
        self.preclamp_angular_rates_rad_s: list[np.ndarray] = []
        self.streamed_load_history_dir: Path | None = None
        self.streamed_load_save_every_n_steps: int | None = None
        self.dense_diagnostics_csv_path: Path | None = None
        self.dense_diagnostics_every_n_steps: int | None = None
        self.global_step_offset: int = 0
        self.delta_time_s: float = 0.0

    def configure_load_streaming(
        self,
        history_save_dir: Path | None,
        save_every_n_steps: int | None,
        *,
        dense_diagnostics_every_n_steps: int | None = None,
        global_step_offset: int = 0,
        delta_time_s: float = 0.0,
    ) -> None:
        """Configure lightweight force/torque snapshots for live inspection."""
        if history_save_dir is None:
            self.streamed_load_history_dir = None
            self.streamed_load_save_every_n_steps = None
            self.dense_diagnostics_csv_path = None
            self.dense_diagnostics_every_n_steps = None
            return
        history_save_dir = Path(history_save_dir)
        if save_every_n_steps is None:
            self.streamed_load_history_dir = None
            self.streamed_load_save_every_n_steps = None
        else:
            self.streamed_load_history_dir = history_save_dir / "clamp_loads"
            self.streamed_load_history_dir.mkdir(parents=True, exist_ok=True)
            self.streamed_load_save_every_n_steps = int(save_every_n_steps)
        if dense_diagnostics_every_n_steps is None:
            self.dense_diagnostics_csv_path = None
            self.dense_diagnostics_every_n_steps = None
        else:
            history_save_dir.mkdir(parents=True, exist_ok=True)
            self.dense_diagnostics_csv_path = (
                history_save_dir / "dense_speed_force_history.csv"
            )
            self.dense_diagnostics_every_n_steps = int(dense_diagnostics_every_n_steps)
            self._initialize_dense_diagnostics_csv()
        self.global_step_offset = int(global_step_offset)
        self.delta_time_s = float(delta_time_s)

    def _initialize_dense_diagnostics_csv(self) -> None:
        """Create the dense live-readable speed/load CSV."""
        if self.dense_diagnostics_csv_path is None:
            return
        fieldnames = [
            "local_step",
            "global_step",
            "time_s",
            "time_over_flap_period",
            "x_m",
            "ux_mps",
            "raw_fx_N",
            "raw_fy_N",
            "raw_fz_N",
            "projected_fx_N",
            "projected_fy_N",
            "projected_fz_N",
            "clamp_fy_N",
            "clamp_fz_N",
            "raw_mx_Nm",
            "raw_my_Nm",
            "raw_mz_Nm",
            "clamp_mx_Nm",
            "clamp_my_Nm",
            "clamp_mz_Nm",
        ]
        with self.dense_diagnostics_csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def install(self) -> None:
        """Install load projection and post-step constrained-state clamping."""
        self.enforce_state(record=False)
        original_apply_loads = self.mujoco_model.apply_loads
        original_step = self.mujoco_model.step

        def apply_streamwise_free_loads(
            forces_E: np.ndarray,
            moments_E_Cg: np.ndarray,
        ) -> None:
            raw_forces_E = np.asarray(forces_E, dtype=float).copy()
            raw_moments_E_Cg = np.asarray(moments_E_Cg, dtype=float).copy()
            projected_forces_E = np.zeros(3, dtype=float)
            projected_forces_E[0] = raw_forces_E[0]
            projected_moments_E_Cg = np.zeros(3, dtype=float)

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
        """Save one force/torque snapshot when disk streaming is enabled."""
        if (
            self.streamed_load_history_dir is None
            or self.streamed_load_save_every_n_steps is None
        ):
            return
        if step % self.streamed_load_save_every_n_steps != 0:
            return
        clamp_forces_E = np.array([0.0, -raw_forces_E[1], -raw_forces_E[2]])
        np.savez_compressed(
            self.streamed_load_history_dir / f"step_{step:06d}.npz",
            step=np.array([step], dtype=int),
            raw_forces_E_N=raw_forces_E.copy(),
            raw_moments_E_Cg_Nm=raw_moments_E_Cg.copy(),
            projected_forces_E_N=projected_forces_E.copy(),
            projected_moments_E_Cg_Nm=projected_moments_E_Cg.copy(),
            clamp_forces_E_N=clamp_forces_E,
            clamp_moments_E_Cg_Nm=-raw_moments_E_Cg.copy(),
        )

    def _save_dense_diagnostics_row_if_requested(self) -> None:
        """Append one small speed/load row for real-time monitoring."""
        if (
            self.dense_diagnostics_csv_path is None
            or self.dense_diagnostics_every_n_steps is None
            or not self.raw_forces_E
            or not self.raw_moments_E_Cg
            or not self.projected_forces_E
            or not self.projected_moments_E_Cg
        ):
            return
        local_step = len(self.raw_forces_E) - 1
        if local_step % self.dense_diagnostics_every_n_steps != 0:
            return
        global_step = self.global_step_offset + local_step
        raw_forces_E = self.raw_forces_E[-1]
        raw_moments_E_Cg = self.raw_moments_E_Cg[-1]
        projected_forces_E = self.projected_forces_E[-1]
        clamp_forces_E = np.array([0.0, -raw_forces_E[1], -raw_forces_E[2]])
        clamp_moments_E_Cg = -raw_moments_E_Cg
        with self.dense_diagnostics_csv_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    local_step,
                    global_step,
                    global_step * self.delta_time_s,
                    global_step * self.delta_time_s / FLAPPING_PERIOD_S,
                    float(self.mujoco_model.data.qpos[0]),
                    float(self.mujoco_model.data.qvel[0]),
                    *raw_forces_E.tolist(),
                    *projected_forces_E.tolist(),
                    float(clamp_forces_E[1]),
                    float(clamp_forces_E[2]),
                    *raw_moments_E_Cg.tolist(),
                    *clamp_moments_E_Cg.tolist(),
                ]
            )

    def enforce_state(self, record: bool) -> None:
        """Clamp y, z, and attitude while preserving x and Ux."""
        velocity_E = np.copy(self.mujoco_model.data.qvel[0:3])
        angular_rate_rad_s = np.copy(self.mujoco_model.data.qvel[3:6])

        if record:
            x_position_m = float(self.mujoco_model.data.qpos[0])
            x_velocity_mps = float(self.mujoco_model.data.qvel[0])
        else:
            x_position_m = float(self.target_position_E_m[0])
            x_velocity_mps = float(self.initial_streamwise_speed_mps)

        target_quat_wxyz = _quat_from_izyx_angles_deg(self.target_angles_deg)
        self.mujoco_model.data.qpos[0] = x_position_m
        self.mujoco_model.data.qpos[1:3] = self.target_position_E_m[1:3]
        self.mujoco_model.data.qpos[3:7] = target_quat_wxyz
        self.mujoco_model.data.qvel[0] = x_velocity_mps
        self.mujoco_model.data.qvel[1:6] = 0.0

        self.forward_fn(self.mujoco_model.model, self.mujoco_model.data)
        if record:
            self.preclamp_velocities_E.append(velocity_E)
            self.preclamp_angular_rates_rad_s.append(angular_rate_rad_s)
            self._save_dense_diagnostics_row_if_requested()
            self.raise_if_diverged()

    def raise_if_diverged(self) -> None:
        """Stop early if the streamwise-free body has clearly diverged."""
        speed_mps = float(self.mujoco_model.data.qvel[0])
        position_m = float(self.mujoco_model.data.qpos[0])
        if not np.isfinite(speed_mps) or not np.isfinite(position_m):
            raise RuntimeError("Streamwise-free flapping case diverged.")
        if abs(speed_mps) > self.max_abs_speed_mps:
            raise RuntimeError(
                "Streamwise-free flapping case exceeded speed guard: "
                f"|Ux|={abs(speed_mps):.3g} m/s."
            )
        if abs(position_m) > self.max_abs_x_m:
            raise RuntimeError(
                "Streamwise-free flapping case exceeded position guard: "
                f"|x|={abs(position_m):.3g} m."
            )

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return recorded load and clamp histories as arrays."""
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
            len(projected_forces_E),
            len(projected_moments_E_Cg),
            len(preclamp_velocities_E),
            len(preclamp_angular_rates_rad_s),
        )
        if num_steps == 0:
            return {
                "raw_forces_E_N": np.zeros((0, 3), dtype=float),
                "raw_moments_E_Cg_Nm": np.zeros((0, 3), dtype=float),
                "projected_forces_E_N": np.zeros((0, 3), dtype=float),
                "projected_moments_E_Cg_Nm": np.zeros((0, 3), dtype=float),
                "preclamp_velocities_E_mps": np.zeros((0, 3), dtype=float),
                "preclamp_angular_rates_rad_s": np.zeros((0, 3), dtype=float),
                "clamp_forces_E_N": np.zeros((0, 3), dtype=float),
                "clamp_moments_E_Cg_Nm": np.zeros((0, 3), dtype=float),
                "clamp_power_proxy_W": np.zeros(0, dtype=float),
            }
        raw_forces_E = raw_forces_E[:num_steps]
        raw_moments_E_Cg = raw_moments_E_Cg[:num_steps]
        projected_forces_E = projected_forces_E[:num_steps]
        projected_moments_E_Cg = projected_moments_E_Cg[:num_steps]
        preclamp_velocities_E = preclamp_velocities_E[-num_steps:]
        preclamp_angular_rates_rad_s = preclamp_angular_rates_rad_s[-num_steps:]

        clamp_forces_E = np.zeros_like(raw_forces_E)
        clamp_forces_E[:, 1:3] = -raw_forces_E[:, 1:3]
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


def build_flapping_wing() -> ps.geometry.wing.Wing:
    """Build the README-style half wing that PteraSoftware mirrors into two wings."""
    return ps.geometry.wing.Wing(
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name=AIRFOIL_NAME),
                chord=ROOT_CHORD_M,
                num_spanwise_panels=NUM_SPANWISE_PANELS_PER_HALF,
                control_surface_symmetry_type="symmetric",
                spanwise_spacing="cosine",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name=AIRFOIL_NAME),
                chord=TIP_CHORD_M,
                Lp_Wcsp_Lpp=(
                    TIP_X_FROM_ROOT_M,
                    TIP_Y_FROM_ROOT_M,
                    TIP_Z_FROM_ROOT_M,
                ),
                angles_Wcsp_to_Wcs_ixyz=(0.0, TIP_TWIST_DEG, 0.0),
                control_surface_symmetry_type="symmetric",
                num_spanwise_panels=None,
            ),
        ],
        name="README-Scaled Main Wing",
        Ler_Gs_Cgs=(0.0, ROOT_LE_Y_FROM_SYMMETRY_M, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, WING_INCIDENCE_DEG, 0.0),
        symmetric=True,
        mirror_only=False,
        symmetryNormal_G=(0.0, 1.0, 0.0),
        symmetryPoint_G_Cg=(0.0, 0.0, 0.0),
        num_chordwise_panels=NUM_CHORDWISE_PANELS,
        chordwise_spacing="uniform",
    )


def build_airplane(weight_n: float = DEFAULT_WEIGHT_N) -> ps.geometry.airplane.Airplane:
    """Build the README-scaled symmetric flapping main wing."""
    return ps.geometry.airplane.Airplane(
        wings=[build_flapping_wing()],
        name="README-Scaled Flapping Streamwise Wing",
        weight=weight_n,
    )


def build_wing_cross_section_movements(
    wing: ps.geometry.wing.Wing,
) -> list[ps.movements.wing_cross_section_movement.WingCrossSectionMovement]:
    """Create static cross-section movements for the rigid flapping wing."""
    return [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section
        )
        for wing_cross_section in wing.wing_cross_sections
    ]


def build_wing_movement(
    wing: ps.geometry.wing.Wing,
    phase_offset_deg: float = 0.0,
) -> ps.movements.wing_movement.WingMovement:
    """Create the README-style single-axis flapping motion."""
    return ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=build_wing_cross_section_movements(wing),
        rotationPointOffset_Gs_Ler=(0.0, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(FLAPPING_AMPLITUDE_DEG, 0.0, 0.0),
        periodAngles_Gs_to_Wn_ixyz=(FLAPPING_PERIOD_S, 0.0, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(phase_offset_deg, 0.0, 0.0),
    )


def build_airplane_movement(
    airplane: ps.geometry.airplane.Airplane,
    phase_offset_deg: float = 0.0,
) -> ps.movements.airplane_movement.AirplaneMovement:
    """Create the prescribed internal flapping motion."""
    return ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            build_wing_movement(wing, phase_offset_deg=phase_offset_deg)
            for wing in airplane.wings
        ],
    )


def build_problem(
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    weight_n: float = DEFAULT_WEIGHT_N,
    initial_speed_mps: float = INITIAL_SPEED_MPS,
    max_abs_speed_mps: float = 50.0,
    max_abs_x_m: float = 200.0,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    StreamwiseFreeClampDiagnostics,
]:
    """Build the streamwise-free coupled problem and diagnostics."""
    delta_time = FLAPPING_PERIOD_S / steps_per_flap
    airplane = build_airplane(weight_n=weight_n)
    airplane_movement = build_airplane_movement(airplane)

    initial_coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
        rho=AIR_DENSITY,
        vCg__E=initial_speed_mps,
        alpha=0.0,
        beta=0.0,
        angles_E_to_BP1_izyx=(
            INITIAL_ROLL_DEG,
            INITIAL_PITCH_DEG,
            INITIAL_YAW_DEG,
        ),
        externalFX_W=0.0,
        nu=KINEMATIC_VISCOSITY,
        g_E=GRAVITY_E,
    )

    coupled_movement = ps.movements.movement.CoupledMovement(
        airplane_movement=airplane_movement,
        initial_coupled_operating_point=initial_coupled_operating_point,
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    coupled_problem = ps.problems.CoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1=inertia_from_weight(weight_n),
    )
    coupled_solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_unsteady_problem=coupled_problem
    )

    clamp_diagnostics = StreamwiseFreeClampDiagnostics(
        mujoco_model=coupled_problem.mujoco_model,
        target_position_E_m=np.zeros(3, dtype=float),
        target_angles_deg=np.array(
            [INITIAL_ROLL_DEG, INITIAL_PITCH_DEG, INITIAL_YAW_DEG], dtype=float
        ),
        initial_streamwise_speed_mps=initial_speed_mps,
        max_abs_speed_mps=max_abs_speed_mps,
        max_abs_x_m=max_abs_x_m,
    )
    clamp_diagnostics.install()
    return coupled_problem, coupled_solver, clamp_diagnostics


def compute_speed_convergence(
    times_s: np.ndarray,
    velocities_E__E: np.ndarray,
    prescribed_num_steps: int,
    steps_per_flap: int,
) -> dict[str, Any]:
    """Compute cycle-averaged streamwise-speed convergence diagnostics."""
    del times_s
    ux_history_mps = velocities_E__E[:, 0]
    start = min(max(prescribed_num_steps, 0), len(ux_history_mps))
    free_ux = ux_history_mps[start:]
    num_cycles = len(free_ux) // steps_per_flap
    if num_cycles <= 0:
        return {
            "cycle_mean_ux_mps": [],
            "converged": False,
            "converged_cycle": None,
            "converged_speed_mps": None,
            "final_5_cycle_mean_ux_mps": None,
            "final_5_cycle_slope_mps_per_cycle": None,
        }

    cycle_mean_ux_mps = np.array(
        [
            free_ux[i * steps_per_flap : (i + 1) * steps_per_flap].mean()
            for i in range(num_cycles)
        ],
        dtype=float,
    )
    final_window = cycle_mean_ux_mps[-min(5, num_cycles) :]
    final_mean = float(final_window.mean())
    tolerance = max(0.01, 0.01 * abs(final_mean))
    if len(final_window) >= 2:
        x = np.arange(len(final_window), dtype=float)
        final_slope = float(np.polyfit(x, final_window, 1)[0])
    else:
        final_slope = 0.0

    converged_cycle: int | None = None
    for cycle_index in range(num_cycles):
        remaining = cycle_mean_ux_mps[cycle_index:]
        if np.all(np.abs(remaining - final_mean) <= tolerance):
            converged_cycle = int(cycle_index + 1)
            break

    converged = (
        num_cycles >= 5 and converged_cycle is not None and abs(final_slope) < 0.005
    )
    return {
        "cycle_mean_ux_mps": cycle_mean_ux_mps.tolist(),
        "converged": bool(converged),
        "converged_cycle": converged_cycle if converged else None,
        "converged_speed_mps": final_mean if converged else None,
        "final_5_cycle_mean_ux_mps": final_mean,
        "final_5_cycle_slope_mps_per_cycle": final_slope,
        "cycle_tolerance_mps": float(tolerance),
    }


def save_streamwise_speed_convergence_plot(
    cycle_mean_ux_mps: list[float],
    save_path: Path,
) -> Path | None:
    """Save the cycle-mean streamwise speed convergence plot."""
    if not cycle_mean_ux_mps:
        return None
    cycles = np.arange(1, len(cycle_mean_ux_mps) + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cycles, cycle_mean_ux_mps, marker="o", linewidth=1.5)
    ax.set_xlabel("Free-flight flapping cycle")
    ax.set_ylabel("Cycle-mean Ux (m/s)")
    ax.set_title("Streamwise Speed Convergence")
    ax.grid(True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_force_torque_history_plot(
    times_over_period: np.ndarray,
    load_history: dict[str, np.ndarray],
    save_path: Path,
) -> Path | None:
    """Save raw and projected force/torque histories in Earth axes."""
    raw_forces_E = load_history["raw_forces_E_N"]
    raw_moments_E_Cg = load_history["raw_moments_E_Cg_Nm"]
    projected_forces_E = load_history["projected_forces_E_N"]
    num_steps = min(len(times_over_period), len(raw_forces_E), len(raw_moments_E_Cg))
    if num_steps <= 0:
        return None
    x_values = times_over_period[:num_steps]
    raw_forces_E = raw_forces_E[:num_steps]
    raw_moments_E_Cg = raw_moments_E_Cg[:num_steps]
    projected_forces_E = projected_forces_E[:num_steps]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(x_values, raw_forces_E[:, 0], label="Raw Fx")
    axes[0].plot(x_values, raw_forces_E[:, 1], label="Raw Fy")
    axes[0].plot(x_values, raw_forces_E[:, 2], label="Raw Fz")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Raw Net Loads Before Clamp Projection")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, projected_forces_E[:, 0], label="MuJoCo Fx")
    axes[1].plot(x_values, projected_forces_E[:, 1], label="MuJoCo Fy")
    axes[1].plot(x_values, projected_forces_E[:, 2], label="MuJoCo Fz")
    axes[1].set_ylabel("Force (N)")
    axes[1].set_title("Loads Passed to MuJoCo")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x_values, raw_moments_E_Cg[:, 0], label="Raw Mx")
    axes[2].plot(x_values, raw_moments_E_Cg[:, 1], label="Raw My")
    axes[2].plot(x_values, raw_moments_E_Cg[:, 2], label="Raw Mz")
    axes[2].set_ylabel("Torque (N m)")
    axes[2].set_xlabel("Time / Flapping Period")
    axes[2].set_title("Raw Net Moments Before Rotational Clamp")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("README-Scaled Flapping Streamwise-Free Loads")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_clamp_reaction_history_plot(
    times_over_period: np.ndarray,
    load_history: dict[str, np.ndarray],
    save_path: Path,
) -> Path | None:
    """Save y/z clamp-force and rotational clamp-torque histories."""
    clamp_forces_E = load_history["clamp_forces_E_N"]
    clamp_moments_E_Cg = load_history["clamp_moments_E_Cg_Nm"]
    num_steps = min(
        len(times_over_period), len(clamp_forces_E), len(clamp_moments_E_Cg)
    )
    if num_steps <= 0:
        return None
    x_values = times_over_period[:num_steps]
    clamp_forces_E = clamp_forces_E[:num_steps]
    clamp_moments_E_Cg = clamp_moments_E_Cg[:num_steps]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x_values, clamp_forces_E[:, 1], label="F clamp y")
    axes[0].plot(x_values, clamp_forces_E[:, 2], label="F clamp z")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Translational Clamp Reactions")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, clamp_moments_E_Cg[:, 0], label="M clamp roll")
    axes[1].plot(x_values, clamp_moments_E_Cg[:, 1], label="M clamp pitch")
    axes[1].plot(x_values, clamp_moments_E_Cg[:, 2], label="M clamp yaw")
    axes[1].set_ylabel("Torque (N m)")
    axes[1].set_xlabel("Time / Flapping Period")
    axes[1].set_title("Rotational Clamp Reactions")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("README-Scaled Flapping Clamp Reactions")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_history_npz(
    output_dir: Path,
    times_s: np.ndarray,
    velocities_E__E: np.ndarray,
    speeds_mps: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
    load_history: dict[str, np.ndarray],
) -> Path:
    """Save compact dense histories for post-processing."""
    save_path = output_dir / "history.npz"
    payload: dict[str, np.ndarray] = {
        "times_s": times_s,
        "velocities_E_mps": velocities_E__E,
        "speeds_mps": speeds_mps,
        "alphas_deg": alphas_deg,
        "euler_angles_deg": euler_angles_deg,
    }
    payload.update(load_history)
    np.savez_compressed(save_path, **payload)
    return save_path


def _rms(values: np.ndarray) -> list[float]:
    """Return component-wise RMS values as a JSON-friendly list."""
    if values.size == 0:
        return []
    return np.sqrt(np.mean(values**2, axis=0)).tolist()


def build_summary(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    clamp_diagnostics: StreamwiseFreeClampDiagnostics,
    steps_per_flap: int,
    prescribed_wake: bool,
    weight_n: float,
    initial_speed_mps: float,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    """Build a machine-readable summary for the completed case."""
    times_s, velocities_E__E, speeds_mps, alphas_deg, euler_angles_deg = (
        ff_utils.get_history_arrays(coupled_problem)
    )
    load_history = clamp_diagnostics.history_arrays()
    clamp_forces_E = load_history["clamp_forces_E_N"]
    clamp_moments_E_Cg = load_history["clamp_moments_E_Cg_Nm"]
    projected_forces_E = load_history["projected_forces_E_N"]
    summary: dict[str, Any] = {
        "case": "readme_scaled_flapping_streamwise_free_x",
        "wake_model": "prescribed" if prescribed_wake else "free",
        "prescribed_wake": bool(prescribed_wake),
        "solver": "coupled_unsteady_ring_vortex_lattice_method",
        "clamp_mode": "streamwise_x_and_ux_free_yz_and_all_rotations_clamped",
        "ground_effect": False,
        "airfoil": AIRFOIL_NAME,
        "airfoil_computation_note": (
            "NACA airfoils enter this ring-VLM model through generated airfoil "
            "geometry and the resampled mean camber line used for panel meshing; no "
            "airfoil polar, stall, or fitted section data are used. For naca0012 the "
            "mean camber line is flat."
        ),
        "full_span_m": FULL_SPAN_M,
        "semi_span_m": SEMI_SPAN_M,
        "root_chord_m": ROOT_CHORD_M,
        "tip_chord_m": TIP_CHORD_M,
        "mean_chord_m": MEAN_CHORD_M,
        "reference_area_m2": REFERENCE_AREA_M2,
        "tip_location_from_root_m": [
            TIP_X_FROM_ROOT_M,
            SEMI_SPAN_M,
            TIP_Z_FROM_ROOT_M,
        ],
        "wing_incidence_deg": WING_INCIDENCE_DEG,
        "tip_twist_deg": TIP_TWIST_DEG,
        "left_tip_reference_incidence_deg": LEFT_TIP_REFERENCE_INCIDENCE_DEG,
        "flapping_amplitude_deg": FLAPPING_AMPLITUDE_DEG,
        "flapping_frequency_hz": FLAPPING_FREQUENCY_HZ,
        "flapping_period_s": FLAPPING_PERIOD_S,
        "steps_per_flap": steps_per_flap,
        "delta_time_s": coupled_problem.delta_time,
        "prescribed_steps": coupled_problem.coupled_movement.prescribed_num_steps,
        "free_steps": coupled_problem.coupled_movement.free_num_steps,
        "periods_total": (
            coupled_problem.coupled_movement.prescribed_num_steps
            + coupled_problem.coupled_movement.free_num_steps
        )
        / steps_per_flap,
        "weight_n": float(weight_n),
        "wing_loading_n_m2": float(weight_n / REFERENCE_AREA_M2),
        "initial_speed_mps": float(initial_speed_mps),
        "initial_euler_izyx_deg": [
            INITIAL_ROLL_DEG,
            INITIAL_PITCH_DEG,
            INITIAL_YAW_DEG,
        ],
        "final_position_E_m": coupled_solver.stackPosition_E_E[-1].tolist(),
        "final_velocity_E_mps": velocities_E__E[-1].tolist(),
        "free_phase_mean_projected_fx_N": (
            float(np.mean(projected_forces_E[:, 0]))
            if len(projected_forces_E) > 0
            else None
        ),
        "free_phase_rms_clamp_force_yz_N": (
            _rms(clamp_forces_E[:, 1:3]) if len(clamp_forces_E) > 0 else []
        ),
        "free_phase_rms_clamp_moment_Nm": (
            _rms(clamp_moments_E_Cg) if len(clamp_moments_E_Cg) > 0 else []
        ),
    }
    summary.update(convergence)
    ff_utils.add_block_summaries(
        summary=summary,
        velocities_E__E=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        steps_per_block=steps_per_flap,
        block_name="period",
    )
    return summary


def run_case(
    output_dir: Path,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    animate: bool,
    show_wake_vortices: bool,
    show_progress: bool,
    prescribed_wake: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    dense_diagnostics_every_n_steps: int | None,
    history_save_dir: Path | None,
    weight_n: float,
    initial_speed_mps: float,
    max_abs_speed_mps: float,
    max_abs_x_m: float,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    StreamwiseFreeClampDiagnostics,
]:
    """Run the streamwise-free README-scaled flapping case and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if history_save_dir is None and (
        save_every_n_steps is not None or dense_diagnostics_every_n_steps is not None
    ):
        history_save_dir = output_dir / "streamed_history"

    coupled_problem, coupled_solver, clamp_diagnostics = build_problem(
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        weight_n=weight_n,
        initial_speed_mps=initial_speed_mps,
        max_abs_speed_mps=max_abs_speed_mps,
        max_abs_x_m=max_abs_x_m,
    )
    clamp_diagnostics.configure_load_streaming(
        history_save_dir=history_save_dir,
        save_every_n_steps=save_every_n_steps,
        dense_diagnostics_every_n_steps=dense_diagnostics_every_n_steps,
        global_step_offset=prescribed_num_steps,
        delta_time_s=coupled_problem.delta_time,
    )
    coupled_solver.run(
        prescribed_wake=prescribed_wake,
        show_progress=show_progress,
        history_stride=history_stride,
        save_every_n_steps=save_every_n_steps,
        history_save_dir=history_save_dir,
    )

    times_s, velocities_E__E, speeds_mps, alphas_deg, euler_angles_deg = (
        ff_utils.get_history_arrays(coupled_problem)
    )
    times_over_period = times_s / FLAPPING_PERIOD_S
    load_history = clamp_diagnostics.history_arrays()
    num_load_steps = len(load_history["raw_forces_E_N"])
    load_times_over_period = (
        times_over_period[-num_load_steps:] if num_load_steps > 0 else np.zeros(0)
    )
    load_history_with_time = dict(load_history)
    load_history_with_time["load_times_s"] = (
        times_s[-num_load_steps:] if num_load_steps > 0 else np.zeros(0)
    )
    convergence = compute_speed_convergence(
        times_s=times_s,
        velocities_E__E=velocities_E__E,
        prescribed_num_steps=prescribed_num_steps,
        steps_per_flap=steps_per_flap,
    )

    velocity_plot_path = ff_utils.save_velocity_history_plot(
        x_values=times_over_period,
        velocities_E__E=velocities_E__E,
        speeds_mps=speeds_mps,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        save_path=output_dir / "velocity_history.png",
        x_label="Time / Flapping Period",
        title="README-Scaled Flapping Streamwise-Free Case",
    )
    convergence_plot_path = save_streamwise_speed_convergence_plot(
        cycle_mean_ux_mps=convergence["cycle_mean_ux_mps"],
        save_path=output_dir / "streamwise_speed_convergence.png",
    )
    force_torque_plot_path = save_force_torque_history_plot(
        times_over_period=load_times_over_period,
        load_history=load_history,
        save_path=output_dir / "force_torque_history.png",
    )
    clamp_plot_path = save_clamp_reaction_history_plot(
        times_over_period=load_times_over_period,
        load_history=load_history,
        save_path=output_dir / "clamp_reaction_history.png",
    )
    history_path = save_history_npz(
        output_dir=output_dir,
        times_s=times_s,
        velocities_E__E=velocities_E__E,
        speeds_mps=speeds_mps,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        load_history=load_history_with_time,
    )

    summary = build_summary(
        coupled_problem=coupled_problem,
        coupled_solver=coupled_solver,
        clamp_diagnostics=clamp_diagnostics,
        steps_per_flap=steps_per_flap,
        prescribed_wake=prescribed_wake,
        weight_n=weight_n,
        initial_speed_mps=initial_speed_mps,
        convergence=convergence,
    )
    summary["history_npz"] = str(history_path)
    summary["velocity_history_plot"] = str(velocity_plot_path)
    if convergence_plot_path is not None:
        summary["streamwise_speed_convergence_plot"] = str(convergence_plot_path)
    if force_torque_plot_path is not None:
        summary["force_torque_history_plot"] = str(force_torque_plot_path)
    if clamp_plot_path is not None:
        summary["clamp_reaction_history_plot"] = str(clamp_plot_path)

    summary_path = ff_utils.write_json(output_dir / "summary.json", summary)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved history to: {history_path}")
    print(f"Saved velocity plot to: {velocity_plot_path}")
    if convergence_plot_path is not None:
        print(f"Saved convergence plot to: {convergence_plot_path}")
    if force_torque_plot_path is not None:
        print(f"Saved force/torque plot to: {force_torque_plot_path}")
    if clamp_plot_path is not None:
        print(f"Saved clamp plot to: {clamp_plot_path}")

    if animate:
        webp_path, mp4_path = ff_utils.save_animation_bundle(
            coupled_solver=coupled_solver,
            output_dir=output_dir,
            filename_stem="AnimateFreeFlight_readme_scaled_flapping_streamwise",
            scalar_type="induced drag",
            show_wake_vortices=show_wake_vortices,
        )
        print(f"Saved animation to: {webp_path}")
        print(f"Saved movie to: {mp4_path}")

    return coupled_problem, coupled_solver, clamp_diagnostics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a README-scaled flapping wing with y/z and attitude clamped, "
            "x and Ux free."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots, summaries, histories, and movies should be saved.",
    )
    parser.add_argument(
        "--prescribed-steps",
        type=int,
        default=DEFAULT_PRESCRIBED_STEPS,
        help="Number of prescribed steps used to build the wake before free x motion.",
    )
    parser.add_argument(
        "--free-steps",
        type=int,
        default=DEFAULT_FREE_STEPS,
        help="Number of streamwise-free coupled steps after the prescribed phase.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Temporal resolution in solver steps per flapping period.",
    )
    parser.add_argument(
        "--weight-n",
        type=float,
        default=DEFAULT_WEIGHT_N,
        help=(
            "Body weight used by MuJoCo. Default preserves the earlier branch "
            "free-flight wing loading on the scaled 1 m x 0.1 m wing."
        ),
    )
    parser.add_argument(
        "--initial-speed-mps",
        type=float,
        default=INITIAL_SPEED_MPS,
        help="Initial streamwise speed.",
    )
    parser.add_argument(
        "--max-abs-speed-mps",
        type=float,
        default=50.0,
        help="Abort if |Ux| exceeds this guard.",
    )
    parser.add_argument(
        "--max-abs-x-m",
        type=float,
        default=200.0,
        help="Abort if |x| exceeds this guard.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render and save a wake-visible movie after the solve.",
    )
    parser.add_argument(
        "--show-wake-vortices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show shed wake vortex rings in the saved animation.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the TQDM progress bar while solving.",
    )
    parser.add_argument(
        "--prescribed-wake",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a prescribed wake instead of the default real free wake.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help=(
            "Retain in-memory solver history every N steps. Use 1 for all retained "
            "steps; values >1 reduce RAM."
        ),
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=None,
        help="If set, write compressed per-step snapshots to disk every N steps.",
    )
    parser.add_argument(
        "--dense-diagnostics-every-n-steps",
        type=int,
        default=1,
        help=(
            "Write a lightweight CSV row for Ux, forces, and clamp loads every N "
            "streamwise-free steps. This is independent of heavy wake snapshots."
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
    """Run the example from the command line."""
    args = parse_args()
    run_case(
        output_dir=args.output_dir,
        prescribed_num_steps=args.prescribed_steps,
        free_num_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        animate=args.animate,
        show_wake_vortices=args.show_wake_vortices,
        show_progress=args.show_progress,
        prescribed_wake=args.prescribed_wake,
        history_stride=args.history_stride,
        save_every_n_steps=args.save_every_n_steps,
        dense_diagnostics_every_n_steps=args.dense_diagnostics_every_n_steps,
        history_save_dir=args.history_save_dir,
        weight_n=args.weight_n,
        initial_speed_mps=args.initial_speed_mps,
        max_abs_speed_mps=args.max_abs_speed_mps,
        max_abs_x_m=args.max_abs_x_m,
    )


if __name__ == "__main__":
    main()
