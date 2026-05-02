"""Study streamwise-only two-bird formation stability and energy metrics.

This example intentionally constrains each body to the theory-document slice:
streamwise translation is free, while lateral/vertical translation and all rotations
are held fixed by an idealized clamp. The removed loads are recorded as clamp reaction
diagnostics so the hidden control effort remains visible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from pterasoftware import _aerodynamics_functions, _transformations

try:
    from examples import free_flight_case_utils as ff_utils
    from examples import free_flight_flapping_forward as flap_case
    from examples import free_flight_gliding_wing as glide_case
    from examples import multibody_two_flapping_forward_inline_gap_sweep as inline_case
except ImportError:
    import free_flight_case_utils as ff_utils
    import free_flight_flapping_forward as flap_case
    import free_flight_gliding_wing as glide_case
    import multibody_two_flapping_forward_inline_gap_sweep as inline_case


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "streamwise_stability_energy"
)

AIRCRAFT_MODEL_FIXED_WING = "fixed_wing"
AIRCRAFT_MODEL_FLAPPING = "flapping"
DEFAULT_AIRCRAFT_MODEL = AIRCRAFT_MODEL_FIXED_WING

REFERENCE_PERIOD_S = glide_case.REFERENCE_PERIOD_S
DEFAULT_ANGLE_OF_ATTACK_DEG = glide_case.DEFAULT_INITIAL_ALPHA_DEG
FULL_SPAN_M = 1.0
SEMI_SPAN_M = FULL_SPAN_M / 2.0
ROOT_CHORD_M = 0.1
TIP_CHORD_M = 0.1
REFERENCE_C_REF_M = ROOT_CHORD_M
REFERENCE_AREA_M2 = FULL_SPAN_M * REFERENCE_C_REF_M
ASPECT_RATIO = FULL_SPAN_M**2 / REFERENCE_AREA_M2
NUM_CHORDWISE_PANELS = 4
NUM_SPANWISE_PANELS_PER_HALF = 6
INITIAL_COLLISION_VERTICAL_CLEARANCE_M = 1.0e-6
DEFAULT_SMOKE_X_OVER_SPAN = (0.0, 1.0, 2.0)
DEFAULT_SMOKE_Y_OVER_SPAN = (0.5, 1.0)
DEFAULT_SMOKE_Z_OVER_SPAN = (0.0, 0.25)
DEFAULT_PRODUCTION_X_OVER_SPAN = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_PRODUCTION_Y_OVER_SPAN = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_PRODUCTION_Z_OVER_SPAN = (-0.5, -0.25, 0.0, 0.25, 0.5)
DEFAULT_PRESCRIBED_PERIODS = 4.0
DEFAULT_SMOKE_TOTAL_PERIODS = 20.0
DEFAULT_PRODUCTION_TOTAL_PERIODS = 50.0
DEFAULT_STEPS_PER_REFERENCE_PERIOD = glide_case.DEFAULT_STEPS_PER_REFERENCE_PERIOD


def parse_float_tuple(values_text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of floats."""
    if not isinstance(values_text, str):
        raise TypeError("values_text must be a string.")
    tokens = [token.strip() for token in values_text.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one float value is required.")
    return tuple(float(token) for token in tokens)


def format_signed_label(prefix: str, value: float) -> str:
    """Return a stable filesystem label for a signed normalized coordinate."""
    sign = "m" if value < 0.0 else "p"
    return f"{prefix}_{sign}{abs(value):.3f}".replace(".", "p")


def run_label(x_over_span: float, y_over_span: float, z_over_span: float) -> str:
    """Return a compact label for one formation state."""
    return "_".join(
        (
            format_signed_label("xB", x_over_span),
            format_signed_label("yB", y_over_span),
            format_signed_label("zB", z_over_span),
        )
    )


def body_positions_from_paper_offsets(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    span_m: float = FULL_SPAN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert paper coordinates to body initial positions.

    The document uses X = x2 - x1, Y = y1 - y2, and Z = z1 - z2. With body 1 at the
    origin, body 2 is therefore [X, -Y, -Z].
    """
    body_1_position_E_m = np.array([0.0, 0.0, 0.0], dtype=float)
    body_2_position_E_m = span_m * np.array(
        [x_over_span, -y_over_span, -z_over_span],
        dtype=float,
    )
    return body_1_position_E_m, body_2_position_E_m


def initial_planforms_overlap(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    span_m: float = FULL_SPAN_M,
    chord_m: float = REFERENCE_C_REF_M,
    vertical_clearance_m: float = INITIAL_COLLISION_VERTICAL_CLEARANCE_M,
) -> bool:
    """Return whether the initial rectangular wing planforms physically overlap."""
    x_separation_m = abs(float(x_over_span) * span_m)
    y_separation_m = abs(float(y_over_span) * span_m)
    z_separation_m = abs(float(z_over_span) * span_m)
    streamwise_overlap = x_separation_m < chord_m
    spanwise_overlap = y_separation_m < span_m
    vertically_coincident = z_separation_m < vertical_clearance_m
    return streamwise_overlap and spanwise_overlap and vertically_coincident


def initial_condition_is_collision_free(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    span_m: float = FULL_SPAN_M,
    chord_m: float = REFERENCE_C_REF_M,
    vertical_clearance_m: float = INITIAL_COLLISION_VERTICAL_CLEARANCE_M,
) -> bool:
    """Return whether a two-wing initial condition is safe to launch."""
    return not initial_planforms_overlap(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        span_m=span_m,
        chord_m=chord_m,
        vertical_clearance_m=vertical_clearance_m,
    )


def _quat_from_izyx_angles_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Return a MuJoCo-compatible wxyz quaternion for intrinsic z-y-x angles."""
    clamped_T_pas_E_to_BP = _transformations.generate_rot_T(
        angles=angles_deg,
        passive=True,
        intrinsic=True,
        order="zyx",
    )
    clamped_R_pas_BP_to_E = clamped_T_pas_E_to_BP[:3, :3].T
    return _transformations.R_to_quat_wxyz(clamped_R_pas_BP_to_E)


@dataclass
class StreamwiseClampDiagnostics:
    """Patch a multibody MuJoCo model to enforce streamwise-only motion."""

    mujoco_model: object
    target_positions_E_m: np.ndarray
    target_angles_deg: np.ndarray
    span_m: float = FULL_SPAN_M
    max_abs_x_over_span: float = 20.0
    max_abs_speed_mps: float = 50.0

    def __post_init__(self) -> None:
        self.target_positions_E_m = np.asarray(
            self.target_positions_E_m, dtype=float
        ).copy()
        self.target_angles_deg = np.asarray(self.target_angles_deg, dtype=float).copy()
        if self.target_positions_E_m.shape != (self.mujoco_model.num_bodies, 3):
            raise ValueError(
                "target_positions_E_m must have shape "
                f"({self.mujoco_model.num_bodies}, 3)."
            )
        if self.target_angles_deg.shape != (3,):
            raise ValueError("target_angles_deg must have shape (3,).")
        if self.span_m <= 0.0:
            raise ValueError("span_m must be positive.")
        if self.max_abs_x_over_span <= 0.0:
            raise ValueError("max_abs_x_over_span must be positive.")
        if self.max_abs_speed_mps <= 0.0:
            raise ValueError("max_abs_speed_mps must be positive.")

        self.raw_forces_E: list[np.ndarray] = []
        self.raw_moments_E_Cg: list[np.ndarray] = []
        self.projected_forces_E: list[np.ndarray] = []
        self.projected_moments_E_Cg: list[np.ndarray] = []
        self.preclamp_yz_velocities_E: list[np.ndarray] = []
        self.preclamp_angular_rates_rad_s: list[np.ndarray] = []

    def install(self) -> None:
        """Install load projection and post-step state clamping."""
        self.enforce_state(record=False)
        original_apply_loads = self.mujoco_model.apply_loads
        original_step = self.mujoco_model.step

        def apply_streamwise_loads(
            forces_E: np.ndarray,
            moments_E_Cg: np.ndarray,
        ) -> None:
            raw_forces_E = np.asarray(forces_E, dtype=float).copy()
            raw_moments_E_Cg = np.asarray(moments_E_Cg, dtype=float).copy()
            projected_forces_E = raw_forces_E.copy()
            projected_moments_E_Cg = np.zeros_like(raw_moments_E_Cg)
            projected_forces_E[:, 1:3] = 0.0

            self.raw_forces_E.append(raw_forces_E)
            self.raw_moments_E_Cg.append(raw_moments_E_Cg)
            self.projected_forces_E.append(projected_forces_E.copy())
            self.projected_moments_E_Cg.append(projected_moments_E_Cg.copy())
            original_apply_loads(projected_forces_E, projected_moments_E_Cg)

        def step_streamwise_only() -> None:
            original_step()
            self.enforce_state(record=True)

        self.mujoco_model.apply_loads = apply_streamwise_loads
        self.mujoco_model.step = step_streamwise_only

    def enforce_state(self, record: bool) -> None:
        """Clamp non-streamwise state components and optionally record pre-clamp rates."""
        yz_velocities_E = np.zeros((self.mujoco_model.num_bodies, 2), dtype=float)
        angular_rates_rad_s = np.zeros((self.mujoco_model.num_bodies, 3), dtype=float)
        target_quat_wxyz = _quat_from_izyx_angles_deg(self.target_angles_deg)

        for body_index, qpos_adr in enumerate(self.mujoco_model.body_qposadrs):
            qvel_adr = int(self.mujoco_model.body_qveladrs[body_index])
            yz_velocities_E[body_index] = self.mujoco_model.data.qvel[
                qvel_adr + 1 : qvel_adr + 3
            ]
            angular_rates_rad_s[body_index] = self.mujoco_model.data.qvel[
                qvel_adr + 3 : qvel_adr + 6
            ]

            self.mujoco_model.data.qpos[qpos_adr + 1] = self.target_positions_E_m[
                body_index, 1
            ]
            self.mujoco_model.data.qpos[qpos_adr + 2] = self.target_positions_E_m[
                body_index, 2
            ]
            self.mujoco_model.data.qpos[qpos_adr + 3 : qpos_adr + 7] = target_quat_wxyz
            self.mujoco_model.data.qvel[qvel_adr + 1 : qvel_adr + 3] = 0.0
            self.mujoco_model.data.qvel[qvel_adr + 3 : qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.mujoco_model.model, self.mujoco_model.data)
        if record:
            self.preclamp_yz_velocities_E.append(yz_velocities_E)
            self.preclamp_angular_rates_rad_s.append(angular_rates_rad_s)
            self.raise_if_diverged()

    def raise_if_diverged(self) -> None:
        """Stop a case early if x-only dynamics has clearly diverged."""
        positions_x_m = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        speeds_x_mps = np.zeros(self.mujoco_model.num_bodies, dtype=float)
        for body_index, qpos_adr in enumerate(self.mujoco_model.body_qposadrs):
            qvel_adr = int(self.mujoco_model.body_qveladrs[body_index])
            positions_x_m[body_index] = self.mujoco_model.data.qpos[qpos_adr]
            speeds_x_mps[body_index] = self.mujoco_model.data.qvel[qvel_adr]

        relative_x_over_span = (positions_x_m[1] - positions_x_m[0]) / self.span_m
        if not np.all(np.isfinite(speeds_x_mps)) or not np.isfinite(
            relative_x_over_span
        ):
            raise RuntimeError("Streamwise-only case diverged with non-finite state.")
        if np.max(np.abs(speeds_x_mps)) > self.max_abs_speed_mps:
            raise RuntimeError(
                "Streamwise-only case exceeded speed guard: "
                f"max |Ux|={np.max(np.abs(speeds_x_mps)):.3g} m/s."
            )
        if abs(relative_x_over_span) > self.max_abs_x_over_span:
            raise RuntimeError(
                "Streamwise-only case exceeded spacing guard: "
                f"|X/B|={abs(relative_x_over_span):.3g}."
            )

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return recorded clamp/load histories as arrays."""
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
        preclamp_yz_velocities_E = _stack_history(
            self.preclamp_yz_velocities_E, self.mujoco_model.num_bodies, 2
        )
        preclamp_angular_rates_rad_s = _stack_history(
            self.preclamp_angular_rates_rad_s, self.mujoco_model.num_bodies, 3
        )

        num_steps = min(
            len(raw_forces_E),
            len(raw_moments_E_Cg),
            len(preclamp_yz_velocities_E),
            len(preclamp_angular_rates_rad_s),
        )
        raw_forces_E = raw_forces_E[:num_steps]
        raw_moments_E_Cg = raw_moments_E_Cg[:num_steps]
        projected_forces_E = projected_forces_E[:num_steps]
        projected_moments_E_Cg = projected_moments_E_Cg[:num_steps]
        preclamp_yz_velocities_E = preclamp_yz_velocities_E[:num_steps]
        preclamp_angular_rates_rad_s = preclamp_angular_rates_rad_s[:num_steps]

        clamp_forces_yz_E = -raw_forces_E[:, :, 1:3]
        clamp_moments_E_Cg = -raw_moments_E_Cg
        clamp_power_proxy_W = np.sum(
            np.abs(clamp_forces_yz_E * preclamp_yz_velocities_E), axis=2
        ) + np.sum(
            np.abs(clamp_moments_E_Cg * preclamp_angular_rates_rad_s),
            axis=2,
        )

        return {
            "raw_forces_E_N": raw_forces_E,
            "raw_moments_E_Cg_Nm": raw_moments_E_Cg,
            "projected_forces_E_N": projected_forces_E,
            "projected_moments_E_Cg_Nm": projected_moments_E_Cg,
            "preclamp_yz_velocities_E_mps": preclamp_yz_velocities_E,
            "preclamp_angular_rates_rad_s": preclamp_angular_rates_rad_s,
            "clamp_forces_yz_E_N": clamp_forces_yz_E,
            "clamp_moments_E_Cg_Nm": clamp_moments_E_Cg,
            "clamp_power_proxy_W": clamp_power_proxy_W,
        }


def _stack_history(
    history: list[np.ndarray], num_bodies: int, width: int
) -> np.ndarray:
    """Stack a possibly empty per-step history list."""
    if not history:
        return np.zeros((0, num_bodies, width), dtype=float)
    return np.stack(history, axis=0)


def install_streamwise_only_projection(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
    target_positions_E_m: tuple[np.ndarray, np.ndarray],
    target_angles_deg: tuple[float, float, float],
    max_abs_x_over_span: float = 20.0,
    max_abs_speed_mps: float = 50.0,
) -> StreamwiseClampDiagnostics:
    """Install streamwise-only dynamics and return its diagnostics recorder."""
    diagnostics = StreamwiseClampDiagnostics(
        mujoco_model=coupled_problem.mujoco_model,
        target_positions_E_m=np.vstack(target_positions_E_m),
        target_angles_deg=np.array(target_angles_deg, dtype=float),
        span_m=FULL_SPAN_M,
        max_abs_x_over_span=max_abs_x_over_span,
        max_abs_speed_mps=max_abs_speed_mps,
    )
    diagnostics.install()
    return diagnostics


def multibody_rigid_body_drag_model(
    body_index: int,
    coupled_operating_point: ps.operating_point.CoupledOperatingPoint,
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the tuned single-body forward-flight body-drag surrogate."""
    return inline_case.multibody_rigid_body_drag_model(
        body_index=body_index,
        coupled_operating_point=coupled_operating_point,
        airplane=airplane,
    )


def build_rectangular_fixed_wing_airplane(
    name: str = "Rectangular Fixed Wing",
) -> ps.geometry.airplane.Airplane:
    """Build the clean rectangular fixed wing used for theory validation."""
    wing = ps.geometry.wing.Wing(
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=TIP_CHORD_M,
                num_spanwise_panels=NUM_SPANWISE_PANELS_PER_HALF,
                spanwise_spacing="uniform",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=ROOT_CHORD_M,
                Lp_Wcsp_Lpp=(0.0, SEMI_SPAN_M, 0.0),
                num_spanwise_panels=NUM_SPANWISE_PANELS_PER_HALF,
                spanwise_spacing="uniform",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=TIP_CHORD_M,
                Lp_Wcsp_Lpp=(0.0, SEMI_SPAN_M, 0.0),
                num_spanwise_panels=None,
            ),
        ],
        name=f"{name} Wing",
        Ler_Gs_Cgs=(0.0, -SEMI_SPAN_M, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=False,
        mirror_only=False,
        symmetryNormal_G=None,
        symmetryPoint_G_Cg=None,
        num_chordwise_panels=NUM_CHORDWISE_PANELS,
        chordwise_spacing="uniform",
    )
    return ps.geometry.airplane.Airplane(
        wings=[wing],
        name=name,
        weight=glide_case.TOTAL_AIRCRAFT_WEIGHT_N,
    )


def build_fixed_wing_airplanes_and_movements() -> tuple[
    ps.geometry.airplane.Airplane,
    ps.geometry.airplane.Airplane,
    ps.movements.airplane_movement.AirplaneMovement,
    ps.movements.airplane_movement.AirplaneMovement,
]:
    """Build two identical static fixed-wing aircraft and movements."""
    airplane_1 = build_rectangular_fixed_wing_airplane(name="Rectangular Fixed Wing 1")
    airplane_2 = build_rectangular_fixed_wing_airplane(name="Rectangular Fixed Wing 2")
    airplane_movement_1 = glide_case.build_airplane_movement(airplane_1)
    airplane_movement_2 = glide_case.build_airplane_movement(airplane_2)
    return airplane_1, airplane_2, airplane_movement_1, airplane_movement_2


def build_flapping_airplanes_and_movements() -> tuple[
    ps.geometry.airplane.Airplane,
    ps.geometry.airplane.Airplane,
    ps.movements.airplane_movement.AirplaneMovement,
    ps.movements.airplane_movement.AirplaneMovement,
]:
    """Build two identical flapping aircraft and movements."""
    airplane_1 = flap_case.build_airplane()
    airplane_2 = flap_case.build_airplane()
    airplane_movement_1 = flap_case.build_airplane_movement(airplane_1)
    airplane_movement_2 = flap_case.build_airplane_movement(airplane_2)
    return airplane_1, airplane_2, airplane_movement_1, airplane_movement_2


def get_aircraft_setup(
    aircraft_model: str,
) -> tuple[
    ps.geometry.airplane.Airplane,
    ps.geometry.airplane.Airplane,
    ps.movements.airplane_movement.AirplaneMovement,
    ps.movements.airplane_movement.AirplaneMovement,
    np.ndarray,
    Any,
]:
    """Return geometry, prescribed motion, inertia, and extra loads for a model."""
    if aircraft_model == AIRCRAFT_MODEL_FIXED_WING:
        (
            airplane_1,
            airplane_2,
            airplane_movement_1,
            airplane_movement_2,
        ) = build_fixed_wing_airplanes_and_movements()
        return (
            airplane_1,
            airplane_2,
            airplane_movement_1,
            airplane_movement_2,
            glide_case.INERTIA_BP1_CGP1,
            None,
        )
    if aircraft_model == AIRCRAFT_MODEL_FLAPPING:
        (
            airplane_1,
            airplane_2,
            airplane_movement_1,
            airplane_movement_2,
        ) = build_flapping_airplanes_and_movements()
        return (
            airplane_1,
            airplane_2,
            airplane_movement_1,
            airplane_movement_2,
            flap_case.INERTIA_BP1_CGP1,
            multibody_rigid_body_drag_model,
        )
    raise ValueError(
        f'aircraft_model must be "{AIRCRAFT_MODEL_FIXED_WING}" or '
        f'"{AIRCRAFT_MODEL_FLAPPING}".'
    )


def build_problem(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    aircraft_model: str = DEFAULT_AIRCRAFT_MODEL,
    angle_of_attack_deg: float = DEFAULT_ANGLE_OF_ATTACK_DEG,
    max_abs_x_over_span: float = 20.0,
    max_abs_speed_mps: float = 50.0,
) -> tuple[
    ps.problems.MultiBodyCoupledUnsteadyProblem,
    ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
    StreamwiseClampDiagnostics,
]:
    """Build one two-body streamwise-only fixed-wing or flapping formation problem."""
    delta_time = REFERENCE_PERIOD_S / steps_per_flap
    (
        airplane_1,
        airplane_2,
        airplane_movement_1,
        airplane_movement_2,
        inertia_BP1_CgP1,
        external_forces_fn,
    ) = get_aircraft_setup(aircraft_model)

    operating_point_kwargs = dict(
        rho=glide_case.AIR_DENSITY,
        vCg__E=glide_case.DEFAULT_INITIAL_SPEED_MPS,
        alpha=angle_of_attack_deg,
        beta=0.0,
        angles_E_to_BP1_izyx=(
            0.0,
            angle_of_attack_deg,
            0.0,
        ),
        externalFX_W=0.0,
        nu=glide_case.KINEMATIC_VISCOSITY,
        g_E=glide_case.GRAVITY_E,
    )
    coupled_operating_point_1 = ps.operating_point.CoupledOperatingPoint(
        **operating_point_kwargs
    )
    coupled_operating_point_2 = ps.operating_point.CoupledOperatingPoint(
        **operating_point_kwargs
    )
    initial_positions_E_E = body_positions_from_paper_offsets(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
    )

    coupled_movement = ps.movements.movement.MultiBodyCoupledMovement(
        airplane_movements=[airplane_movement_1, airplane_movement_2],
        initial_coupled_operating_points=[
            coupled_operating_point_1,
            coupled_operating_point_2,
        ],
        initial_positions_E_E=initial_positions_E_E,
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )
    coupled_problem = ps.problems.MultiBodyCoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1s=[inertia_BP1_CgP1, inertia_BP1_CgP1],
        external_forces_fn=external_forces_fn,
    )
    coupled_solver = ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_problem
    )
    clamp_diagnostics = install_streamwise_only_projection(
        coupled_problem=coupled_problem,
        target_positions_E_m=initial_positions_E_E,
        target_angles_deg=(0.0, angle_of_attack_deg, 0.0),
        max_abs_x_over_span=max_abs_x_over_span,
        max_abs_speed_mps=max_abs_speed_mps,
    )
    return coupled_problem, coupled_solver, clamp_diagnostics


def get_history_arrays(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the existing multibody history extractor."""
    return inline_case.get_multibody_history_arrays(coupled_problem)


def aerodynamic_forces_E_history(
    coupled_problem: ps.problems.MultiBodyCoupledUnsteadyProblem,
) -> np.ndarray:
    """Transform recorded aerodynamic forces from wind axes to Earth axes."""
    forces_W = np.asarray(coupled_problem.forces_W, dtype=float)
    num_steps = len(forces_W)
    forces_E = np.zeros_like(forces_W)
    coupled_operating_points_by_step = (
        coupled_problem.coupled_movement.coupled_operating_points
    )
    for step in range(num_steps):
        for body_index, coupled_operating_point in enumerate(
            coupled_operating_points_by_step[step]
        ):
            forces_E[step, body_index] = _transformations.apply_T_to_vectors(
                coupled_operating_point.T_pas_W_CgP1_to_E_CgP1,
                forces_W[step, body_index],
                has_point=False,
            )
    return forces_E


def _empty_velocity(points_E: np.ndarray) -> np.ndarray:
    """Return a zero velocity array matching a point array."""
    return np.zeros_like(points_E, dtype=float)


def _collect_receiver_points_and_weights(
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect receiver collocation points and lift-like weights."""
    points: list[np.ndarray] = []
    weights: list[float] = []
    areas: list[float] = []
    for wing in airplane.wings:
        panels = wing.panels
        if panels is None:
            continue
        for panel in np.ravel(panels):
            points.append(np.asarray(panel.Cpp_GP1_CgP1, dtype=float))
            areas.append(float(panel.area))
            panel_force = np.asarray(
                getattr(panel, "forces_GP1", np.zeros(3, dtype=float)),
                dtype=float,
            )
            weights.append(abs(float(panel_force[2])))

    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

    points_array = np.vstack(points)
    weights_array = np.asarray(weights, dtype=float)
    if np.sum(weights_array) <= 1e-14:
        weights_array = np.asarray(areas, dtype=float)
    return points_array, weights_array


def _collect_ring_arrays(
    rings: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect ring-vortex vertices, strengths, and ages for Biot-Savart calls."""
    if not rings:
        empty_points = np.zeros((0, 3), dtype=float)
        return (
            empty_points,
            empty_points,
            empty_points,
            empty_points,
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )

    stack_brrvp = np.vstack([ring.Brrvp_GP1_CgP1 for ring in rings])
    stack_frrvp = np.vstack([ring.Frrvp_GP1_CgP1 for ring in rings])
    stack_flrvp = np.vstack([ring.Flrvp_GP1_CgP1 for ring in rings])
    stack_blrvp = np.vstack([ring.Blrvp_GP1_CgP1 for ring in rings])
    strengths = np.asarray([ring.strength for ring in rings], dtype=float)
    ages = np.asarray([getattr(ring, "age", 0.0) for ring in rings], dtype=float)
    return stack_brrvp, stack_frrvp, stack_flrvp, stack_blrvp, strengths, ages


def _collect_source_rings(
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[list[Any], list[Any]]:
    """Collect bound and wake rings for one source body."""
    bound_rings: list[Any] = []
    wake_rings: list[Any] = []
    for wing in airplane.wings:
        panels = wing.panels
        if panels is not None:
            for panel in np.ravel(panels):
                if panel.ring_vortex is not None:
                    bound_rings.append(panel.ring_vortex)
        if wing.wake_ring_vortices is not None:
            for wake_ring in np.ravel(wing.wake_ring_vortices):
                if wake_ring is not None:
                    wake_rings.append(wake_ring)
    return bound_rings, wake_rings


def _induced_velocity_from_rings(
    points_E: np.ndarray,
    rings: list[Any],
    nu: float,
    use_ages: bool,
) -> np.ndarray:
    """Compute induced velocity at points from a source ring-vortex collection."""
    if len(points_E) == 0 or not rings:
        return _empty_velocity(points_E)

    (
        stack_brrvp,
        stack_frrvp,
        stack_flrvp,
        stack_blrvp,
        strengths,
        ages,
    ) = _collect_ring_arrays(rings)
    return _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
        stackP_GP1_CgP1=points_E,
        stackBrrvp_GP1_CgP1=stack_brrvp,
        stackFrrvp_GP1_CgP1=stack_frrvp,
        stackFlrvp_GP1_CgP1=stack_flrvp,
        stackBlrvp_GP1_CgP1=stack_blrvp,
        strengths=strengths,
        r_c0s=np.zeros(len(strengths), dtype=float),
        singularity_counts=np.zeros(4, dtype=np.int64),
        ages=ages if use_ages else None,
        nu=nu,
    )


def compute_source_separated_wbar_history(
    coupled_solver: ps.multibody_coupled_unsteady_ring_vortex_lattice_method.MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver,
) -> dict[str, np.ndarray]:
    """Approximate lift-weighted Wbar from source-separated bound+wake rings.

    The output convention is ``wbar_source_to_receiver_mps[step, source, receiver]``.
    Diagonal entries are NaN because a body is not treated as a source to itself.
    """
    num_steps = len(coupled_solver.multi_body_coupled_steady_problems)
    wbar = np.full(
        (num_steps, coupled_solver.num_bodies, coupled_solver.num_bodies),
        np.nan,
        dtype=float,
    )
    delta_power = np.full_like(wbar, np.nan)

    for step, problem in enumerate(coupled_solver.multi_body_coupled_steady_problems):
        if problem is None:
            continue
        for receiver_index, receiver_airplane in enumerate(problem.airplanes):
            points_E, weights = _collect_receiver_points_and_weights(receiver_airplane)
            if len(points_E) == 0 or np.sum(weights) <= 1e-14:
                continue
            for source_index, source_airplane in enumerate(problem.airplanes):
                if source_index == receiver_index:
                    continue
                bound_rings, wake_rings = _collect_source_rings(source_airplane)
                source_velocity_E = _induced_velocity_from_rings(
                    points_E=points_E,
                    rings=bound_rings,
                    nu=coupled_solver._shared_nu,
                    use_ages=False,
                ) + _induced_velocity_from_rings(
                    points_E=points_E,
                    rings=wake_rings,
                    nu=coupled_solver._shared_nu,
                    use_ages=True,
                )
                wbar_value = float(np.average(source_velocity_E[:, 2], weights=weights))
                wbar[step, source_index, receiver_index] = wbar_value
                delta_power[step, source_index, receiver_index] = (
                    -receiver_airplane.weight * wbar_value
                )

    return {
        "wbar_source_to_receiver_mps": wbar,
        "delta_power_wbar_source_to_receiver_W": delta_power,
    }


def clamp_statistics(
    values: np.ndarray, component_names: tuple[str, ...]
) -> dict[str, Any]:
    """Return mean, RMS, and max-absolute statistics for clamp diagnostics."""
    stats: dict[str, Any] = {}
    if values.size == 0:
        for body_index in range(2):
            for component_name in component_names:
                prefix = f"body_{body_index}_{component_name}"
                stats[f"{prefix}_mean"] = 0.0
                stats[f"{prefix}_rms"] = 0.0
                stats[f"{prefix}_max_abs"] = 0.0
        return stats

    for body_index in range(values.shape[1]):
        for component_index, component_name in enumerate(component_names):
            component_values = values[:, body_index, component_index]
            prefix = f"body_{body_index}_{component_name}"
            stats[f"{prefix}_mean"] = float(np.mean(component_values))
            stats[f"{prefix}_rms"] = float(np.sqrt(np.mean(component_values**2)))
            stats[f"{prefix}_max_abs"] = float(np.max(np.abs(component_values)))
    return stats


def period_average(
    values: np.ndarray,
    steps_per_flap: int,
    num_periods: int = 1,
) -> np.ndarray:
    """Average the final integer number of reference periods."""
    block_size = min(len(values), max(1, steps_per_flap * num_periods))
    return np.mean(values[-block_size:], axis=0)


def finite_array_to_jsonable(values: np.ndarray) -> Any:
    """Convert a numeric array to nested JSON values, replacing NaN/Inf with null."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        scalar = float(values)
        return None if not np.isfinite(scalar) else scalar
    return [finite_array_to_jsonable(child) for child in values]


def compute_run_metrics(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    prescribed_wake: bool,
    run_status: str,
    error_message: str | None,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    euler_angles_deg: np.ndarray,
    aero_forces_E: np.ndarray,
    clamp_arrays: dict[str, np.ndarray],
    baseline_power_W: np.ndarray | None = None,
    wbar_arrays: dict[str, np.ndarray] | None = None,
    aircraft_model: str = DEFAULT_AIRCRAFT_MODEL,
    angle_of_attack_deg: float = DEFAULT_ANGLE_OF_ATTACK_DEG,
    max_abs_x_over_span: float = 20.0,
    max_abs_speed_mps: float = 50.0,
) -> dict[str, Any]:
    """Build scalar diagnostics for one streamwise formation run."""
    n_force = min(len(aero_forces_E), len(clamp_arrays["raw_forces_E_N"]))
    n = min(n_force, len(times_s) - 1, len(velocities_E__E) - 1)
    if n <= 0:
        n = min(len(times_s), len(velocities_E__E))

    positions_sample = positions_E_E[:n]
    velocities_sample = velocities_E__E[:n]
    aero_forces_sample_E = aero_forces_E[:n]
    raw_forces_sample_E = clamp_arrays["raw_forces_E_N"][:n]
    clamp_forces_yz_E = clamp_arrays["clamp_forces_yz_E_N"][:n]
    clamp_moments_E_Cg = clamp_arrays["clamp_moments_E_Cg_Nm"][:n]
    clamp_power_proxy_W = clamp_arrays["clamp_power_proxy_W"][:n]

    x_history_m = positions_sample[:, 1, 0] - positions_sample[:, 0, 0]
    y_history_m = positions_sample[:, 0, 1] - positions_sample[:, 1, 1]
    z_history_m = positions_sample[:, 0, 2] - positions_sample[:, 1, 2]
    d_x_dt_mps = velocities_sample[:, 1, 0] - velocities_sample[:, 0, 0]

    aero_power_W = -aero_forces_sample_E[:, :, 0] * velocities_sample[:, :, 0]
    applied_power_W = -raw_forces_sample_E[:, :, 0] * velocities_sample[:, :, 0]
    final_period_aero_power_W = period_average(aero_power_W, steps_per_flap)
    final_period_applied_power_W = period_average(applied_power_W, steps_per_flap)

    summary: dict[str, Any] = {
        "case": f"streamwise_stability_energy_two_{aircraft_model}_bodies",
        "aircraft_model": aircraft_model,
        "wake_model": "free" if not prescribed_wake else "prescribed",
        "prescribed_wake": bool(prescribed_wake),
        "constraint_mode": "streamwise_x_free_yz_and_attitude_clamped",
        "max_abs_x_over_span_guard": max_abs_x_over_span,
        "max_abs_speed_mps_guard": max_abs_speed_mps,
        "x_over_span_initial": x_over_span,
        "y_over_span_prescribed": y_over_span,
        "z_over_span_prescribed": z_over_span,
        "span_m": FULL_SPAN_M,
        "semispan_m": SEMI_SPAN_M,
        "root_chord_m": ROOT_CHORD_M,
        "tip_chord_m": TIP_CHORD_M,
        "chord_ref_m": REFERENCE_C_REF_M,
        "reference_area_m2": REFERENCE_AREA_M2,
        "aspect_ratio": ASPECT_RATIO,
        "num_chordwise_panels": NUM_CHORDWISE_PANELS,
        "num_spanwise_panels_total": 2 * NUM_SPANWISE_PANELS_PER_HALF,
        "initial_collision_free": initial_condition_is_collision_free(
            x_over_span=x_over_span,
            y_over_span=y_over_span,
            z_over_span=z_over_span,
        ),
        "body_1_initial_position_E_m": positions_E_E[0, 0].tolist(),
        "body_2_initial_position_E_m": positions_E_E[0, 1].tolist(),
        "fixed_roll_deg": 0.0,
        "angle_of_attack_deg": angle_of_attack_deg,
        "initial_alpha_deg": angle_of_attack_deg,
        "fixed_pitch_deg": angle_of_attack_deg,
        "fixed_pitch_matches_angle_of_attack": True,
        "fixed_yaw_deg": 0.0,
        "reference_period_s": REFERENCE_PERIOD_S,
        "steps_per_flap": steps_per_flap,
        "steps_per_reference_period": steps_per_flap,
        "delta_time_s": REFERENCE_PERIOD_S / steps_per_flap,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "periods_total_requested": (prescribed_num_steps + free_num_steps)
        / steps_per_flap,
        "periods_total_completed": float(times_s[-1] / REFERENCE_PERIOD_S),
        "run_status": run_status,
        "error_message": error_message,
        "final_x_over_span": float(x_history_m[-1] / FULL_SPAN_M),
        "final_y_over_span": float(y_history_m[-1] / FULL_SPAN_M),
        "final_z_over_span": float(z_history_m[-1] / FULL_SPAN_M),
        "last_period_mean_x_over_span": float(
            period_average(x_history_m / FULL_SPAN_M, steps_per_flap)
        ),
        "last_period_mean_dXdt_mps": float(period_average(d_x_dt_mps, steps_per_flap)),
        "final_body_0_Ux_mps": float(velocities_sample[-1, 0, 0]),
        "final_body_1_Ux_mps": float(velocities_sample[-1, 1, 0]),
        "last_period_mean_Ux_mps": period_average(
            velocities_sample[:, :, 0], steps_per_flap
        ).tolist(),
        "last_period_mean_aero_streamwise_power_W": final_period_aero_power_W.tolist(),
        "last_period_mean_applied_streamwise_power_W": (
            final_period_applied_power_W.tolist()
        ),
        "last_period_mean_pair_applied_streamwise_power_W": float(
            np.mean(final_period_applied_power_W)
        ),
        "last_period_mean_clamp_power_proxy_W": period_average(
            clamp_power_proxy_W, steps_per_flap
        ).tolist(),
        "max_yz_drift_m": float(
            np.max(
                np.abs(
                    np.stack(
                        (
                            y_history_m - y_history_m[0],
                            z_history_m - z_history_m[0],
                        ),
                        axis=1,
                    )
                )
            )
        ),
        "max_euler_deviation_deg": float(
            np.max(
                np.abs(
                    euler_angles_deg
                    - np.array([0.0, angle_of_attack_deg, 0.0])[None, None, :]
                )
            )
        ),
        "stage_1_energy_metric": (
            "direct streamwise force-power from recorded aerodynamic/applied loads"
        ),
        "stage_2_wbar_status": "computed" if wbar_arrays is not None else "disabled",
    }
    summary["clamp_force_yz_stats_N"] = clamp_statistics(
        clamp_forces_yz_E,
        ("Fy", "Fz"),
    )
    summary["clamp_torque_stats_Nm"] = clamp_statistics(
        clamp_moments_E_Cg,
        ("Mx", "My", "Mz"),
    )

    if baseline_power_W is not None:
        delta_power_W = final_period_applied_power_W - baseline_power_W
        summary["baseline_last_period_mean_applied_power_W"] = baseline_power_W.tolist()
        summary["last_period_delta_power_vs_baseline_W"] = delta_power_W.tolist()
        summary["last_period_pair_mean_delta_power_vs_baseline_W"] = float(
            np.mean(delta_power_W)
        )

    if wbar_arrays is not None:
        wbar = wbar_arrays["wbar_source_to_receiver_mps"][:n]
        delta_power_wbar = wbar_arrays["delta_power_wbar_source_to_receiver_W"][:n]
        summary["last_period_mean_wbar_source_to_receiver_mps"] = period_average(
            wbar,
            steps_per_flap,
        )
        summary["last_period_mean_delta_power_wbar_source_to_receiver_W"] = (
            period_average(delta_power_wbar, steps_per_flap)
        )
        summary["last_period_mean_wbar_source_to_receiver_mps"] = (
            finite_array_to_jsonable(
                summary["last_period_mean_wbar_source_to_receiver_mps"]
            )
        )
        summary["last_period_mean_delta_power_wbar_source_to_receiver_W"] = (
            finite_array_to_jsonable(
                summary["last_period_mean_delta_power_wbar_source_to_receiver_W"]
            )
        )
        summary["wbar_convention"] = (
            "wbar_source_to_receiver[step, source_body_index, receiver_body_index]"
        )

    return summary


def save_history_npz(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    alphas_deg: np.ndarray,
    euler_angles_deg: np.ndarray,
    aero_forces_E: np.ndarray,
    clamp_arrays: dict[str, np.ndarray],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    """Persist compact time-history data for later post-processing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "history.npz"
    arrays_to_save = dict(clamp_arrays)
    if extra_arrays is not None:
        arrays_to_save.update(extra_arrays)
    np.savez_compressed(
        save_path,
        times_s=times_s,
        positions_E_E_m=positions_E_E,
        velocities_E_mps=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        aero_forces_E_N=aero_forces_E,
        **arrays_to_save,
    )
    return save_path


def save_separation_velocity_plot(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
) -> Path:
    """Save streamwise separation and velocity histories."""
    n = min(len(times_s), len(positions_E_E), len(velocities_E__E))
    x_periods = times_s[:n] / REFERENCE_PERIOD_S
    x_over_span = (positions_E_E[:n, 1, 0] - positions_E_E[:n, 0, 0]) / FULL_SPAN_M
    y_over_span = (positions_E_E[:n, 0, 1] - positions_E_E[:n, 1, 1]) / FULL_SPAN_M
    z_over_span = (positions_E_E[:n, 0, 2] - positions_E_E[:n, 1, 2]) / FULL_SPAN_M
    d_x_dt = velocities_E__E[:n, 1, 0] - velocities_E__E[:n, 0, 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x_periods, x_over_span, color="black", label="X/B")
    axes[0].plot(x_periods, y_over_span, label="Y/B")
    axes[0].plot(x_periods, z_over_span, label="Z/B")
    axes[0].set_ylabel("Relative Position")
    axes[0].set_title("Paper-Coordinate Formation Offsets")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_periods, velocities_E__E[:n, 0, 0], label="Body 1 Ux")
    axes[1].plot(x_periods, velocities_E__E[:n, 1, 0], label="Body 2 Ux")
    axes[1].set_ylabel("Ux (m/s)")
    axes[1].set_title("Streamwise Speeds")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x_periods, d_x_dt, color="tab:red")
    axes[2].set_ylabel("dX/dt (m/s)")
    axes[2].set_xlabel("Time / Flapping Period")
    axes[2].set_title("Streamwise Spacing Rate")
    axes[2].grid(True)

    fig.tight_layout()
    save_path = output_dir / "streamwise_separation_velocity.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_power_plot(
    output_dir: Path,
    times_s: np.ndarray,
    velocities_E__E: np.ndarray,
    aero_forces_E: np.ndarray,
    raw_forces_E: np.ndarray,
) -> Path:
    """Save aerodynamic and total-applied streamwise power histories."""
    n = min(len(times_s), len(velocities_E__E), len(aero_forces_E), len(raw_forces_E))
    x_periods = times_s[:n] / REFERENCE_PERIOD_S
    aero_power_W = -aero_forces_E[:n, :, 0] * velocities_E__E[:n, :, 0]
    applied_power_W = -raw_forces_E[:n, :, 0] * velocities_E__E[:n, :, 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for body_index in range(2):
        axes[0].plot(x_periods, aero_power_W[:, body_index], label=f"Body {body_index}")
        axes[1].plot(
            x_periods,
            applied_power_W[:, body_index],
            label=f"Body {body_index}",
        )
    axes[0].set_ylabel("Aero Px (W)")
    axes[0].set_title("Aerodynamic Streamwise Force-Power")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylabel("Applied Px (W)")
    axes[1].set_xlabel("Time / Flapping Period")
    axes[1].set_title("Applied Streamwise Force-Power")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    save_path = output_dir / "streamwise_power_history.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_clamp_load_plot(
    output_dir: Path,
    times_s: np.ndarray,
    clamp_forces_yz_E: np.ndarray,
    clamp_moments_E_Cg: np.ndarray,
) -> Path:
    """Save clamp force and torque histories."""
    n = min(len(times_s), len(clamp_forces_yz_E), len(clamp_moments_E_Cg))
    x_periods = times_s[:n] / REFERENCE_PERIOD_S

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for body_index in range(2):
        axes[0].plot(
            x_periods,
            clamp_forces_yz_E[:n, body_index, 0],
            label=f"Body {body_index} Fy",
        )
        axes[0].plot(
            x_periods,
            clamp_forces_yz_E[:n, body_index, 1],
            linestyle="--",
            label=f"Body {body_index} Fz",
        )
        axes[1].plot(
            x_periods,
            clamp_moments_E_Cg[:n, body_index, 0],
            label=f"Body {body_index} Mx",
        )
        axes[1].plot(
            x_periods,
            clamp_moments_E_Cg[:n, body_index, 1],
            linestyle="--",
            label=f"Body {body_index} My",
        )
        axes[1].plot(
            x_periods,
            clamp_moments_E_Cg[:n, body_index, 2],
            linestyle=":",
            label=f"Body {body_index} Mz",
        )

    axes[0].set_ylabel("Clamp Force (N)")
    axes[0].set_title("Clamp Forces in Y/Z")
    axes[0].grid(True)
    axes[0].legend(ncol=2)

    axes[1].set_ylabel("Clamp Torque (N m)")
    axes[1].set_xlabel("Time / Flapping Period")
    axes[1].set_title("Clamp Torques About CG")
    axes[1].grid(True)
    axes[1].legend(ncol=3)

    fig.tight_layout()
    save_path = output_dir / "clamp_force_torque_history.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_case_plots(
    output_dir: Path,
    times_s: np.ndarray,
    positions_E_E: np.ndarray,
    velocities_E__E: np.ndarray,
    aero_forces_E: np.ndarray,
    clamp_arrays: dict[str, np.ndarray],
) -> dict[str, str]:
    """Save all per-case diagnostic plots."""
    plots = {
        "streamwise_separation_velocity": str(
            save_separation_velocity_plot(
                output_dir=output_dir,
                times_s=times_s,
                positions_E_E=positions_E_E,
                velocities_E__E=velocities_E__E,
            )
        ),
        "streamwise_power_history": str(
            save_power_plot(
                output_dir=output_dir,
                times_s=times_s,
                velocities_E__E=velocities_E__E,
                aero_forces_E=aero_forces_E,
                raw_forces_E=clamp_arrays["raw_forces_E_N"],
            )
        ),
        "clamp_force_torque_history": str(
            save_clamp_load_plot(
                output_dir=output_dir,
                times_s=times_s,
                clamp_forces_yz_E=clamp_arrays["clamp_forces_yz_E_N"],
                clamp_moments_E_Cg=clamp_arrays["clamp_moments_E_Cg_Nm"],
            )
        ),
    }
    return plots


def make_initial_collision_skipped_summary(
    output_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    baseline_power_W: np.ndarray | None,
    aircraft_model: str,
    angle_of_attack_deg: float,
    max_abs_x_over_span: float,
    max_abs_speed_mps: float,
) -> dict[str, Any]:
    """Write a summary for a case skipped before launch due to initial overlap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_positions_E_E = np.asarray(
        body_positions_from_paper_offsets(
            x_over_span=x_over_span,
            y_over_span=y_over_span,
            z_over_span=z_over_span,
        ),
        dtype=float,
    )
    error_message = (
        "Skipped before launch: initial rectangular wing planforms overlap. "
        "Increase |X|/B, |Y|/B, or |Z|/B."
    )
    summary: dict[str, Any] = {
        "case": f"streamwise_stability_energy_two_{aircraft_model}_bodies",
        "aircraft_model": aircraft_model,
        "wake_model": "free",
        "prescribed_wake": False,
        "constraint_mode": "streamwise_x_free_yz_and_attitude_clamped",
        "max_abs_x_over_span_guard": max_abs_x_over_span,
        "max_abs_speed_mps_guard": max_abs_speed_mps,
        "x_over_span_initial": x_over_span,
        "y_over_span_prescribed": y_over_span,
        "z_over_span_prescribed": z_over_span,
        "span_m": FULL_SPAN_M,
        "semispan_m": SEMI_SPAN_M,
        "root_chord_m": ROOT_CHORD_M,
        "tip_chord_m": TIP_CHORD_M,
        "chord_ref_m": REFERENCE_C_REF_M,
        "reference_area_m2": REFERENCE_AREA_M2,
        "aspect_ratio": ASPECT_RATIO,
        "num_chordwise_panels": NUM_CHORDWISE_PANELS,
        "num_spanwise_panels_total": 2 * NUM_SPANWISE_PANELS_PER_HALF,
        "initial_collision_free": False,
        "body_1_initial_position_E_m": initial_positions_E_E[0].tolist(),
        "body_2_initial_position_E_m": initial_positions_E_E[1].tolist(),
        "fixed_roll_deg": 0.0,
        "angle_of_attack_deg": angle_of_attack_deg,
        "initial_alpha_deg": angle_of_attack_deg,
        "fixed_pitch_deg": angle_of_attack_deg,
        "fixed_pitch_matches_angle_of_attack": True,
        "fixed_yaw_deg": 0.0,
        "reference_period_s": REFERENCE_PERIOD_S,
        "steps_per_flap": steps_per_flap,
        "steps_per_reference_period": steps_per_flap,
        "delta_time_s": REFERENCE_PERIOD_S / steps_per_flap,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "periods_total_requested": (prescribed_num_steps + free_num_steps)
        / steps_per_flap,
        "periods_total_completed": 0.0,
        "run_status": "skipped_initial_collision",
        "error_message": error_message,
        "final_x_over_span": x_over_span,
        "final_y_over_span": y_over_span,
        "final_z_over_span": z_over_span,
        "last_period_mean_x_over_span": x_over_span,
        "last_period_mean_dXdt_mps": None,
        "final_body_0_Ux_mps": None,
        "final_body_1_Ux_mps": None,
        "last_period_mean_Ux_mps": [None, None],
        "last_period_mean_aero_streamwise_power_W": [None, None],
        "last_period_mean_applied_streamwise_power_W": [None, None],
        "last_period_mean_pair_applied_streamwise_power_W": None,
        "last_period_mean_clamp_power_proxy_W": [None, None],
        "max_yz_drift_m": 0.0,
        "max_euler_deviation_deg": 0.0,
        "stage_1_energy_metric": "not_run_initial_collision",
        "stage_2_wbar_status": "not_run_initial_collision",
        "clamp_force_yz_stats_N": clamp_statistics(
            np.zeros((0, 2, 2), dtype=float),
            ("Fy", "Fz"),
        ),
        "clamp_torque_stats_Nm": clamp_statistics(
            np.zeros((0, 2, 3), dtype=float),
            ("Mx", "My", "Mz"),
        ),
    }
    if baseline_power_W is not None:
        summary["baseline_last_period_mean_applied_power_W"] = baseline_power_W.tolist()
        summary["last_period_delta_power_vs_baseline_W"] = [None, None]
        summary["last_period_pair_mean_delta_power_vs_baseline_W"] = None
    ff_utils.write_json(output_dir / "summary.json", summary)
    return summary


def run_streamwise_case(
    output_dir: Path,
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    show_progress: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    history_save_dir: Path | None,
    render_wake_movie: bool,
    baseline_power_W: np.ndarray | None = None,
    compute_wbar: bool = False,
    aircraft_model: str = DEFAULT_AIRCRAFT_MODEL,
    angle_of_attack_deg: float = DEFAULT_ANGLE_OF_ATTACK_DEG,
    max_abs_x_over_span: float = 20.0,
    max_abs_speed_mps: float = 50.0,
) -> dict[str, Any]:
    """Run one streamwise-only formation case and write diagnostics."""
    if not initial_condition_is_collision_free(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
    ):
        print(
            "Skipping initial-overlap case "
            f"X/B={x_over_span:.3f}, Y/B={y_over_span:.3f}, Z/B={z_over_span:.3f}."
        )
        return make_initial_collision_skipped_summary(
            output_dir=output_dir,
            x_over_span=x_over_span,
            y_over_span=y_over_span,
            z_over_span=z_over_span,
            prescribed_num_steps=prescribed_num_steps,
            free_num_steps=free_num_steps,
            steps_per_flap=steps_per_flap,
            baseline_power_W=baseline_power_W,
            aircraft_model=aircraft_model,
            angle_of_attack_deg=angle_of_attack_deg,
            max_abs_x_over_span=max_abs_x_over_span,
            max_abs_speed_mps=max_abs_speed_mps,
        )

    coupled_problem, coupled_solver, clamp_diagnostics = build_problem(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        aircraft_model=aircraft_model,
        angle_of_attack_deg=angle_of_attack_deg,
        max_abs_x_over_span=max_abs_x_over_span,
        max_abs_speed_mps=max_abs_speed_mps,
    )
    run_status = "ok"
    error_message: str | None = None
    try:
        coupled_solver.run(
            prescribed_wake=False,
            show_progress=show_progress,
            history_stride=history_stride,
            save_every_n_steps=save_every_n_steps,
            history_save_dir=history_save_dir,
        )
    except Exception as exc:
        run_status = "failed"
        error_message = repr(exc)
        print(
            "Streamwise case failed; saving partial diagnostics for "
            f"X/B={x_over_span:.3f}, Y/B={y_over_span:.3f}, Z/B={z_over_span:.3f}. "
            f"Error: {error_message}"
        )

    times_s, positions_E_E, velocities_E__E, alphas_deg, euler_angles_deg = (
        get_history_arrays(coupled_problem)
    )
    aero_forces_E = aerodynamic_forces_E_history(coupled_problem)
    clamp_arrays = clamp_diagnostics.history_arrays()
    wbar_arrays = (
        compute_source_separated_wbar_history(coupled_solver)
        if compute_wbar and history_stride == 1
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_history_npz(
        output_dir=output_dir,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        aero_forces_E=aero_forces_E,
        clamp_arrays=clamp_arrays,
        extra_arrays=wbar_arrays,
    )
    summary = compute_run_metrics(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        prescribed_wake=False,
        run_status=run_status,
        error_message=error_message,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        euler_angles_deg=euler_angles_deg,
        aero_forces_E=aero_forces_E,
        clamp_arrays=clamp_arrays,
        baseline_power_W=baseline_power_W,
        wbar_arrays=wbar_arrays,
        aircraft_model=aircraft_model,
        angle_of_attack_deg=angle_of_attack_deg,
        max_abs_x_over_span=max_abs_x_over_span,
        max_abs_speed_mps=max_abs_speed_mps,
    )
    if compute_wbar and history_stride != 1:
        summary["stage_2_wbar_status"] = "skipped_requires_history_stride_1"
    summary["history_npz"] = str(history_path)
    summary["diagnostic_plots"] = save_case_plots(
        output_dir=output_dir,
        times_s=times_s,
        positions_E_E=positions_E_E,
        velocities_E__E=velocities_E__E,
        aero_forces_E=aero_forces_E,
        clamp_arrays=clamp_arrays,
    )

    if render_wake_movie and history_stride == 1 and len(times_s) >= 2:
        try:
            movie_path = inline_case.save_standard_wake_movie(
                coupled_solver=coupled_solver,
                output_dir=output_dir,
                filename_stem="AnimateFreeFlight_streamwise_standard_wake",
                follow_body_index=0,
                show_wake_vortices=True,
            )
            summary["standard_wake_movie_mp4"] = str(movie_path)
        except Exception as exc:
            summary["standard_wake_movie_error"] = repr(exc)
            print(f"Standard wake render failed for {output_dir}: {exc!r}")
    elif render_wake_movie:
        summary["standard_wake_movie_skipped"] = (
            "standard wake movies require history_stride=1 and at least 2 time samples"
        )

    ff_utils.write_json(output_dir / "summary.json", summary)
    return summary


def summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the high-value summary fields for sweep CSV output."""
    row_keys = (
        "run_status",
        "x_over_span_initial",
        "y_over_span_prescribed",
        "z_over_span_prescribed",
        "final_x_over_span",
        "last_period_mean_x_over_span",
        "last_period_mean_dXdt_mps",
        "last_period_mean_pair_applied_streamwise_power_W",
        "last_period_pair_mean_delta_power_vs_baseline_W",
        "max_yz_drift_m",
        "max_euler_deviation_deg",
        "periods_total_completed",
    )
    row = {key: summary.get(key, "") for key in row_keys}
    for body_index in range(2):
        prefix = f"body_{body_index}_"
        row[f"{prefix}last_period_mean_Ux_mps"] = summary["last_period_mean_Ux_mps"][
            body_index
        ]
        row[f"{prefix}last_period_mean_applied_power_W"] = summary[
            "last_period_mean_applied_streamwise_power_W"
        ][body_index]
        row[f"{prefix}clamp_Fy_rms_N"] = summary["clamp_force_yz_stats_N"][
            f"{prefix}Fy_rms"
        ]
        row[f"{prefix}clamp_Fz_rms_N"] = summary["clamp_force_yz_stats_N"][
            f"{prefix}Fz_rms"
        ]
        row[f"{prefix}clamp_Mx_rms_Nm"] = summary["clamp_torque_stats_Nm"][
            f"{prefix}Mx_rms"
        ]
        row[f"{prefix}clamp_My_rms_Nm"] = summary["clamp_torque_stats_Nm"][
            f"{prefix}My_rms"
        ]
        row[f"{prefix}clamp_Mz_rms_Nm"] = summary["clamp_torque_stats_Nm"][
            f"{prefix}Mz_rms"
        ]
    return row


def write_sweep_csv(output_root: Path, summaries: list[dict[str, Any]]) -> Path:
    """Write flattened sweep summaries."""
    rows = [summary_csv_row(summary) for summary in summaries]
    save_path = output_root / "sweep_summary.csv"
    if not rows:
        save_path.write_text("")
        return save_path
    with save_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return save_path


def save_sweep_maps(
    output_root: Path, summaries: list[dict[str, Any]]
) -> dict[str, str]:
    """Save compact scatter maps for energy, stability, and clamp cost."""
    ok_summaries = [summary for summary in summaries if summary["run_status"] == "ok"]
    if not ok_summaries:
        return {}

    y_values = np.array(
        [s["y_over_span_prescribed"] for s in ok_summaries], dtype=float
    )
    z_values = np.array(
        [s["z_over_span_prescribed"] for s in ok_summaries], dtype=float
    )
    x_values = np.array([s["x_over_span_initial"] for s in ok_summaries], dtype=float)
    drift_values = np.array(
        [s["last_period_mean_dXdt_mps"] for s in ok_summaries],
        dtype=float,
    )
    power_values = np.array(
        [
            s.get(
                "last_period_pair_mean_delta_power_vs_baseline_W",
                s["last_period_mean_pair_applied_streamwise_power_W"],
            )
            for s in ok_summaries
        ],
        dtype=float,
    )
    clamp_values = np.array(
        [
            0.5
            * (
                s["clamp_force_yz_stats_N"]["body_0_Fz_rms"]
                + s["clamp_force_yz_stats_N"]["body_1_Fz_rms"]
            )
            for s in ok_summaries
        ],
        dtype=float,
    )

    paths: dict[str, str] = {}
    map_specs = (
        ("energy_map.png", power_values, "Power metric (W)"),
        ("stability_map.png", -np.abs(drift_values), "-|mean dX/dt| (m/s)"),
        ("clamp_load_map.png", clamp_values, "Mean Fz clamp RMS (N)"),
    )
    for filename, color_values, color_label in map_specs:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            x_values,
            y_values,
            z_values,
            c=color_values,
            cmap="coolwarm",
            s=55,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel("X/B initial")
        ax.set_ylabel("Y/B prescribed")
        ax.set_zlabel("Z/B prescribed")
        ax.set_title(color_label)
        fig.colorbar(scatter, ax=ax, shrink=0.75, label=color_label)
        fig.tight_layout()
        save_path = output_root / filename
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        paths[filename.removesuffix(".png")] = str(save_path)
    return paths


def _nearest_value(values: np.ndarray, target: float) -> float:
    """Return the available value closest to a target slice."""
    unique_values = np.unique(values)
    return float(unique_values[np.argmin(np.abs(unique_values - target))])


def _plot_slice_scatter(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    fixed_key: str,
    fixed_value: float,
    x_key: str,
    y_key: str,
    color_key: str,
    title: str,
) -> None:
    """Plot one 2D slice for the theory-validation panel."""
    slice_summaries = [
        summary
        for summary in summaries
        if np.isclose(summary[fixed_key], fixed_value, atol=1e-12)
    ]
    if not slice_summaries:
        ax.set_title(f"{title}\n(no data)")
        ax.set_axis_off()
        return

    x_values = np.array([summary[x_key] for summary in slice_summaries], dtype=float)
    y_values = np.array([summary[y_key] for summary in slice_summaries], dtype=float)
    color_values = np.array(
        [
            summary.get(
                color_key, summary["last_period_mean_pair_applied_streamwise_power_W"]
            )
            for summary in slice_summaries
        ],
        dtype=float,
    )
    scatter = ax.scatter(
        x_values,
        y_values,
        c=color_values,
        cmap="coolwarm",
        s=50,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_over_span_", "/B "))
    ax.set_ylabel(y_key.replace("_over_span_", "/B "))
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def save_theory_validation_9panel(
    output_root: Path,
    summaries: list[dict[str, Any]],
) -> Path | None:
    """Save a compact 9-panel figure aligned with the theory-document slices."""
    ok_summaries = [summary for summary in summaries if summary["run_status"] == "ok"]
    if not ok_summaries:
        return None

    for summary in ok_summaries:
        summary["stability_margin_proxy_mps"] = -abs(
            summary["last_period_mean_dXdt_mps"]
        )
        summary["pair_power_metric_W"] = summary.get(
            "last_period_pair_mean_delta_power_vs_baseline_W",
            summary["last_period_mean_pair_applied_streamwise_power_W"],
        )
        summary["body_0_power_metric_W"] = summary.get(
            "last_period_delta_power_vs_baseline_W",
            summary["last_period_mean_applied_streamwise_power_W"],
        )[0]

    x_values = np.array([s["x_over_span_initial"] for s in ok_summaries], dtype=float)
    y_values = np.array(
        [s["y_over_span_prescribed"] for s in ok_summaries], dtype=float
    )
    z_values = np.array(
        [s["z_over_span_prescribed"] for s in ok_summaries], dtype=float
    )
    x_slice = _nearest_value(x_values, 0.0)
    y_slice = _nearest_value(y_values, 0.0)
    z_slice = _nearest_value(z_values, 0.0)

    rows = (
        ("body_0_power_metric_W", "Body 1 Power Metric"),
        ("pair_power_metric_W", "Pair-Average Power Metric"),
        ("stability_margin_proxy_mps", "Streamwise Stability Proxy"),
    )
    columns = (
        (
            "z_over_span_prescribed",
            z_slice,
            "x_over_span_initial",
            "y_over_span_prescribed",
            f"Z/B = {z_slice:g}",
        ),
        (
            "x_over_span_initial",
            x_slice,
            "y_over_span_prescribed",
            "z_over_span_prescribed",
            f"X/B = {x_slice:g}",
        ),
        (
            "y_over_span_prescribed",
            y_slice,
            "x_over_span_initial",
            "z_over_span_prescribed",
            f"Y/B = {y_slice:g}",
        ),
    )

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row_index, (color_key, row_title) in enumerate(rows):
        for col_index, (fixed_key, fixed_value, x_key, y_key, col_title) in enumerate(
            columns
        ):
            _plot_slice_scatter(
                ax=axes[row_index, col_index],
                summaries=ok_summaries,
                fixed_key=fixed_key,
                fixed_value=fixed_value,
                x_key=x_key,
                y_key=y_key,
                color_key=color_key,
                title=f"{row_title}\n{col_title}",
            )

    fig.suptitle("Streamwise Formation Energy And Stability Slices")
    fig.tight_layout()
    save_path = output_root / "theory_validation_9panel.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def add_x_perturbations(
    x_over_span_values: tuple[float, ...],
    perturbation_over_span: float,
) -> tuple[float, ...]:
    """Add X/B +/- perturbations around each requested initial streamwise station."""
    augmented: set[float] = set()
    for value in x_over_span_values:
        augmented.add(float(value))
        augmented.add(float(value - perturbation_over_span))
        augmented.add(float(value + perturbation_over_span))
    return tuple(sorted(augmented))


def run_sweep(
    output_root: Path,
    x_over_span_values: tuple[float, ...],
    y_over_span_values: tuple[float, ...],
    z_over_span_values: tuple[float, ...],
    total_periods: float,
    prescribed_periods: float,
    steps_per_flap: int,
    show_progress: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    render_wake_movies: bool,
    render_limit: int,
    run_baseline: bool,
    baseline_y_over_span: float,
    include_x_perturbations: bool = False,
    x_perturbation_over_span: float = 0.1,
    compute_wbar: bool = False,
    aircraft_model: str = DEFAULT_AIRCRAFT_MODEL,
    angle_of_attack_deg: float = DEFAULT_ANGLE_OF_ATTACK_DEG,
    max_abs_x_over_span: float = 20.0,
    max_abs_speed_mps: float = 50.0,
) -> dict[str, Any]:
    """Run the requested streamwise stability and energy sweep."""
    output_root.mkdir(parents=True, exist_ok=True)
    requested_x_over_span_values = x_over_span_values
    if include_x_perturbations:
        x_over_span_values = add_x_perturbations(
            x_over_span_values=x_over_span_values,
            perturbation_over_span=x_perturbation_over_span,
        )
    prescribed_num_steps = int(round(prescribed_periods * steps_per_flap))
    total_num_steps = int(round(total_periods * steps_per_flap))
    free_num_steps = max(1, total_num_steps - prescribed_num_steps)

    baseline_power_W: np.ndarray | None = None
    baseline_summary: dict[str, Any] | None = None
    if run_baseline:
        baseline_dir = output_root / "baseline_far_lateral"
        baseline_summary = run_streamwise_case(
            output_dir=baseline_dir,
            x_over_span=0.0,
            y_over_span=baseline_y_over_span,
            z_over_span=0.0,
            prescribed_num_steps=prescribed_num_steps,
            free_num_steps=free_num_steps,
            steps_per_flap=steps_per_flap,
            show_progress=show_progress,
            history_stride=history_stride,
            save_every_n_steps=save_every_n_steps,
            history_save_dir=None,
            render_wake_movie=False,
            baseline_power_W=None,
            compute_wbar=compute_wbar,
            aircraft_model=aircraft_model,
            angle_of_attack_deg=angle_of_attack_deg,
            max_abs_x_over_span=max_abs_x_over_span,
            max_abs_speed_mps=max_abs_speed_mps,
        )
        baseline_power_W = np.asarray(
            baseline_summary["last_period_mean_applied_streamwise_power_W"],
            dtype=float,
        )

    summaries: list[dict[str, Any]] = []
    render_count = 0
    for z_over_span in z_over_span_values:
        for y_over_span in y_over_span_values:
            for x_over_span in x_over_span_values:
                this_label = run_label(x_over_span, y_over_span, z_over_span)
                this_output_dir = output_root / this_label
                this_history_save_dir: Path | None = None
                if save_every_n_steps is not None:
                    this_history_save_dir = this_output_dir / "streamed_history"
                render_this = render_wake_movies and render_count < render_limit
                summary = run_streamwise_case(
                    output_dir=this_output_dir,
                    x_over_span=x_over_span,
                    y_over_span=y_over_span,
                    z_over_span=z_over_span,
                    prescribed_num_steps=prescribed_num_steps,
                    free_num_steps=free_num_steps,
                    steps_per_flap=steps_per_flap,
                    show_progress=show_progress,
                    history_stride=history_stride,
                    save_every_n_steps=save_every_n_steps,
                    history_save_dir=this_history_save_dir,
                    render_wake_movie=render_this,
                    baseline_power_W=baseline_power_W,
                    compute_wbar=compute_wbar,
                    aircraft_model=aircraft_model,
                    angle_of_attack_deg=angle_of_attack_deg,
                    max_abs_x_over_span=max_abs_x_over_span,
                    max_abs_speed_mps=max_abs_speed_mps,
                )
                if render_this:
                    render_count += 1
                summaries.append(summary)
                print(
                    f"Saved streamwise case {this_label}: "
                    f"{summary['run_status']} -> {this_output_dir / 'summary.json'}"
                )

    csv_path = write_sweep_csv(output_root=output_root, summaries=summaries)
    map_paths = save_sweep_maps(output_root=output_root, summaries=summaries)
    theory_panel_path = save_theory_validation_9panel(
        output_root=output_root,
        summaries=summaries,
    )
    sweep_summary = {
        "case": "streamwise_stability_energy_sweep",
        "aircraft_model": aircraft_model,
        "angle_of_attack_deg": angle_of_attack_deg,
        "fixed_pitch_deg": angle_of_attack_deg,
        "fixed_pitch_matches_angle_of_attack": True,
        "wake_model": "free",
        "prescribed_wake": False,
        "requested_x_over_span_values": list(requested_x_over_span_values),
        "x_over_span_values": list(x_over_span_values),
        "y_over_span_values": list(y_over_span_values),
        "z_over_span_values": list(z_over_span_values),
        "span_m": FULL_SPAN_M,
        "semispan_m": SEMI_SPAN_M,
        "root_chord_m": ROOT_CHORD_M,
        "tip_chord_m": TIP_CHORD_M,
        "chord_ref_m": REFERENCE_C_REF_M,
        "reference_area_m2": REFERENCE_AREA_M2,
        "aspect_ratio": ASPECT_RATIO,
        "num_chordwise_panels": NUM_CHORDWISE_PANELS,
        "num_spanwise_panels_total": 2 * NUM_SPANWISE_PANELS_PER_HALF,
        "initial_collision_rule": (
            "skip if |X| < chord, |Y| < span, and |Z| is effectively zero"
        ),
        "total_periods": total_periods,
        "prescribed_periods": prescribed_periods,
        "steps_per_flap": steps_per_flap,
        "steps_per_reference_period": steps_per_flap,
        "prescribed_steps": prescribed_num_steps,
        "free_steps": free_num_steps,
        "baseline_summary": baseline_summary,
        "num_cases": len(summaries),
        "num_ok_cases": sum(summary["run_status"] == "ok" for summary in summaries),
        "num_skipped_initial_collision_cases": sum(
            summary["run_status"] == "skipped_initial_collision"
            for summary in summaries
        ),
        "sweep_summary_csv": str(csv_path),
        "sweep_maps": map_paths,
        "theory_validation_9panel": (
            None if theory_panel_path is None else str(theory_panel_path)
        ),
        "include_x_perturbations": include_x_perturbations,
        "x_perturbation_over_span": x_perturbation_over_span,
        "compute_wbar": compute_wbar,
        "max_abs_x_over_span_guard": max_abs_x_over_span,
        "max_abs_speed_mps_guard": max_abs_speed_mps,
        "case_summaries": summaries,
    }
    ff_utils.write_json(output_root / "sweep_summary.json", sweep_summary)
    return sweep_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run two fixed-wing bodies in a streamwise-only formation slice using a "
            "free vortex wake and record energy/stability/clamp diagnostics."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where sweep outputs should be written.",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=("smoke", "production", "custom"),
        default="smoke",
        help="Choose built-in smoke/production grids or custom coordinate lists.",
    )
    parser.add_argument(
        "--x-over-span",
        type=str,
        default=None,
        help="Comma-separated initial X/B values for custom or override sweeps.",
    )
    parser.add_argument(
        "--y-over-span",
        type=str,
        default=None,
        help="Comma-separated prescribed Y/B values for custom or override sweeps.",
    )
    parser.add_argument(
        "--z-over-span",
        type=str,
        default=None,
        help="Comma-separated prescribed Z/B values for custom or override sweeps.",
    )
    parser.add_argument(
        "--total-periods",
        type=float,
        default=None,
        help="Total reference periods including the prescribed startup phase.",
    )
    parser.add_argument(
        "--prescribed-periods",
        type=float,
        default=DEFAULT_PRESCRIBED_PERIODS,
        help="Prescribed startup periods before x-only dynamics receives loads.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=DEFAULT_STEPS_PER_REFERENCE_PERIOD,
        help=(
            "Temporal resolution in steps per reference period. The option name is "
            "kept for compatibility with earlier flapping runs."
        ),
    )
    parser.add_argument(
        "--aircraft-model",
        choices=(AIRCRAFT_MODEL_FIXED_WING, AIRCRAFT_MODEL_FLAPPING),
        default=DEFAULT_AIRCRAFT_MODEL,
        help=(
            "Use static fixed wings for the theory study by default. The flapping "
            "model is retained only for comparison with earlier runs."
        ),
    )
    parser.add_argument(
        "--angle-of-attack-deg",
        type=float,
        default=DEFAULT_ANGLE_OF_ATTACK_DEG,
        help=(
            "Initial alpha and fixed pitch angle for the clamped fixed-wing slice. "
            "For this x-only theory study, pitch is kept equal to AOA."
        ),
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show solver progress bars.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Retain full in-memory history every N steps.",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=None,
        help="Write compressed solver snapshots every N steps.",
    )
    parser.add_argument(
        "--render-wake-movies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render standard wake-visible movies for the first selected cases.",
    )
    parser.add_argument(
        "--render-limit",
        type=int,
        default=3,
        help="Maximum number of per-case standard wake movies to render.",
    )
    parser.add_argument(
        "--run-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a far-lateral baseline for direct power-delta comparisons.",
    )
    parser.add_argument(
        "--baseline-y-over-span",
        type=float,
        default=10.0,
        help="Y/B offset for the far-lateral baseline pair.",
    )
    parser.add_argument(
        "--include-x-perturbations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run X/B +/- perturbation cases around each requested X/B value.",
    )
    parser.add_argument(
        "--x-perturbation-over-span",
        type=float,
        default=0.1,
        help="Perturbation amplitude in X/B when perturbation cases are enabled.",
    )
    parser.add_argument(
        "--compute-wbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute source-separated lift-weighted Wbar and DeltaP=-L*Wbar "
            "diagnostics. Requires retained full history."
        ),
    )
    parser.add_argument(
        "--max-abs-x-over-span",
        type=float,
        default=20.0,
        help="Fail a case early if the streamwise separation magnitude exceeds this X/B.",
    )
    parser.add_argument(
        "--max-abs-speed-mps",
        type=float,
        default=50.0,
        help="Fail a case early if any body's streamwise speed magnitude exceeds this.",
    )
    return parser.parse_args()


def _resolve_grid(args: argparse.Namespace) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
]:
    """Resolve CLI grid defaults and overrides."""
    if args.sweep_mode == "production":
        x_values = DEFAULT_PRODUCTION_X_OVER_SPAN
        y_values = DEFAULT_PRODUCTION_Y_OVER_SPAN
        z_values = DEFAULT_PRODUCTION_Z_OVER_SPAN
        total_periods = DEFAULT_PRODUCTION_TOTAL_PERIODS
    else:
        x_values = DEFAULT_SMOKE_X_OVER_SPAN
        y_values = DEFAULT_SMOKE_Y_OVER_SPAN
        z_values = DEFAULT_SMOKE_Z_OVER_SPAN
        total_periods = DEFAULT_SMOKE_TOTAL_PERIODS

    if args.x_over_span is not None:
        x_values = parse_float_tuple(args.x_over_span)
    if args.y_over_span is not None:
        y_values = parse_float_tuple(args.y_over_span)
    if args.z_over_span is not None:
        z_values = parse_float_tuple(args.z_over_span)
    if args.total_periods is not None:
        total_periods = float(args.total_periods)
    return x_values, y_values, z_values, total_periods


def main() -> None:
    """Run the requested sweep."""
    args = parse_args()
    x_values, y_values, z_values, total_periods = _resolve_grid(args)
    run_sweep(
        output_root=args.output_root,
        x_over_span_values=x_values,
        y_over_span_values=y_values,
        z_over_span_values=z_values,
        total_periods=total_periods,
        prescribed_periods=args.prescribed_periods,
        steps_per_flap=args.steps_per_flap,
        show_progress=args.show_progress,
        history_stride=args.history_stride,
        save_every_n_steps=args.save_every_n_steps,
        render_wake_movies=args.render_wake_movies,
        render_limit=args.render_limit,
        run_baseline=args.run_baseline,
        baseline_y_over_span=args.baseline_y_over_span,
        include_x_perturbations=args.include_x_perturbations,
        x_perturbation_over_span=args.x_perturbation_over_span,
        compute_wbar=args.compute_wbar,
        aircraft_model=args.aircraft_model,
        angle_of_attack_deg=args.angle_of_attack_deg,
        max_abs_x_over_span=args.max_abs_x_over_span,
        max_abs_speed_mps=args.max_abs_speed_mps,
    )


if __name__ == "__main__":
    main()
