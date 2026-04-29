"""Run the tuned fixed-pitch flapping free-flight benchmark case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps

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

FLAPPING_FREQUENCY_HZ = 8.0
FLAPPING_PERIOD_S = 1.0 / FLAPPING_FREQUENCY_HZ
DEFAULT_STEPS_PER_FLAP = 48

INITIAL_SPEED_MPS = 2.5
INITIAL_ALPHA_DEG = 5.0
INITIAL_YAW_DEG = 0.0
FIXED_PITCH_DEG = 5.0

TOTAL_AIRCRAFT_WEIGHT_N = 0.0235
BODY_DRAG_LINEAR_N_PER_MPS = 0.00105
BODY_DRAG_CD_AREA_M2 = 0.0007

INERTIA_BP1_CGP1 = np.array(
    [
        [2.5e-5, 0.0, 0.0],
        [0.0, 4.5e-5, 0.0],
        [0.0, 0.0, 3.0e-5],
    ],
    dtype=float,
)

WING_ROOT_Y_M = 0.015
SEMI_SPAN_M = 0.11
ROOT_CHORD_M = 0.04
TIP_CHORD_M = 0.03

STROKE_AMPLITUDE_DEG = 30.0
FEATHER_AMPLITUDE_DEG = 12.0
FEATHER_PHASE_DEG = 110.0

DEFAULT_PRESCRIBED_STEPS = 4 * DEFAULT_STEPS_PER_FLAP
DEFAULT_FREE_STEPS = 48 * DEFAULT_STEPS_PER_FLAP
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "03_flapping_forward"
    / "fixed_pitch"
)


def build_half_wing(*, name: str, mirror_only: bool) -> ps.geometry.wing.Wing:
    """Build one rigid half-wing."""
    symmetry_normal = (0.0, 1.0, 0.0) if mirror_only else None
    symmetry_point = (0.0, 0.0, 0.0) if mirror_only else None

    return ps.geometry.wing.Wing(
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=ROOT_CHORD_M,
                num_spanwise_panels=4,
                spanwise_spacing="uniform",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=TIP_CHORD_M,
                Lp_Wcsp_Lpp=(0.0, SEMI_SPAN_M, 0.0),
                num_spanwise_panels=None,
            ),
        ],
        name=name,
        Ler_Gs_Cgs=(0.0, WING_ROOT_Y_M, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=False,
        mirror_only=mirror_only,
        symmetryNormal_G=symmetry_normal,
        symmetryPoint_G_Cg=symmetry_point,
        num_chordwise_panels=4,
        chordwise_spacing="uniform",
    )


def build_airplane() -> ps.geometry.airplane.Airplane:
    """Build the simple two-wing flapping aircraft."""
    return ps.geometry.airplane.Airplane(
        wings=[
            build_half_wing(name="Left Wing", mirror_only=False),
            build_half_wing(name="Right Wing", mirror_only=True),
        ],
        name="Flapping Forward Free-Flight Case",
        weight=TOTAL_AIRCRAFT_WEIGHT_N,
    )


def rigid_body_drag_model(
    coupled_operating_point: ps.operating_point.CoupledOperatingPoint,
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tuned rigid-body drag surrogate in wind axes."""
    del airplane

    speed_mps = coupled_operating_point.vCg__E
    rho = coupled_operating_point.rho

    linear_drag_n = BODY_DRAG_LINEAR_N_PER_MPS * speed_mps
    quadratic_drag_n = 0.5 * rho * BODY_DRAG_CD_AREA_M2 * speed_mps**2
    total_drag_n = linear_drag_n + quadratic_drag_n

    return (
        np.array([-total_drag_n, 0.0, 0.0], dtype=float),
        np.zeros(3, dtype=float),
    )


def build_wing_cross_section_movements(
    wing: ps.geometry.wing.Wing,
) -> list[ps.movements.wing_cross_section_movement.WingCrossSectionMovement]:
    """Create static cross-section movements for a rigid wing."""
    return [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing.wing_cross_sections[0]
        ),
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing.wing_cross_sections[1]
        ),
    ]


def build_wing_movement(
    wing: ps.geometry.wing.Wing,
) -> ps.movements.wing_movement.WingMovement:
    """Create one flapping wing movement."""
    return ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=build_wing_cross_section_movements(wing),
        rotationPointOffset_Gs_Ler=(0.25 * ROOT_CHORD_M, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(
            STROKE_AMPLITUDE_DEG,
            FEATHER_AMPLITUDE_DEG,
            0.0,
        ),
        periodAngles_Gs_to_Wn_ixyz=(FLAPPING_PERIOD_S, FLAPPING_PERIOD_S, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, FEATHER_PHASE_DEG, 0.0),
    )


def build_airplane_movement(
    airplane: ps.geometry.airplane.Airplane,
) -> ps.movements.airplane_movement.AirplaneMovement:
    """Create the prescribed internal flapping motion."""
    return ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            build_wing_movement(airplane.wings[0]),
            build_wing_movement(airplane.wings[1]),
        ],
    )


def build_problem(
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Build the tuned fixed-pitch coupled problem and solver."""
    delta_time = FLAPPING_PERIOD_S / steps_per_flap
    airplane = build_airplane()
    airplane_movement = build_airplane_movement(airplane)

    initial_coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
        rho=AIR_DENSITY,
        vCg__E=INITIAL_SPEED_MPS,
        alpha=INITIAL_ALPHA_DEG,
        beta=0.0,
        angles_E_to_BP1_izyx=(0.0, INITIAL_ALPHA_DEG, INITIAL_YAW_DEG),
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
        I_BP1_CgP1=INERTIA_BP1_CGP1,
        external_forces_fn=rigid_body_drag_model,
    )

    coupled_solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_unsteady_problem=coupled_problem
    )
    ff_utils.install_fixed_pitch_rotation(
        coupled_problem=coupled_problem,
        target_pitch_deg=FIXED_PITCH_DEG,
    )
    return coupled_problem, coupled_solver


def build_summary(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    steps_per_flap: int,
    prescribed_wake: bool,
) -> dict[str, Any]:
    """Build a machine-readable summary of a completed run."""
    times_s, velocities_E__E, _, alphas_deg, euler_angles_deg = (
        ff_utils.get_history_arrays(coupled_problem)
    )
    summary: dict[str, Any] = {
        "case": "03_flapping_forward",
        "pitch_mode": "fixed",
        "wake_model": "prescribed" if prescribed_wake else "free",
        "weight_n": TOTAL_AIRCRAFT_WEIGHT_N,
        "linear_drag_n_per_mps": BODY_DRAG_LINEAR_N_PER_MPS,
        "cda_m2": BODY_DRAG_CD_AREA_M2,
        "fixed_pitch_deg": FIXED_PITCH_DEG,
        "steps_per_flap": steps_per_flap,
        "flapping_frequency_hz": FLAPPING_FREQUENCY_HZ,
        "flapping_period_s": FLAPPING_PERIOD_S,
        "delta_time_s": coupled_problem.delta_time,
        "prescribed_steps": coupled_problem.coupled_movement.prescribed_num_steps,
        "free_steps": coupled_problem.coupled_movement.free_num_steps,
        "periods_total": (
            coupled_problem.coupled_movement.prescribed_num_steps
            + coupled_problem.coupled_movement.free_num_steps
        )
        / steps_per_flap,
        "final_position_E_m": coupled_solver.stackPosition_E_E[-1].tolist(),
        "final_velocity_E_mps": velocities_E__E[-1].tolist(),
    }
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
    history_save_dir: Path | None,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Run the tuned flapping forward benchmark and save its outputs."""
    coupled_problem, coupled_solver = build_problem(
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = ff_utils.save_velocity_history_plot(
        x_values=times_over_period,
        velocities_E__E=velocities_E__E,
        speeds_mps=speeds_mps,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        save_path=output_dir / "velocity_history.png",
        x_label="Time / Flapping Period",
        title="Flapping Forward Free-Flight Case",
    )

    summary_path = ff_utils.write_json(
        output_dir / "summary.json",
        build_summary(
            coupled_problem=coupled_problem,
            coupled_solver=coupled_solver,
            steps_per_flap=steps_per_flap,
            prescribed_wake=prescribed_wake,
        ),
    )

    print(f"Saved summary to: {summary_path}")
    print(f"Saved velocity plot to: {plot_path}")

    if animate:
        webp_path, mp4_path = ff_utils.save_animation_bundle(
            coupled_solver=coupled_solver,
            output_dir=output_dir,
            filename_stem="AnimateFreeFlight_flapping_forward_fixed_pitch_with_wake",
            show_wake_vortices=show_wake_vortices,
        )
        print(f"Saved animation to: {webp_path}")
        print(f"Saved movie to: {mp4_path}")

    return coupled_problem, coupled_solver


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the tuned fixed-pitch flapping forward free-flight case."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots, summaries, and movies should be saved.",
    )
    parser.add_argument(
        "--prescribed-steps",
        type=int,
        default=DEFAULT_PRESCRIBED_STEPS,
        help="Number of prescribed steps used to build the wake before free flight.",
    )
    parser.add_argument(
        "--free-steps",
        type=int,
        default=DEFAULT_FREE_STEPS,
        help="Number of coupled free-flight steps to run after the prescribed phase.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Temporal resolution in solver steps per flapping period.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    return parser.parse_args()


def main() -> None:
    """Run the script."""
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
        history_save_dir=args.history_save_dir,
    )


if __name__ == "__main__":
    main()
