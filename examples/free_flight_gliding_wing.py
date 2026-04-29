"""Run a geometry-matched gliding-wing free-flight case."""

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

REFERENCE_PERIOD_S = 0.125
DEFAULT_STEPS_PER_REFERENCE_PERIOD = 24

DEFAULT_INITIAL_SPEED_MPS = 2.5
DEFAULT_INITIAL_ALPHA_DEG = 5.0
INITIAL_YAW_DEG = 0.0

TOTAL_AIRCRAFT_WEIGHT_N = 0.0235
INERTIA_BP1_CGP1 = np.array(
    [
        [2.5e-5, 0.0, 0.0],
        [0.0, 4.5e-5, 0.0],
        [0.0, 0.0, 3.0e-5],
    ],
    dtype=float,
)

SEMI_SPAN_M = 0.11
ROOT_CHORD_M = 0.04
TIP_CHORD_M = 0.03
FIXED_PITCH_DEG = 5.0

DEFAULT_PRESCRIBED_STEPS = 4 * DEFAULT_STEPS_PER_REFERENCE_PERIOD
DEFAULT_FREE_STEPS = 20 * DEFAULT_STEPS_PER_REFERENCE_PERIOD
DEFAULT_SEARCH_PRESCRIBED_STEPS = 2 * DEFAULT_STEPS_PER_REFERENCE_PERIOD
DEFAULT_SEARCH_FREE_STEPS = 8 * DEFAULT_STEPS_PER_REFERENCE_PERIOD
DEFAULT_SEARCH_SPEEDS_MPS = (2.0, 2.5, 3.0, 3.5, 4.0)
DEFAULT_SEARCH_ALPHAS_DEG = (0.0, 2.5, 5.0, 7.5, 10.0)

DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "01_gliding_wing"
)


def build_gliding_wing() -> ps.geometry.wing.Wing:
    """Build one continuous static wing with the flapping-case planform family."""
    return ps.geometry.wing.Wing(
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=TIP_CHORD_M,
                num_spanwise_panels=4,
                spanwise_spacing="uniform",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=ROOT_CHORD_M,
                Lp_Wcsp_Lpp=(0.0, SEMI_SPAN_M, 0.0),
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
        name="Continuous Gliding Wing",
        Ler_Gs_Cgs=(0.0, -SEMI_SPAN_M, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=False,
        mirror_only=False,
        symmetryNormal_G=None,
        symmetryPoint_G_Cg=None,
        num_chordwise_panels=4,
        chordwise_spacing="uniform",
    )


def build_airplane() -> ps.geometry.airplane.Airplane:
    """Build the gliding wing aircraft."""
    return ps.geometry.airplane.Airplane(
        wings=[build_gliding_wing()],
        name="Geometry-Matched Gliding Wing",
        weight=TOTAL_AIRCRAFT_WEIGHT_N,
    )


def build_wing_cross_section_movements(
    wing: ps.geometry.wing.Wing,
) -> list[ps.movements.wing_cross_section_movement.WingCrossSectionMovement]:
    """Create static cross-section movements for the rigid wing."""
    return [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing_cross_section
        )
        for wing_cross_section in wing.wing_cross_sections
    ]


def build_airplane_movement(
    airplane: ps.geometry.airplane.Airplane,
) -> ps.movements.airplane_movement.AirplaneMovement:
    """Create the static airplane movement for the gliding wing."""
    return ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            ps.movements.wing_movement.WingMovement(
                base_wing=airplane.wings[0],
                wing_cross_section_movements=build_wing_cross_section_movements(
                    airplane.wings[0]
                ),
            )
        ],
    )


def build_problem(
    initial_speed_mps: float,
    initial_alpha_deg: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_reference_period: int,
    fix_pitch_rotation: bool,
    fixed_pitch_deg: float,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Build the gliding wing coupled problem and solver."""
    delta_time = REFERENCE_PERIOD_S / steps_per_reference_period
    airplane = build_airplane()
    airplane_movement = build_airplane_movement(airplane)

    initial_coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
        rho=AIR_DENSITY,
        vCg__E=initial_speed_mps,
        alpha=initial_alpha_deg,
        beta=0.0,
        angles_E_to_BP1_izyx=(0.0, initial_alpha_deg, INITIAL_YAW_DEG),
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
    )
    coupled_solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_unsteady_problem=coupled_problem
    )

    if fix_pitch_rotation:
        ff_utils.install_fixed_pitch_rotation(
            coupled_problem=coupled_problem,
            target_pitch_deg=fixed_pitch_deg,
        )

    return coupled_problem, coupled_solver


def score_search_case(
    velocities_E__E: np.ndarray,
    steps_per_reference_period: int,
) -> float:
    """Score a short gliding run by late-time vertical drift and forward motion."""
    two_period_block = min(len(velocities_E__E), 2 * steps_per_reference_period)
    last_two = velocities_E__E[-two_period_block:].mean(axis=0)
    if len(velocities_E__E) >= 2 * two_period_block:
        prev_two = velocities_E__E[-2 * two_period_block : -two_period_block].mean(
            axis=0
        )
    else:
        prev_two = last_two
    vx_penalty = max(0.0, 0.5 - last_two[0]) * 10.0
    return (
        abs(float(last_two[2]))
        + 0.75 * abs(float(last_two[2] - prev_two[2]))
        + vx_penalty
    )


def search_glide_start(
    output_root: Path,
    search_speed_values_mps: tuple[float, ...],
    search_alpha_values_deg: tuple[float, ...],
    search_prescribed_steps: int,
    search_free_steps: int,
    steps_per_reference_period: int,
    prescribed_wake: bool,
) -> dict[str, Any]:
    """Search for a reasonable fixed-pitch glide-start condition."""
    candidates: list[dict[str, Any]] = []
    for initial_speed_mps in search_speed_values_mps:
        for initial_alpha_deg in search_alpha_values_deg:
            candidate: dict[str, Any] = {
                "initial_speed_mps": initial_speed_mps,
                "initial_alpha_deg": initial_alpha_deg,
            }
            try:
                coupled_problem, coupled_solver = build_problem(
                    initial_speed_mps=initial_speed_mps,
                    initial_alpha_deg=initial_alpha_deg,
                    prescribed_num_steps=search_prescribed_steps,
                    free_num_steps=search_free_steps,
                    steps_per_reference_period=steps_per_reference_period,
                    fix_pitch_rotation=True,
                    fixed_pitch_deg=initial_alpha_deg,
                )
                coupled_solver.run(
                    prescribed_wake=prescribed_wake,
                    show_progress=False,
                )
                _, velocities_E__E, _, alphas_deg, euler_angles_deg = (
                    ff_utils.get_history_arrays(coupled_problem)
                )
                candidate["score"] = score_search_case(
                    velocities_E__E=velocities_E__E,
                    steps_per_reference_period=steps_per_reference_period,
                )
                candidate["last_2_reference_period_mean_velocity_E_mps"] = (
                    velocities_E__E[-2 * steps_per_reference_period :]
                    .mean(axis=0)
                    .tolist()
                )
                candidate["last_reference_period_mean_alpha_deg"] = float(
                    alphas_deg[-steps_per_reference_period:].mean()
                )
                candidate["pitch_span_deg"] = float(np.ptp(euler_angles_deg[:, 1]))
                candidate["status"] = "ok"
            except Exception as exc:
                candidate["status"] = "fail"
                candidate["error"] = repr(exc)
                candidate["score"] = float("inf")
            candidates.append(candidate)

    valid_candidates = [
        candidate for candidate in candidates if candidate["status"] == "ok"
    ]
    valid_candidates.sort(key=lambda candidate: float(candidate["score"]))
    if not valid_candidates:
        raise RuntimeError("No valid glide-start candidate was found.")

    search_summary = {
        "search_prescribed_steps": search_prescribed_steps,
        "search_free_steps": search_free_steps,
        "steps_per_reference_period": steps_per_reference_period,
        "search_speed_values_mps": list(search_speed_values_mps),
        "search_alpha_values_deg": list(search_alpha_values_deg),
        "best_candidate": valid_candidates[0],
        "top_candidates": valid_candidates[:5],
        "all_candidates": candidates,
    }
    ff_utils.write_json(output_root / "glide_start_search.json", search_summary)
    return valid_candidates[0]


def build_summary(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
    steps_per_reference_period: int,
    pitch_mode: str,
    initial_speed_mps: float,
    initial_alpha_deg: float,
    search_result: dict[str, Any] | None,
    kinetic_energy_j: np.ndarray,
    potential_energy_j: np.ndarray,
    total_energy_j: np.ndarray,
    prescribed_wake: bool,
) -> dict[str, Any]:
    """Build a machine-readable summary for one gliding-wing run."""
    times_s, velocities_E__E, _, alphas_deg, euler_angles_deg = (
        ff_utils.get_history_arrays(coupled_problem)
    )
    summary: dict[str, Any] = {
        "case": "01_gliding_wing",
        "geometry_representation": "single_three_station_wing_no_center_gap",
        "pitch_mode": pitch_mode,
        "wake_model": "prescribed" if prescribed_wake else "free",
        "weight_n": TOTAL_AIRCRAFT_WEIGHT_N,
        "fixed_pitch_deg": initial_alpha_deg,
        "delta_time_s": coupled_problem.delta_time,
        "reference_period_s": REFERENCE_PERIOD_S,
        "steps_per_reference_period": steps_per_reference_period,
        "prescribed_steps": coupled_problem.coupled_movement.prescribed_num_steps,
        "free_steps": coupled_problem.coupled_movement.free_num_steps,
        "initial_speed_mps": initial_speed_mps,
        "initial_alpha_deg": initial_alpha_deg,
        "final_position_E_m": coupled_solver.stackPosition_E_E[-1].tolist(),
        "final_velocity_E_mps": velocities_E__E[-1].tolist(),
        "initial_kinetic_energy_j": float(kinetic_energy_j[0]),
        "final_kinetic_energy_j": float(kinetic_energy_j[-1]),
        "initial_potential_energy_j": float(potential_energy_j[0]),
        "final_potential_energy_j": float(potential_energy_j[-1]),
        "initial_total_energy_j": float(total_energy_j[0]),
        "final_total_energy_j": float(total_energy_j[-1]),
        "total_energy_drift_j": float(total_energy_j[-1] - total_energy_j[0]),
    }
    if search_result is not None:
        summary["glide_start_search_best_score"] = float(search_result["score"])
    ff_utils.add_block_summaries(
        summary=summary,
        velocities_E__E=velocities_E__E,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        steps_per_block=steps_per_reference_period,
        block_name="reference_period",
    )
    return summary


def get_full_position_history_E_E(
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
) -> np.ndarray:
    """Return the stored position history with the final MuJoCo state appended."""
    final_position_E_E = np.asarray(
        coupled_solver.mujoco_model.get_state()["position_E_E"], dtype=float
    )
    return np.vstack((coupled_solver.stackPosition_E_E, final_position_E_E))


def run_variant(
    output_dir: Path,
    pitch_mode: str,
    initial_speed_mps: float,
    initial_alpha_deg: float,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_reference_period: int,
    animate: bool,
    show_wake_vortices: bool,
    show_progress: bool,
    search_result: dict[str, Any] | None,
    prescribed_wake: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    history_save_dir: Path | None,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Run one gliding-wing pitch variant and save its outputs."""
    fix_pitch_rotation = pitch_mode == "fixed"
    coupled_problem, coupled_solver = build_problem(
        initial_speed_mps=initial_speed_mps,
        initial_alpha_deg=initial_alpha_deg,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_reference_period=steps_per_reference_period,
        fix_pitch_rotation=fix_pitch_rotation,
        fixed_pitch_deg=initial_alpha_deg,
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
    positions_E_E = get_full_position_history_E_E(coupled_solver)
    _, kinetic_energy_j, potential_energy_j, total_energy_j = (
        ff_utils.compute_energy_histories(
            positions_E_E=positions_E_E,
            velocities_E__E=velocities_E__E,
            weight_n=TOTAL_AIRCRAFT_WEIGHT_N,
            gravity_E=GRAVITY_E,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = ff_utils.save_velocity_history_plot(
        x_values=times_s,
        velocities_E__E=velocities_E__E,
        speeds_mps=speeds_mps,
        alphas_deg=alphas_deg,
        euler_angles_deg=euler_angles_deg,
        save_path=output_dir / "velocity_history.png",
        x_label="Time (s)",
        title=f"Geometry-Matched Gliding Wing ({pitch_mode.replace('_', ' ').title()})",
    )
    energy_plot_path = ff_utils.save_energy_history_plot(
        x_values=times_s,
        kinetic_energy_j=kinetic_energy_j,
        potential_energy_j=potential_energy_j,
        total_energy_j=total_energy_j,
        save_path=output_dir / "energy_history.png",
        x_label="Time (s)",
        title=f"Energy History ({pitch_mode.replace('_', ' ').title()})",
    )

    summary_path = ff_utils.write_json(
        output_dir / "summary.json",
        build_summary(
            coupled_problem=coupled_problem,
            coupled_solver=coupled_solver,
            steps_per_reference_period=steps_per_reference_period,
            pitch_mode=pitch_mode,
            initial_speed_mps=initial_speed_mps,
            initial_alpha_deg=initial_alpha_deg,
            search_result=search_result,
            kinetic_energy_j=kinetic_energy_j,
            potential_energy_j=potential_energy_j,
            total_energy_j=total_energy_j,
            prescribed_wake=prescribed_wake,
        ),
    )

    print(f"Saved summary to: {summary_path}")
    print(f"Saved velocity plot to: {plot_path}")
    print(f"Saved energy plot to: {energy_plot_path}")

    if animate:
        webp_path, mp4_path = ff_utils.save_animation_bundle(
            coupled_solver=coupled_solver,
            output_dir=output_dir,
            filename_stem=f"AnimateFreeFlight_gliding_wing_{pitch_mode}_with_wake",
            show_wake_vortices=show_wake_vortices,
        )
        print(f"Saved animation to: {webp_path}")
        print(f"Saved movie to: {mp4_path}")

    return coupled_problem, coupled_solver


def run_case(
    output_root: Path,
    pitch_mode: str,
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_reference_period: int,
    animate: bool,
    show_wake_vortices: bool,
    show_progress: bool,
    search_glide_start_enabled: bool,
    search_prescribed_steps: int,
    search_free_steps: int,
    initial_speed_mps: float,
    initial_alpha_deg: float,
    prescribed_wake: bool,
    history_stride: int,
    save_every_n_steps: int | None,
    history_save_dir: Path | None,
) -> None:
    """Run the requested gliding-wing variants and save organized outputs."""
    output_root.mkdir(parents=True, exist_ok=True)
    search_result: dict[str, Any] | None = None
    if search_glide_start_enabled:
        search_result = search_glide_start(
            output_root=output_root,
            search_speed_values_mps=DEFAULT_SEARCH_SPEEDS_MPS,
            search_alpha_values_deg=DEFAULT_SEARCH_ALPHAS_DEG,
            search_prescribed_steps=search_prescribed_steps,
            search_free_steps=search_free_steps,
            steps_per_reference_period=steps_per_reference_period,
            prescribed_wake=prescribed_wake,
        )
        initial_speed_mps = float(search_result["initial_speed_mps"])
        initial_alpha_deg = float(search_result["initial_alpha_deg"])
        print(
            "Selected glide start:",
            f"speed={initial_speed_mps:.3f} m/s, alpha={initial_alpha_deg:.3f} deg",
        )

    pitch_modes = ("fixed", "free") if pitch_mode == "both" else (pitch_mode,)
    for variant_pitch_mode in pitch_modes:
        variant_history_dir: Path | None = None
        if history_save_dir is not None:
            variant_history_dir = (
                history_save_dir / f"{variant_pitch_mode}_pitch_history_snapshots"
            )
        run_variant(
            output_dir=output_root / f"{variant_pitch_mode}_pitch",
            pitch_mode=variant_pitch_mode,
            initial_speed_mps=initial_speed_mps,
            initial_alpha_deg=initial_alpha_deg,
            prescribed_num_steps=prescribed_num_steps,
            free_num_steps=free_num_steps,
            steps_per_reference_period=steps_per_reference_period,
            animate=animate,
            show_wake_vortices=show_wake_vortices,
            show_progress=show_progress,
            search_result=search_result,
            prescribed_wake=prescribed_wake,
            history_stride=history_stride,
            save_every_n_steps=save_every_n_steps,
            history_save_dir=variant_history_dir,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the geometry-matched gliding-wing free-flight cases."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where the case outputs should be organized.",
    )
    parser.add_argument(
        "--pitch-mode",
        choices=("fixed", "free", "both"),
        default="both",
        help="Which pitch variants to run.",
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
        "--steps-per-reference-period",
        type=int,
        default=DEFAULT_STEPS_PER_REFERENCE_PERIOD,
        help="Temporal resolution in solver steps per reference period.",
    )
    parser.add_argument(
        "--search-glide-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search for a short-run glide-start condition before the production runs.",
    )
    parser.add_argument(
        "--search-prescribed-steps",
        type=int,
        default=DEFAULT_SEARCH_PRESCRIBED_STEPS,
        help="Prescribed steps per candidate during the glide-start search.",
    )
    parser.add_argument(
        "--search-free-steps",
        type=int,
        default=DEFAULT_SEARCH_FREE_STEPS,
        help="Free-flight steps per candidate during the glide-start search.",
    )
    parser.add_argument(
        "--initial-speed-mps",
        type=float,
        default=DEFAULT_INITIAL_SPEED_MPS,
        help="Initial speed used when the glide-start search is disabled.",
    )
    parser.add_argument(
        "--initial-alpha-deg",
        type=float,
        default=DEFAULT_INITIAL_ALPHA_DEG,
        help="Initial alpha used when the glide-start search is disabled.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and save wake-visible movies after the solve.",
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
        output_root=args.output_root,
        pitch_mode=args.pitch_mode,
        prescribed_num_steps=args.prescribed_steps,
        free_num_steps=args.free_steps,
        steps_per_reference_period=args.steps_per_reference_period,
        animate=args.animate,
        show_wake_vortices=args.show_wake_vortices,
        show_progress=args.show_progress,
        search_glide_start_enabled=args.search_glide_start,
        search_prescribed_steps=args.search_prescribed_steps,
        search_free_steps=args.search_free_steps,
        initial_speed_mps=args.initial_speed_mps,
        initial_alpha_deg=args.initial_alpha_deg,
        prescribed_wake=args.prescribed_wake,
        history_stride=args.history_stride,
        save_every_n_steps=args.save_every_n_steps,
        history_save_dir=args.history_save_dir,
    )


if __name__ == "__main__":
    main()
