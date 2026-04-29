"""Phase-0 multibody validation with two far-separated gliding wings.

This is the first multibody validation slice. It does not yet couple two rigid
bodies to MuJoCo. Instead, it uses the existing multi-airplane steady aerodynamic
path to verify that two identical gliding wings placed far apart reproduce the
single-wing result airplane-by-airplane when aerodynamic interference is negligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from examples import free_flight_gliding_wing as glider_case
from pterasoftware import _transformations

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "phase0_multibody_validation"
    / "two_gliding_wings_far_apart"
)
DEFAULT_SEPARATION_M = 5.0
DEFAULT_SPEED_MPS = 3.5
DEFAULT_ALPHA_DEG = 5.0
DEFAULT_BETA_DEG = 0.0


def build_validation_operating_point(
    speed_mps: float,
    alpha_deg: float,
    beta_deg: float,
) -> ps.operating_point.OperatingPoint:
    """Build the steady operating point used for the far-field validation."""
    return ps.operating_point.OperatingPoint(
        rho=glider_case.AIR_DENSITY,
        vCg__E=speed_mps,
        alpha=alpha_deg,
        beta=beta_deg,
        nu=glider_case.KINEMATIC_VISCOSITY,
    )


def build_single_glider_problem(
    speed_mps: float,
    alpha_deg: float,
    beta_deg: float,
) -> ps.problems.SteadyProblem:
    """Build the single-glider baseline problem."""
    return ps.problems.SteadyProblem(
        airplanes=[glider_case.build_airplane()],
        operating_point=build_validation_operating_point(
            speed_mps=speed_mps,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
        ),
    )


def build_two_glider_problem(
    speed_mps: float,
    alpha_deg: float,
    beta_deg: float,
    separation_m: float,
) -> ps.problems.SteadyProblem:
    """Build the two-glider far-separation validation problem.

    The second glider is translated laterally in GP1 by ``separation_m`` so the
    pair sees negligible aerodynamic interference.
    """
    if separation_m <= 0.0:
        raise ValueError("separation_m must be greater than 0.")

    lead_airplane = glider_case.build_airplane()
    trailing_airplane = lead_airplane.deep_copy_with_Cg_GP1_CgP1(
        np.array([0.0, separation_m, 0.0], dtype=float)
    )
    return ps.problems.SteadyProblem(
        airplanes=[lead_airplane, trailing_airplane],
        operating_point=build_validation_operating_point(
            speed_mps=speed_mps,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
        ),
    )


def run_solver(
    problem: ps.problems.SteadyProblem,
) -> ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver:
    """Run the steady ring-vortex solver on a problem."""
    solver = ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver(
        problem
    )
    solver.run()
    return solver


def _relative_error_vector(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    """Return a component-wise relative error with a safe small-value guard."""
    denom = np.maximum(np.abs(reference), 1e-12)
    return (np.abs(candidate - reference) / denom).astype(float).tolist()


def _relative_error_norm(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the vector relative error norm with a safe small-value guard."""
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    return float(np.linalg.norm(candidate - reference) / denom)


def summarize_validation(
    single_solver: ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver,
    pair_solver: ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver,
    separation_m: float,
    speed_mps: float,
    alpha_deg: float,
    beta_deg: float,
) -> dict[str, Any]:
    """Build a machine-readable summary of the far-field validation."""
    single_airplane = single_solver.airplanes[0]
    pair_airplanes = pair_solver.airplanes
    pair_operating_point = pair_solver.operating_point
    reference_force_coefficients = np.array(
        single_airplane.forceCoefficients_W, dtype=float
    )
    reference_moment_coefficients = np.array(
        single_airplane.momentCoefficients_W_CgP1, dtype=float
    )
    reference_forces = np.array(single_airplane.forces_W, dtype=float)
    reference_moments = np.array(single_airplane.moments_W_CgP1, dtype=float)

    pair_summaries: list[dict[str, Any]] = []
    max_force_coeff_error = 0.0
    max_moment_coeff_error = 0.0

    for airplane_id, airplane in enumerate(pair_airplanes):
        candidate_force_coefficients = np.array(
            airplane.forceCoefficients_W, dtype=float
        )
        candidate_forces = np.array(airplane.forces_W, dtype=float)

        # Existing multi-airplane steady loads are reported about the first
        # airplane's CG. For like-for-like comparison we convert the pair-case
        # moments to each airplane's own CG before forming coefficients.
        candidate_forces_gp1 = _transformations.apply_T_to_vectors(
            pair_operating_point.T_pas_W_CgP1_to_GP1_CgP1,
            candidate_forces,
            has_point=False,
        )
        candidate_moments_gp1_about_first = _transformations.apply_T_to_vectors(
            pair_operating_point.T_pas_W_CgP1_to_GP1_CgP1,
            np.array(airplane.moments_W_CgP1, dtype=float),
            has_point=True,
        )
        candidate_moments_gp1_about_own = candidate_moments_gp1_about_first - np.cross(
            airplane.Cg_GP1_CgP1, candidate_forces_gp1
        )
        candidate_moments = _transformations.apply_T_to_vectors(
            pair_operating_point.T_pas_GP1_CgP1_to_W_CgP1,
            candidate_moments_gp1_about_own,
            has_point=True,
        )
        candidate_moment_coefficients = np.array(
            [
                candidate_moments[0]
                / pair_operating_point.qInf__E
                / airplane.s_ref
                / airplane.b_ref,
                candidate_moments[1]
                / pair_operating_point.qInf__E
                / airplane.s_ref
                / airplane.c_ref,
                candidate_moments[2]
                / pair_operating_point.qInf__E
                / airplane.s_ref
                / airplane.b_ref,
            ],
            dtype=float,
        )

        force_coeff_error = _relative_error_norm(
            reference_force_coefficients, candidate_force_coefficients
        )
        moment_coeff_error = _relative_error_norm(
            reference_moment_coefficients, candidate_moment_coefficients
        )
        max_force_coeff_error = max(max_force_coeff_error, force_coeff_error)
        max_moment_coeff_error = max(max_moment_coeff_error, moment_coeff_error)

        pair_summaries.append(
            {
                "airplane_id": airplane_id,
                "cg_gp1_cgp1_m": airplane.Cg_GP1_CgP1.tolist(),
                "forces_w_n": candidate_forces.tolist(),
                "moments_w_about_own_cg_nm": candidate_moments.tolist(),
                "force_coefficients_w": candidate_force_coefficients.tolist(),
                "moment_coefficients_w_about_own_cg": candidate_moment_coefficients.tolist(),
                "force_coefficients_relative_error_norm_vs_single": force_coeff_error,
                "moment_coefficients_relative_error_norm_vs_single": moment_coeff_error,
                "forces_relative_error_norm_vs_single": _relative_error_norm(
                    reference_forces, candidate_forces
                ),
                "moments_relative_error_norm_vs_single": _relative_error_norm(
                    reference_moments, candidate_moments
                ),
            }
        )

    return {
        "case": "phase0_multibody_validation_two_gliding_wings_far_apart",
        "description": (
            "Two identical gliding wings are solved together at large lateral "
            "separation and compared against the single-glider baseline."
        ),
        "separation_direction_gp1": [0.0, 1.0, 0.0],
        "separation_m": separation_m,
        "speed_mps": speed_mps,
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "single_glider": {
            "forces_w_n": reference_forces.tolist(),
            "moments_w_about_own_cg_nm": reference_moments.tolist(),
            "force_coefficients_w": reference_force_coefficients.tolist(),
            "moment_coefficients_w_about_own_cg": reference_moment_coefficients.tolist(),
        },
        "pair_gliders": pair_summaries,
        "max_force_coefficients_relative_error_norm_vs_single": max_force_coeff_error,
        "max_moment_coefficients_relative_error_norm_vs_single": max_moment_coeff_error,
        "passes_far_field_equivalence_check": (
            max_force_coeff_error < 1e-3 and max_moment_coeff_error < 1e-3
        ),
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the validation summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "summary.json"
    save_path.write_text(json.dumps(summary, indent=2))
    return save_path


def run_validation(
    output_dir: Path,
    separation_m: float,
    speed_mps: float,
    alpha_deg: float,
    beta_deg: float,
) -> dict[str, Any]:
    """Run the far-field multibody validation and save its summary."""
    single_solver = run_solver(
        build_single_glider_problem(
            speed_mps=speed_mps,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
        )
    )
    pair_solver = run_solver(
        build_two_glider_problem(
            speed_mps=speed_mps,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            separation_m=separation_m,
        )
    )
    summary = summarize_validation(
        single_solver=single_solver,
        pair_solver=pair_solver,
        separation_m=separation_m,
        speed_mps=speed_mps,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
    )
    write_summary(output_dir=output_dir, summary=summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the first multibody slice by checking that two far-separated "
            "gliding wings reproduce the single-glider steady loads."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the validation summary should be saved.",
    )
    parser.add_argument(
        "--separation-m",
        type=float,
        default=DEFAULT_SEPARATION_M,
        help="Lateral GP1 separation between the two gliders in meters.",
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=DEFAULT_SPEED_MPS,
        help="Steady flight speed for the validation operating point.",
    )
    parser.add_argument(
        "--alpha-deg",
        type=float,
        default=DEFAULT_ALPHA_DEG,
        help="Steady angle of attack for the validation operating point.",
    )
    parser.add_argument(
        "--beta-deg",
        type=float,
        default=DEFAULT_BETA_DEG,
        help="Steady sideslip angle for the validation operating point.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the phase-0 multibody validation."""
    args = parse_args()
    summary = run_validation(
        output_dir=args.output_dir,
        separation_m=args.separation_m,
        speed_mps=args.speed_mps,
        alpha_deg=args.alpha_deg,
        beta_deg=args.beta_deg,
    )
    summary_path = args.output_dir / "summary.json"
    print(f"Saved summary to: {summary_path}")
    print(
        "Max relative coefficient errors vs single:",
        (
            f"force={summary['max_force_coefficients_relative_error_norm_vs_single']:.3e}, "
            f"moment={summary['max_moment_coefficients_relative_error_norm_vs_single']:.3e}"
        ),
    )
    print(
        "Far-field equivalence check:",
        summary["passes_far_field_equivalence_check"],
    )


if __name__ == "__main__":
    main()
