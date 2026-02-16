"""Diagnose the NC=13, AR=1 prescribed wake divergence.

Runs the UVLM solver for NC=12, 13, 14 at AR=1 with WL=1, dumping per-time-step
diagnostics: influence matrix condition number, min collocation-to-bound-vortex
distances, and vortex strength statistics.

Usage:
    python diagnose_nc13.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import json

import dxf_to_csv
import numpy as np

import pterasoftware as ps
from pterasoftware.convergence.unsteady_non_trapezoidal import (
    _get_num_cross_sections_for_panel_ar,
)

import configs

# ── Parameters ────────────────────────────────────────────────────────────────
TARGET_AR = 1
CHORDWISE_VALUES = [12, 13, 14]
CONFIG_NAME = "L170V_R180V_170Hz"
NUM_CYCLES = 1
# ──────────────────────────────────────────────────────────────────────────────


def build_problem(nc: int) -> ps.problems.UnsteadyProblem:
    """Build an UnsteadyProblem at the given chordwise panel count with AR=1."""
    wing_config = configs.get_config(CONFIG_NAME)
    shared = configs.SHARED_PARAMS

    velocity = float(shared["velocity"])
    alpha = float(shared["alpha"])
    flapping_frequency = float(shared["flapping_frequency"])
    wing_spacing = float(shared["wing_spacing"])
    x_offset = shared["x_offset"]
    y_offset = shared["y_offset"]
    chordwise_spacing = str(shared["chordwise_spacing"])

    flapping_period = 1.0 / flapping_frequency
    dxf_filepath = Path(__file__).parent / "gammabot_approximate_wing.dxf"

    # Compute num_sections for AR=1.
    ref_data = dxf_to_csv.process_dxf_to_wing_section_data(str(dxf_filepath), 8)
    span = float(np.sum(np.sqrt(np.sum(ref_data[1:, :3] ** 2, axis=1))))
    avg_chord = float(np.mean(ref_data[:, 3]))
    num_sections = _get_num_cross_sections_for_panel_ar(span, avg_chord, TARGET_AR, nc)

    # Get optimized dt from cache.
    cache_path = Path(__file__).parent / "convergence_cache" / f"{CONFIG_NAME}.json"
    with open(cache_path, "r") as f:
        cache_data = json.load(f)
    dt_cache = cache_data.get("_optimized_dt", {})
    dt_key = f"{NUM_CYCLES},{nc}"
    delta_time = dt_cache.get(dt_key, flapping_period / 30)
    print(f"  Using delta_time={delta_time:.6e} (key={dt_key})")

    wing_section_data = dxf_to_csv.process_dxf_to_wing_section_data(
        str(dxf_filepath), num_sections
    )
    num_wcs = num_sections + 1

    def make_cross_sections():
        cross_sections = []
        for i in range(num_wcs):
            num_spanwise_panels = 1 if i < num_sections else None
            wcs = ps.geometry.wing_cross_section.WingCrossSection(
                Lp_Wcsp_Lpp=tuple(wing_section_data[i, :3]),
                angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                chord=float(wing_section_data[i, 3]),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                num_spanwise_panels=num_spanwise_panels,
            )
            cross_sections.append(wcs)
        return cross_sections

    left_params = wing_config["left"]
    right_params = wing_config["right"]

    def compute_flapping(params):
        phi_max = params["phi_max"]
        psi_max = params["psi_max"]
        delta = params["delta"]
        return {
            "amp_x": phi_max,
            "amp_y": psi_max,
            "period_x": flapping_period if phi_max != 0.0 else 0.0,
            "period_y": flapping_period if psi_max != 0.0 else 0.0,
            "phase_y": (90.0 + delta) if psi_max != 0.0 else 0.0,
        }

    left_flap = compute_flapping(left_params)
    right_flap = compute_flapping(right_params)

    airplane = ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=make_cross_sections(),
                Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
                angles_Gs_to_Wn_ixyz=(
                    left_params["phi_v_shift"], left_params["psi_v_shift"], 0.0,
                ),
                symmetric=False, mirror_only=False,
                num_chordwise_panels=nc, chordwise_spacing=chordwise_spacing,
            ),
            ps.geometry.wing.Wing(
                wing_cross_sections=make_cross_sections(),
                Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
                angles_Gs_to_Wn_ixyz=(
                    right_params["phi_v_shift"], right_params["psi_v_shift"], 0.0,
                ),
                symmetric=False, mirror_only=True,
                symmetryNormal_G=(0, 1, 0), symmetryPoint_G_Cg=(0, 0, 0),
                num_chordwise_panels=nc, chordwise_spacing=chordwise_spacing,
            ),
        ],
        name="GammaBot",
    )

    # Wing cross section movements.
    def make_wcs_movements(wing):
        return [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wcs,
            )
            for wcs in wing.wing_cross_sections
        ]

    left_wcsm = make_wcs_movements(airplane.wings[0])
    right_wcsm = make_wcs_movements(airplane.wings[1])

    left_wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[0],
        wing_cross_section_movements=left_wcsm,
        rotationPointOffset_Gs_Ler=(x_offset, y_offset, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(left_flap["amp_x"], left_flap["amp_y"], 0.0),
        periodAngles_Gs_to_Wn_ixyz=(left_flap["period_x"], left_flap["period_y"], 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, left_flap["phase_y"], 0.0),
    )
    right_wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[1],
        wing_cross_section_movements=right_wcsm,
        rotationPointOffset_Gs_Ler=(x_offset, y_offset, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(right_flap["amp_x"], right_flap["amp_y"], 0.0),
        periodAngles_Gs_to_Wn_ixyz=(right_flap["period_x"], right_flap["period_y"], 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, right_flap["phase_y"], 0.0),
    )

    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[left_wm, right_wm],
    )

    operating_point = ps.operating_point.OperatingPoint(vCg__E=velocity, alpha=alpha)
    op_movement = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=operating_point,
    )

    movement = ps.movements.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=op_movement,
        num_cycles=NUM_CYCLES,
        delta_time=delta_time,
    )

    return ps.problems.UnsteadyProblem(movement=movement)


def run_with_diagnostics(nc: int) -> None:
    """Run the solver for a given NC and print per-step diagnostics."""
    print(f"\n{'=' * 70}")
    print(f"NC = {nc}, AR = {TARGET_AR}, WL = {NUM_CYCLES} (prescribed wake)")
    print(f"{'=' * 70}")

    problem = build_problem(nc)
    solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem
    )

    print(f"  num_panels: {solver.num_panels}")
    print(f"  num_steps:  {solver.num_steps}")

    # Monkey-patch _calculate_vortex_strengths to capture diagnostics.
    original_calc = solver._calculate_vortex_strengths

    step_diagnostics = []

    def patched_calc():
        A = solver._currentGridWingWingInfluences__E
        b = (
            -solver._currentStackWakeWingInfluences__E
            - solver._currentStackFreestreamWingInfluences__E
        )

        # Condition number (use 2-norm).
        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cond = float("inf")

        # Matrix stats.
        a_min = np.min(np.abs(A))
        a_max = np.max(np.abs(A))
        a_diag_min = np.min(np.abs(np.diag(A)))
        a_diag_max = np.max(np.abs(np.diag(A)))

        # Min distance between collocation points across wings.
        cpp = solver.stackCpp_GP1_CgP1
        half = len(cpp) // 2
        if half > 0:
            cpp_left = cpp[:half]
            cpp_right = cpp[half:]
            cross_diffs = cpp_left[:, np.newaxis, :] - cpp_right[np.newaxis, :, :]
            cross_dists = np.linalg.norm(cross_diffs, axis=2)
            min_cross_dist = float(np.min(cross_dists))
        else:
            min_cross_dist = float("inf")

        # RHS stats.
        b_min = np.min(np.abs(b))
        b_max = np.max(np.abs(b))

        # Now call the original to solve.
        original_calc()

        # Solution stats.
        x = solver._current_bound_vortex_strengths
        x_min = np.min(x)
        x_max = np.max(x)
        x_absmax = np.max(np.abs(x))

        step_diagnostics.append({
            "step": solver._current_step,
            "cond": cond,
            "A_absmin": a_min,
            "A_absmax": a_max,
            "A_diag_absmin": a_diag_min,
            "A_diag_absmax": a_diag_max,
            "b_absmin": b_min,
            "b_absmax": b_max,
            "x_min": x_min,
            "x_max": x_max,
            "x_absmax": x_absmax,
            "min_cross_wing_dist": min_cross_dist,
        })

    solver._calculate_vortex_strengths = patched_calc

    # Run the solver.
    t0 = time.time()
    solver.run(prescribed_wake=True, show_progress=False)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f} s")

    # Print diagnostics table.
    print(f"\n  {'Step':>4} {'Cond#':>12} {'|A|min':>12} {'|A|max':>12} "
          f"{'|diag|min':>12} {'x_min':>12} {'x_max':>12} {'|x|max':>12} "
          f"{'|b|max':>12} {'CrossDist':>12}")
    print(f"  {'-' * 136}")

    for d in step_diagnostics:
        print(
            f"  {d['step']:>4} {d['cond']:>12.4e} {d['A_absmin']:>12.4e} "
            f"{d['A_absmax']:>12.4e} {d['A_diag_absmin']:>12.4e} "
            f"{d['x_min']:>12.4e} {d['x_max']:>12.4e} {d['x_absmax']:>12.4e} "
            f"{d['b_absmax']:>12.4e} {d['min_cross_wing_dist']:>12.4e}"
        )

    # Summary.
    conds = [d["cond"] for d in step_diagnostics]
    x_absmaxes = [d["x_absmax"] for d in step_diagnostics]
    print(f"\n  Condition number: min={min(conds):.4e}, max={max(conds):.4e}")
    print(f"  |x| max:         min={min(x_absmaxes):.4e}, max={max(x_absmaxes):.4e}")

    # Flag steps where things look bad.
    bad_steps = [d for d in step_diagnostics if d["cond"] > 1e14 or d["x_absmax"] > 1e6]
    if bad_steps:
        print(f"\n  WARNING: {len(bad_steps)} steps with cond > 1e14 or |x| > 1e6!")
        for d in bad_steps[:5]:
            print(f"    Step {d['step']}: cond={d['cond']:.4e}, |x|max={d['x_absmax']:.4e}")
    else:
        print(f"\n  All steps look healthy.")


def main() -> None:
    for nc in CHORDWISE_VALUES:
        run_with_diagnostics(nc)


if __name__ == "__main__":
    main()
