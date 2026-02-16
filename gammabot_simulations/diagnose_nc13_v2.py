"""Deeper diagnosis of the AR=1 prescribed wake divergence.

Focuses on:
1. Step 0 behavior (no wake) - |b| vs |x| comparison across NC values
2. Spatial pattern of vortex strengths (sign oscillation vs monotonic growth)
3. Row/column scaling of the influence matrix beyond 2-norm condition number
4. NC=15 hidden divergence check

Usage:
    python diagnose_nc13_v2.py
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
CHORDWISE_VALUES = [12, 13, 14, 15]
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

    ref_data = dxf_to_csv.process_dxf_to_wing_section_data(str(dxf_filepath), 8)
    span = float(np.sum(np.sqrt(np.sum(ref_data[1:, :3] ** 2, axis=1))))
    avg_chord = float(np.mean(ref_data[:, 3]))
    num_sections = _get_num_cross_sections_for_panel_ar(span, avg_chord, TARGET_AR, nc)

    cache_path = Path(__file__).parent / "convergence_cache" / f"{CONFIG_NAME}.json"
    with open(cache_path, "r") as f:
        cache_data = json.load(f)
    dt_cache = cache_data.get("_optimized_dt", {})
    dt_key = f"{NUM_CYCLES},{nc}"
    # NC=15 has no optimized dt for wake_length=1, but other wake lengths
    # all use 5.823762e-05 (dt is independent of wake length and AR).
    _DT_OVERRIDES = {"1,15": 5.823762e-05}
    delta_time = dt_cache.get(dt_key, _DT_OVERRIDES.get(dt_key, flapping_period / 30))

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

    wing_config_data = configs.get_config(CONFIG_NAME)
    left_params = wing_config_data["left"]
    right_params = wing_config_data["right"]

    def compute_flapping(params):
        phi_max = params["phi_max"]
        psi_max = params["psi_max"]
        delta = params["delta"]
        return {
            "amp_x": phi_max, "amp_y": psi_max,
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

    return ps.problems.UnsteadyProblem(movement=movement), num_sections, delta_time


def analyze_spatial_pattern(x: np.ndarray, num_chordwise: int) -> dict:
    """Analyze spatial pattern of vortex strengths for one wing.

    :param x: 1D array of vortex strengths for one wing.
    :param num_chordwise: Number of chordwise panels.
    :return: Dictionary with spatial analysis metrics.
    """
    n = len(x)
    num_spanwise = n // num_chordwise

    # Reshape into (spanwise, chordwise) grid.
    grid = x.reshape(num_spanwise, num_chordwise)

    # Check sign changes along spanwise direction (for each chordwise column).
    spanwise_sign_changes = 0
    for col in range(num_chordwise):
        column = grid[:, col]
        signs = np.sign(column)
        changes = np.sum(np.abs(np.diff(signs)) > 0)
        spanwise_sign_changes += changes

    avg_spanwise_sign_changes = spanwise_sign_changes / num_chordwise

    # Check sign changes along chordwise direction (for each spanwise row).
    chordwise_sign_changes = 0
    for row in range(num_spanwise):
        row_data = grid[row, :]
        signs = np.sign(row_data)
        changes = np.sum(np.abs(np.diff(signs)) > 0)
        chordwise_sign_changes += changes

    avg_chordwise_sign_changes = chordwise_sign_changes / num_spanwise

    # Max absolute strength by spanwise position (averaged over chordwise).
    spanwise_max = np.max(np.abs(grid), axis=1)

    return {
        "avg_spanwise_sign_changes": avg_spanwise_sign_changes,
        "avg_chordwise_sign_changes": avg_chordwise_sign_changes,
        "num_spanwise": num_spanwise,
        "num_chordwise": num_chordwise,
        "spanwise_max": spanwise_max,
    }


def analyze_matrix_scaling(A: np.ndarray) -> dict:
    """Analyze row/column scaling of the influence matrix.

    :param A: The influence coefficient matrix.
    :return: Dictionary with scaling analysis metrics.
    """
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)

    # Row scaling ratio.
    row_min = np.min(row_norms[row_norms > 0]) if np.any(row_norms > 0) else 0.0
    row_max = np.max(row_norms)
    row_ratio = row_max / row_min if row_min > 0 else float("inf")

    # Column scaling ratio.
    col_min = np.min(col_norms[col_norms > 0]) if np.any(col_norms > 0) else 0.0
    col_max = np.max(col_norms)
    col_ratio = col_max / col_min if col_min > 0 else float("inf")

    # Diagonal dominance: ratio of |diag| to sum of off-diagonal |entries| per row.
    diag = np.abs(np.diag(A))
    off_diag_sum = np.sum(np.abs(A), axis=1) - diag
    dominance = diag / (off_diag_sum + 1e-300)

    # Check for zero or near-zero rows/columns.
    zero_rows = np.sum(row_norms < 1e-15)
    zero_cols = np.sum(col_norms < 1e-15)

    # Singular values for deeper analysis.
    s = np.linalg.svd(A, compute_uv=False)
    s_nonzero = s[s > 1e-15 * s[0]]
    effective_rank = len(s_nonzero)
    smallest_sv = s[-1] if len(s) > 0 else 0.0
    largest_sv = s[0] if len(s) > 0 else 0.0

    return {
        "row_ratio": row_ratio,
        "col_ratio": col_ratio,
        "row_min_norm": row_min,
        "row_max_norm": row_max,
        "col_min_norm": col_min,
        "col_max_norm": col_max,
        "zero_rows": zero_rows,
        "zero_cols": zero_cols,
        "diag_dominance_min": float(np.min(dominance)),
        "diag_dominance_max": float(np.max(dominance)),
        "diag_dominance_mean": float(np.mean(dominance)),
        "effective_rank": effective_rank,
        "matrix_size": A.shape[0],
        "smallest_sv": smallest_sv,
        "largest_sv": largest_sv,
        "cond_2norm": largest_sv / smallest_sv if smallest_sv > 0 else float("inf"),
        "num_small_sv": int(np.sum(s < 1e-10 * s[0])),
    }


def run_step0_deep_analysis(nc: int) -> None:
    """Deep analysis of step 0 only for a given NC."""
    print(f"\n{'=' * 70}")
    print(f"NC = {nc}, AR = {TARGET_AR} — STEP 0 DEEP ANALYSIS")
    print(f"{'=' * 70}")

    problem, num_sections, dt = build_problem(nc)
    print(f"  num_sections={num_sections}, delta_time={dt:.6e}")

    solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem
    )
    print(f"  num_panels={solver.num_panels}, num_steps={solver.num_steps}")

    # Capture step 0 data.
    step0_data = {}

    original_calc = solver._calculate_vortex_strengths

    def patched_calc():
        A = solver._currentGridWingWingInfluences__E.copy()
        b = (
            -solver._currentStackWakeWingInfluences__E
            - solver._currentStackFreestreamWingInfluences__E
        ).copy()

        step0_data["A"] = A
        step0_data["b"] = b

        original_calc()

        step0_data["x"] = solver._current_bound_vortex_strengths.copy()

    solver._calculate_vortex_strengths = patched_calc

    # Run just step 0 by running the full solver but we only care about step 0.
    # Actually, let's just run the full solver and capture step 0.
    # We'll restore the original after step 0.
    class _Step0Done(Exception):
        pass

    step_count = [0]

    def patched_calc_counted():
        if step_count[0] == 0:
            A = solver._currentGridWingWingInfluences__E.copy()
            b_wake = solver._currentStackWakeWingInfluences__E.copy()
            b_free = solver._currentStackFreestreamWingInfluences__E.copy()
            b = -b_wake - b_free

            step0_data["A"] = A
            step0_data["b"] = b
            step0_data["b_wake"] = b_wake
            step0_data["b_free"] = b_free

        original_calc()

        if step_count[0] == 0:
            step0_data["x"] = solver._current_bound_vortex_strengths.copy()
            # We only need step 0 data — abort the solver early.
            raise _Step0Done()

        step_count[0] += 1

    solver._calculate_vortex_strengths = patched_calc_counted

    t0 = time.time()
    try:
        solver.run(prescribed_wake=True, show_progress=False)
    except _Step0Done:
        pass
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f} s")

    A = step0_data["A"]
    b = step0_data["b"]
    x = step0_data["x"]
    b_wake = step0_data["b_wake"]
    b_free = step0_data["b_free"]

    # ── 1. Step 0 RHS comparison ──
    print(f"\n  -- Step 0: RHS (b) --")
    print(f"    |b|_2:    {np.linalg.norm(b):.6e}")
    print(f"    |b|_inf:  {np.max(np.abs(b)):.6e}")
    print(f"    |b|_1:    {np.sum(np.abs(b)):.6e}")
    print(f"    b_min:    {np.min(b):.6e}")
    print(f"    b_max:    {np.max(b):.6e}")
    print(f"    |b_wake|: {np.max(np.abs(b_wake)):.6e} (should be ~0 at step 0)")
    print(f"    |b_free|: {np.max(np.abs(b_free)):.6e}")

    print(f"\n  -- Step 0: Solution (x) --")
    print(f"    |x|_2:    {np.linalg.norm(x):.6e}")
    print(f"    |x|_inf:  {np.max(np.abs(x)):.6e}")
    print(f"    x_min:    {np.min(x):.6e}")
    print(f"    x_max:    {np.max(x):.6e}")

    # ── 2. Matrix scaling analysis ──
    print(f"\n  -- Step 0: Matrix A scaling --")
    scaling = analyze_matrix_scaling(A)
    print(f"    Size:              {scaling['matrix_size']}x{scaling['matrix_size']}")
    print(f"    Effective rank:    {scaling['effective_rank']}/{scaling['matrix_size']}")
    print(f"    Cond (2-norm):     {scaling['cond_2norm']:.6e}")
    print(f"    Largest SV:        {scaling['largest_sv']:.6e}")
    print(f"    Smallest SV:       {scaling['smallest_sv']:.6e}")
    print(f"    Num SVs < 1e-10*s1:{scaling['num_small_sv']}")
    print(f"    Row norm ratio:    {scaling['row_ratio']:.4e} "
          f"(min={scaling['row_min_norm']:.4e}, max={scaling['row_max_norm']:.4e})")
    print(f"    Col norm ratio:    {scaling['col_ratio']:.4e} "
          f"(min={scaling['col_min_norm']:.4e}, max={scaling['col_max_norm']:.4e})")
    print(f"    Zero rows:         {scaling['zero_rows']}")
    print(f"    Zero cols:         {scaling['zero_cols']}")
    print(f"    Diag dominance:    min={scaling['diag_dominance_min']:.4e}, "
          f"max={scaling['diag_dominance_max']:.4e}, "
          f"mean={scaling['diag_dominance_mean']:.4e}")

    # ── 3. Spatial pattern of vortex strengths ──
    print(f"\n  -- Step 0: Spatial pattern of x --")
    half = len(x) // 2
    left_x = x[:half]
    right_x = x[half:]

    for label, wing_x in [("Left wing", left_x), ("Right wing", right_x)]:
        pattern = analyze_spatial_pattern(wing_x, nc)
        print(f"    {label}:")
        print(f"      Grid: {pattern['num_spanwise']} spanwise x {pattern['num_chordwise']} chordwise")
        print(f"      Avg spanwise sign changes per column: {pattern['avg_spanwise_sign_changes']:.1f}"
              f" / {pattern['num_spanwise'] - 1} possible")
        print(f"      Avg chordwise sign changes per row:   {pattern['avg_chordwise_sign_changes']:.1f}"
              f" / {pattern['num_chordwise'] - 1} possible")

        # Show the spanwise profile of max |x| (first 10 and last 5).
        sm = pattern['spanwise_max']
        n_show = min(10, len(sm))
        print(f"      Spanwise |x| profile (root to tip):")
        print(f"        First {n_show}: {', '.join(f'{v:.4e}' for v in sm[:n_show])}")
        if len(sm) > 15:
            print(f"        Last  5:     {', '.join(f'{v:.4e}' for v in sm[-5:])}")

    # ── 4. Residual check ──
    residual = A @ x - b
    print(f"\n  -- Step 0: Residual ||Ax - b|| --")
    print(f"    |r|_2:    {np.linalg.norm(residual):.6e}")
    print(f"    |r|_inf:  {np.max(np.abs(residual)):.6e}")
    print(f"    |r|/|b|:  {np.linalg.norm(residual) / np.linalg.norm(b):.6e}")

    print()
    sys.stdout.flush()


def main() -> None:
    for nc in CHORDWISE_VALUES:
        run_step0_deep_analysis(nc)


if __name__ == "__main__":
    main()
