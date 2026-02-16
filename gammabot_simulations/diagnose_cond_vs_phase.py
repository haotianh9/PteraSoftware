"""Condition number and smallest SV vs. flap phase for NC=12,13,14,15.

Builds the airplane geometry at each time step in one flapping cycle (no wake,
no solve), constructs the wing-wing influence matrix A, and logs the condition
number and smallest singular value. Runs both the two-wing (original) and
single-wing (left only, no mirror) configurations. Plots all results together.

Usage:
    python diagnose_cond_vs_phase.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import json

import dxf_to_csv
import matplotlib.pyplot as plt
import numpy as np

import pterasoftware as ps
from pterasoftware import (
    _aerodynamics_functions,
    _panel,
    _parameter_validation,
    _transformations,
    problems,
)
from pterasoftware.convergence.unsteady_non_trapezoidal import (
    _get_num_cross_sections_for_panel_ar,
)

import configs

# -- Parameters ----------------------------------------------------------------
TARGET_AR = 1
CHORDWISE_VALUES = [12, 13, 14, 15]
CONFIG_NAME = "L170V_R180V_170Hz"
NUM_CYCLES = 1
# ------------------------------------------------------------------------------

# dt overrides for values missing from the cache.
_DT_OVERRIDES = {"1,15": 5.823762e-05}


def build_problem(nc: int, single_wing: bool = False):
    """Build an UnsteadyProblem at the given chordwise panel count with AR=1.

    :param nc: Number of chordwise panels.
    :param single_wing: If True, build with only the left wing (no mirror).
    :return: (problem, num_sections, delta_time)
    """
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

    left_params = wing_config["left"]
    right_params = wing_config["right"]

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

    left_wing = ps.geometry.wing.Wing(
        wing_cross_sections=make_cross_sections(),
        Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
        angles_Gs_to_Wn_ixyz=(
            left_params["phi_v_shift"], left_params["psi_v_shift"], 0.0,
        ),
        symmetric=False, mirror_only=False,
        num_chordwise_panels=nc, chordwise_spacing=chordwise_spacing,
    )

    if single_wing:
        wings = [left_wing]
    else:
        right_wing = ps.geometry.wing.Wing(
            wing_cross_sections=make_cross_sections(),
            Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
            angles_Gs_to_Wn_ixyz=(
                right_params["phi_v_shift"], right_params["psi_v_shift"], 0.0,
            ),
            symmetric=False, mirror_only=True,
            symmetryNormal_G=(0, 1, 0), symmetryPoint_G_Cg=(0, 0, 0),
            num_chordwise_panels=nc, chordwise_spacing=chordwise_spacing,
        )
        wings = [left_wing, right_wing]

    airplane = ps.geometry.airplane.Airplane(wings=wings, name="GammaBot")

    def make_wcs_movements(wing):
        return [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wcs,
            )
            for wcs in wing.wing_cross_sections
        ]

    left_wcsm = make_wcs_movements(airplane.wings[0])
    left_wm = ps.movements.wing_movement.WingMovement(
        base_wing=airplane.wings[0],
        wing_cross_section_movements=left_wcsm,
        rotationPointOffset_Gs_Ler=(x_offset, y_offset, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(left_flap["amp_x"], left_flap["amp_y"], 0.0),
        periodAngles_Gs_to_Wn_ixyz=(left_flap["period_x"], left_flap["period_y"], 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, left_flap["phase_y"], 0.0),
    )

    if single_wing:
        wing_movements = [left_wm]
    else:
        right_wcsm = make_wcs_movements(airplane.wings[1])
        right_wm = ps.movements.wing_movement.WingMovement(
            base_wing=airplane.wings[1],
            wing_cross_section_movements=right_wcsm,
            rotationPointOffset_Gs_Ler=(x_offset, y_offset, 0.0),
            ampAngles_Gs_to_Wn_ixyz=(right_flap["amp_x"], right_flap["amp_y"], 0.0),
            periodAngles_Gs_to_Wn_ixyz=(
                right_flap["period_x"], right_flap["period_y"], 0.0,
            ),
            spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
            phaseAngles_Gs_to_Wn_ixyz=(0.0, right_flap["phase_y"], 0.0),
        )
        wing_movements = [left_wm, right_wm]

    airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=wing_movements,
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

    problem = ps.problems.UnsteadyProblem(movement=movement)
    return problem, num_sections, delta_time


def _init_step_arrays(solver, num_panels):
    """Initialize all per-step arrays needed by _collapse_geometry."""
    solver._currentGridWingWingInfluences__E = np.zeros(
        (num_panels, num_panels), dtype=float
    )
    solver._current_bound_vortex_strengths = np.ones(num_panels, dtype=float)
    solver.panels = np.empty(num_panels, dtype=object)
    solver.stackUnitNormals_GP1 = np.zeros((num_panels, 3), dtype=float)
    solver.panel_areas = np.zeros(num_panels, dtype=float)
    solver.stackCpp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackBrbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackFrbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackFlbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackBlbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackRbrv_GP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackFbrv_GP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackLbrv_GP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackBbrv_GP1 = np.zeros((num_panels, 3), dtype=float)
    solver.panel_is_trailing_edge = np.zeros(num_panels, dtype=bool)
    solver.panel_is_leading_edge = np.zeros(num_panels, dtype=bool)
    solver.panel_is_left_edge = np.zeros(num_panels, dtype=bool)
    solver.panel_is_right_edge = np.zeros(num_panels, dtype=bool)
    solver.stackCblvpr_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackCblvpf_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackCblvpl_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackCblvpb_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._stackLastCpp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._last_bound_vortex_strengths = np.zeros(num_panels, dtype=float)
    solver._lastStackBrbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackFrbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackFlbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackBlbrvp_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackCblvpr_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackCblvpf_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackCblvpl_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver._lastStackCblvpb_GP1_CgP1 = np.zeros((num_panels, 3), dtype=float)
    solver.stackSeedPoints_GP1_CgP1 = np.zeros((0, 3), dtype=float)
    num_wake = 0
    for airplane in solver.current_airplanes:
        for wing in airplane.wings:
            wr = wing.wake_ring_vortices
            if wr is not None:
                num_wake += np.ravel(wr).size
    solver._current_wake_vortex_strengths = np.zeros(num_wake, dtype=float)
    solver._current_wake_vortex_ages = np.zeros(num_wake, dtype=float)
    solver._currentStackFrwrvp_GP1_CgP1 = np.zeros((num_wake, 3), dtype=float)
    solver._currentStackFlwrvp_GP1_CgP1 = np.zeros((num_wake, 3), dtype=float)
    solver._currentStackBlwrvp_GP1_CgP1 = np.zeros((num_wake, 3), dtype=float)
    solver._currentStackBrwrvp_GP1_CgP1 = np.zeros((num_wake, 3), dtype=float)


def _transform_points_to_Wn(points_GP1, T_inv):
    """Transform (N,3) points from GP1 frame to Wn frame using inverse transform."""
    return _transformations.apply_T_to_vectors(T_inv, points_GP1, has_point=True)


def _transform_vectors_to_Wn(vectors_GP1, T_inv):
    """Transform (N,3) direction vectors from GP1 frame to Wn frame."""
    return _transformations.apply_T_to_vectors(T_inv, vectors_GP1, has_point=False)


def _calculate_wing_wing_influences_in_Wn(solver):
    """Compute the influence matrix in the Wing's own frame (Wn).

    Gets the inverse of T_pas_Wn_Ler_to_G_Cg from the first wing at the current
    time step, transforms all collocation points, ring vortex vertices, and normals
    to the Wn frame, then computes the influence matrix there.
    """
    # Get the wing's transform. For single-airplane, G_Cg == GP1_CgP1.
    wing = solver.current_airplanes[0].wings[0]
    T_fwd = wing.T_pas_Wn_Ler_to_G_Cg
    T_inv = np.linalg.inv(T_fwd)

    # Transform all geometry to the Wn frame.
    cpp_Wn = _transform_points_to_Wn(solver.stackCpp_GP1_CgP1, T_inv)
    br_Wn = _transform_points_to_Wn(solver.stackBrbrvp_GP1_CgP1, T_inv)
    fr_Wn = _transform_points_to_Wn(solver.stackFrbrvp_GP1_CgP1, T_inv)
    fl_Wn = _transform_points_to_Wn(solver.stackFlbrvp_GP1_CgP1, T_inv)
    bl_Wn = _transform_points_to_Wn(solver.stackBlbrvp_GP1_CgP1, T_inv)
    normals_Wn = _transform_vectors_to_Wn(solver.stackUnitNormals_GP1, T_inv)

    # Compute induced velocities in Wn frame.
    gridNormVInd_Wn = _aerodynamics_functions.expanded_velocities_from_ring_vortices(
        stackP_GP1_CgP1=cpp_Wn,
        stackBrrvp_GP1_CgP1=br_Wn,
        stackFrrvp_GP1_CgP1=fr_Wn,
        stackFlrvp_GP1_CgP1=fl_Wn,
        stackBlrvp_GP1_CgP1=bl_Wn,
        strengths=solver._current_bound_vortex_strengths,
        ages=None,
        nu=solver.current_operating_point.nu,
    )

    # Dot with normals in Wn frame.
    A_Wn = np.einsum(
        "...k,...k->...",
        gridNormVInd_Wn,
        np.expand_dims(normals_Wn, axis=1),
    )
    return A_Wn


def compute_cond_and_sv_per_step(nc: int, single_wing: bool = False,
                                  wing_frame: bool = False):
    """For each time step, build geometry and A matrix, compute cond and min SV.

    :param nc: Number of chordwise panels.
    :param single_wing: If True, use only the left wing.
    :param wing_frame: If True, compute A in the Wing's Wn frame instead of GP1.
        Only valid when single_wing is True.
    :return: (steps, conds, min_svs, delta_time)
    """
    if wing_frame:
        label = "1-wing Wn"
    elif single_wing:
        label = "1-wing"
    else:
        label = "2-wing"

    problem, num_sections, dt = build_problem(nc, single_wing=single_wing or wing_frame)

    solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=problem
    )
    num_steps = solver.num_steps
    num_panels = solver.num_panels

    print(f"  NC={nc} ({label}): {num_panels} panels, {num_steps} steps, dt={dt:.6e}")

    # Initialize bound ring vortices (needed once).
    solver._initialize_panel_vortices()

    conds = np.zeros(num_steps)
    min_svs = np.zeros(num_steps)

    for step in range(num_steps):
        # Set current step and load geometry for this time step.
        solver._current_step = step
        current_problem = solver.steady_problems[step]
        solver.current_airplanes = current_problem.airplanes
        solver.current_operating_point = current_problem.operating_point
        solver._currentVInf_GP1__E = solver.current_operating_point.vInf_GP1__E

        _init_step_arrays(solver, num_panels)

        # Collapse geometry into 1D arrays (in GP1 frame).
        solver._collapse_geometry()

        if wing_frame:
            A = _calculate_wing_wing_influences_in_Wn(solver)
        else:
            solver._calculate_wing_wing_influences()
            A = solver._currentGridWingWingInfluences__E

        # Compute SVD (only singular values).
        s = np.linalg.svd(A, compute_uv=False)
        conds[step] = s[0] / s[-1] if s[-1] > 0 else float("inf")
        min_svs[step] = s[-1]

    return np.arange(num_steps), conds, min_svs, dt


def _run_variant(label, nc_values, **kwargs):
    """Run compute_cond_and_sv_per_step for all NC values and return results dict."""
    results = {}
    for nc in nc_values:
        t0 = time.time()
        steps, conds, min_svs, dt = compute_cond_and_sv_per_step(nc, **kwargs)
        elapsed = time.time() - t0
        print(f"    Elapsed: {elapsed:.1f} s")
        print(f"    Cond range: [{np.min(conds):.4e}, {np.max(conds):.4e}]")
        print(f"    Min SV range: [{np.min(min_svs):.4e}, {np.max(min_svs):.4e}]")
        results[nc] = (steps, conds, min_svs, dt)
        sys.stdout.flush()
    return results


def main():
    print("=== Two-wing (GP1 frame) ===")
    results_2w = _run_variant("2-wing", CHORDWISE_VALUES, single_wing=False)

    print("\n=== Single-wing (GP1 frame) ===")
    results_1w = _run_variant("1-wing", CHORDWISE_VALUES, single_wing=True)

    print("\n=== Single-wing (Wn frame) ===")
    results_wn = _run_variant("1-wing Wn", CHORDWISE_VALUES, wing_frame=True)

    # -- Plot ------------------------------------------------------------------
    colors = {12: "tab:blue", 13: "tab:green", 14: "tab:red", 15: "tab:orange"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for nc in CHORDWISE_VALUES:
        c = colors[nc]

        # Two-wing GP1 (solid lines).
        steps, conds, min_svs, dt = results_2w[nc]
        phase = steps / (len(steps) - 1) if len(steps) > 1 else steps
        ax1.semilogy(phase, conds, color=c, linestyle="-",
                     label=f"NC={nc} 2-wing")
        ax2.semilogy(phase, min_svs, color=c, linestyle="-",
                     label=f"NC={nc} 2-wing")

        # Single-wing GP1 (dashed lines).
        steps, conds, min_svs, dt = results_1w[nc]
        phase = steps / (len(steps) - 1) if len(steps) > 1 else steps
        ax1.semilogy(phase, conds, color=c, linestyle="--",
                     label=f"NC={nc} 1-wing GP1")
        ax2.semilogy(phase, min_svs, color=c, linestyle="--",
                     label=f"NC={nc} 1-wing GP1")

        # Single-wing Wn frame (dotted lines).
        steps, conds, min_svs, dt = results_wn[nc]
        phase = steps / (len(steps) - 1) if len(steps) > 1 else steps
        ax1.semilogy(phase, conds, color=c, linestyle=":",
                     label=f"NC={nc} 1-wing Wn")
        ax2.semilogy(phase, min_svs, color=c, linestyle=":",
                     label=f"NC={nc} 1-wing Wn")

    ax1.set_ylabel("Condition Number (2-norm)")
    ax1.set_title(
        f"Influence Matrix Conditioning vs. Flap Phase\n"
        f"AR={TARGET_AR}, Config={CONFIG_NAME}\n"
        f"(solid=2-wing, dashed=1-wing GP1, dotted=1-wing Wn frame)"
    )
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Flap Phase (fraction of cycle)")
    ax2.set_ylabel("Smallest Singular Value")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    out_path = Path(__file__).parent / "cond_vs_phase.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
