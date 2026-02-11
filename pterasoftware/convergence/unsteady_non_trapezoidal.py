"""Contains the non-trapezoidal unsteady convergence analysis functions and their
helpers.

**Contains the following classes:**

None

**Contains the following functions:**

analyze_unsteady_convergence_non_trapezoidal: Finds the converged parameters of an
UnsteadyProblem with non-trapezoidal wings.

analyze_unsteady_convergence_non_trapezoidal_optimized_dt: Like
analyze_unsteady_convergence_non_trapezoidal, but uses Movement's "optimize" option for
delta_time instead of sweeping it as a convergence parameter.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Generator

import numpy as np

from .. import (
    _parameter_validation,
    geometry,
    movements,
    problems,
    unsteady_ring_vortex_lattice_method,
)
from ._functions import (
    _COEFFICIENT_LABELS,
    _LOAD_LABELS,
    _LOAD_UNITS,
    _check_coefficient_convergence,
    _validate_non_trapezoidal_problem,
    _verify_panel_aspect_ratio,
    _visualize_wing_mesh,
    convergence_logger,
)


def _get_num_cross_sections_for_panel_ar(
    span: float,
    avg_chord: float,
    target_panel_ar: int,
    num_chordwise_panels: int,
    min_sections: int = 1,
) -> int:
    """Calculate number of spanwise sections to achieve target panel aspect ratio.

    For non-trapezoidal wings where each WingCrossSection has num_spanwise_panels=1, the
    number of spanwise panels equals the number of spanwise sections. This function
    calculates how many sections are needed to achieve a target panel aspect ratio.

    The panel aspect ratio is defined as:     panel_AR = panel_span / panel_chord =
    (span / num_sections) / (avg_chord / num_chordwise_panels)

    Solving for num_sections:     num_sections = (span * num_chordwise_panels) /
    (avg_chord * panel_AR)

    :param span: Total span of the wing.
    :param avg_chord: Average chord length across the span.
    :param target_panel_ar: Desired panel aspect ratio (integer >= 1).
    :param num_chordwise_panels: Number of chordwise panels.
    :param min_sections: Minimum number of sections to return. Default is 1.
    :return: Number of spanwise sections (integer >= min_sections).
    """
    # Calculate the ideal number of sections
    num_sections_float = (span * num_chordwise_panels) / (avg_chord * target_panel_ar)

    # Round to nearest integer, respecting minimum
    num_sections = max(min_sections, round(num_sections_float))

    return num_sections


def _calculate_wing_span_and_avg_chord(
    wing_section_data: np.ndarray,
) -> tuple[float, float]:
    """Calculate span and average chord from wing section data.

    This function takes wing section data in the format produced by geometry resamplers
    (such as the GammaBot DXF processor) and calculates the total span and average
    chord.

    :param wing_section_data: (N, 4) array with [dx, dy, dz, chord] per section. The
        displacements are relative to the previous section (first is from origin).
    :return: Tuple of (total_span, average_chord).
    """
    # Calculate cumulative positions from displacements
    positions = np.cumsum(wing_section_data[:, :3], axis=0)

    # Span is the Y-coordinate of the last section (assuming Y is spanwise)
    total_span = float(positions[-1, 1])

    # Average chord is the mean of all chord values
    avg_chord = float(np.mean(wing_section_data[:, 3]))

    return total_span, avg_chord


@contextlib.contextmanager
def _lock_cache_file(cache_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for safe concurrent cache access.

    Uses a .lock file adjacent to the cache file. On Windows uses msvcrt locking, on
    Unix uses fcntl file locking.

    :param cache_path: Path to the cache file to lock.
    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt

            # Write a byte so there is content to lock.
            lock_fh.write(" ")
            lock_fh.flush()
            lock_fh.seek(0)
            # LK_LOCK retries for approximately 10 seconds before raising OSError.
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


def analyze_unsteady_convergence_non_trapezoidal(
    ref_problem: problems.UnsteadyProblem,
    wing_geometry_resampler: Callable[[int, int], np.ndarray],
    prescribed_wake: bool | np.bool_ = True,
    free_wake: bool | np.bool_ = True,
    num_cycles_bounds: tuple[int, int] | None = None,
    num_chords_bounds: tuple[int, int] | None = None,
    panel_aspect_ratio_bounds: tuple[int, int] = (4, 1),
    num_chordwise_panels_bounds: tuple[int, int] = (3, 12),
    rtol: float | int = 0.05,
    atol: float | int = 0.001,
    show_solver_progress: bool | np.bool_ = True,
    visualize_meshes: bool | np.bool_ = False,
    visualization_dir: str | None = None,
    delta_time: float | None = None,
    delta_time_bounds: tuple[float, float] | None = None,
) -> tuple[float | None, bool, int, int, int] | tuple[None, None, None, None, None]:
    """Finds the converged parameters of an UnsteadyProblem with non-trapezoidal wings.

    This function is designed for wings defined with many WingCrossSection objects, each
    with num_spanwise_panels=1, where the planform shape is preserved by resampling the
    original geometry at different spanwise resolutions.

    **Key Difference from analyze_unsteady_convergence:**

    Instead of varying num_spanwise_panels per WingCrossSection, this function varies
    the NUMBER of WingCrossSections to achieve the target panel aspect ratio. This
    preserves non-trapezoidal planform shapes (e.g., elliptical, curved edges).

    **Restrictions:**

    - Only supports UnsteadyProblems with exactly one Airplane. - All Wings must use
    num_spanwise_panels=1 for all WingCrossSections (except last).

    **Procedure:**

    Convergence is found by varying the UnsteadyRingVortexLatticeMethodSolver's wake
    state (prescribed or free), the final length of the UnsteadyProblem's wake (in
    number of chord lengths for static geometry or number of maximum-period motion
    cycles for variable geometry), the Airplanes' Wings' Panels' aspect ratios (by
    resampling the wing geometry at different resolutions), and the Airplanes' Wings'
    numbers of chordwise Panels. These values are iterated over via four nested loops.
    The outermost loop is the wake state. The next loop is the wake length. The loop
    after that is the Panel aspect ratios, and the innermost loop is the number of
    chordwise Panels.

    With each new combination of these values, the UnsteadyProblem is solved, and each
    Airplane's 6 individual final load coefficients (cFX, cFY, cFZ, cMX, cMY, cMZ) are
    stored. As this function deals with UnsteadyProblems, it considers the final load
    coefficients to be the final cycle's mean load coefficients for UnsteadyProblems
    with variable geometry, and the final time step's load coefficients for static
    geometry cases. Convergence is checked per coefficient using an absolute plus
    relative tolerance: a coefficient is converged when abs(current - coarser) <= atol +
    rtol * max(abs(current), abs(coarser)).

    :param ref_problem: The UnsteadyProblem whose converged parameters will be found.
        Must contain exactly one Airplane with non-trapezoidal wings.
    :param wing_geometry_resampler: A callable that takes (wing_id, num_sections) and
        returns an (N+1, 4) ndarray with [dx, dy, dz, chord] data for each cross
        section. This allows the function to resample the wing geometry at different
        spanwise resolutions while preserving the planform shape.
    :param prescribed_wake: Determines if a prescribed wake state should be analyzed.
        Default is True.
    :param free_wake: Determines if a free wake state should be analyzed. Default is
        True.
    :param num_cycles_bounds: For problems with variable geometry, the range of wake
        lengths in cycles. Must be None for static geometry problems.
    :param num_chords_bounds: For problems with static geometry, the range of wake
        lengths in chord lengths. Must be None for variable geometry problems.
    :param panel_aspect_ratio_bounds: Range of panel aspect ratios to test, from
        coarsest to finest (descending order). Default is (4, 1).
    :param num_chordwise_panels_bounds: Range of chordwise panel counts to test
        (ascending order). Default is (3, 12).
    :param rtol: The relative tolerance for convergence checking. A coefficient is
        converged when its absolute change is within atol + rtol * max(abs(current),
        abs(coarser)). Must be a positive number (int or float). Values are converted to
        floats internally. The default is 0.05 (5%).
    :param atol: The absolute tolerance for convergence checking. Provides a floor
        tolerance for coefficients near zero. Must be a positive number (int or float).
        Values are converted to floats internally. The default is 0.001.
    :param show_solver_progress: Show TQDM progress bar during solver runs. Default is
        True.
    :param visualize_meshes: If True, save mesh visualizations for each panel AR /
        chordwise panel combination. Useful for verifying mesh quality. Default is
        False.
    :param visualization_dir: Directory to save mesh visualizations. Required if
        visualize_meshes is True. Default is None.
    :param delta_time: The time step to use for all simulations. If None (default),
        Movement will calculate a time step automatically. For high-frequency flapping,
        consider setting this explicitly (e.g., 1/frequency/steps_per_cycle) to ensure
        adequate temporal resolution. Cannot be used with delta_time_bounds.
    :param delta_time_bounds: Range of delta_time values to test, from coarsest
        (largest) to finest (smallest). Must be a tuple of two positive floats with the
        first value greater than or equal to the second. If None (default), delta_time
        convergence is not checked. Cannot be used with delta_time parameter. The units
        are seconds.
    :return: Tuple of (converged_delta_time, converged_wake, converged_wake_length,
        converged_panel_ar, converged_num_chordwise_panels), or (None, None, None, None,
        None) if not converged.
    """
    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    # Validate ref_problem is an UnsteadyProblem
    if not isinstance(ref_problem, problems.UnsteadyProblem):
        raise TypeError("ref_problem must be an UnsteadyProblem.")

    # Validate non-trapezoidal requirements (single airplane, num_spanwise_panels=1)
    _validate_non_trapezoidal_problem(ref_problem)

    # Validate wing_geometry_resampler is callable
    if not callable(wing_geometry_resampler):
        raise TypeError("wing_geometry_resampler must be a callable.")

    # Validate wake type parameters
    prescribed_wake = _parameter_validation.boolLike_return_bool(
        prescribed_wake, "prescribed_wake"
    )
    free_wake = _parameter_validation.boolLike_return_bool(free_wake, "free_wake")
    if not (prescribed_wake or free_wake):
        raise ValueError("At least one of prescribed_wake or free_wake must be True.")

    # Validate wake length bounds parameters
    ref_movement: movements.movement.Movement = ref_problem.movement
    static = ref_movement.static
    if static:
        if num_cycles_bounds is not None:
            raise ValueError(
                "num_cycles_bounds must be None for UnsteadyProblems "
                "with static geometry."
            )
        if not (isinstance(num_chords_bounds, tuple) and len(num_chords_bounds) == 2):
            raise TypeError("num_chords_bounds must be a tuple with length 2.")
        if not all(isinstance(bound, int) for bound in num_chords_bounds):
            raise TypeError("Both values in num_chords_bounds must be ints.")
        if num_chords_bounds[1] < num_chords_bounds[0]:
            raise ValueError(
                "The second value in num_chords_bounds must be greater than or equal "
                "to the first value."
            )
        if num_chords_bounds[1] <= 0:
            raise ValueError("Both values in num_chords_bounds must be positive.")
    else:
        if num_chords_bounds is not None:
            raise ValueError(
                "num_chords_bounds must be None for UnsteadyProblems "
                "with variable geometry."
            )
        if not (isinstance(num_cycles_bounds, tuple) and len(num_cycles_bounds) == 2):
            raise TypeError("num_cycles_bounds must be a tuple with length 2.")
        if not all(isinstance(bound, int) for bound in num_cycles_bounds):
            raise TypeError("Both values in num_cycles_bounds must be ints.")
        if num_cycles_bounds[1] < num_cycles_bounds[0]:
            raise ValueError(
                "The second value in num_cycles_bounds must be greater than or equal "
                "to the first value."
            )
        if num_cycles_bounds[1] <= 0:
            raise ValueError("Both values in num_cycles_bounds must be positive.")

    # Validate panel_aspect_ratio_bounds
    if not (
        isinstance(panel_aspect_ratio_bounds, tuple)
        and len(panel_aspect_ratio_bounds) == 2
    ):
        raise TypeError("panel_aspect_ratio_bounds must be a tuple with length 2.")
    if not all(isinstance(bound, int) for bound in panel_aspect_ratio_bounds):
        raise TypeError("Both values in panel_aspect_ratio_bounds must be ints.")
    if panel_aspect_ratio_bounds[0] < panel_aspect_ratio_bounds[1]:
        raise ValueError(
            "The first value in panel_aspect_ratio_bounds must be greater than or "
            "equal to the second value."
        )
    if panel_aspect_ratio_bounds[1] <= 0:
        raise ValueError("Both values in panel_aspect_ratio_bounds must be positive.")

    # Validate num_chordwise_panels_bounds
    if not (
        isinstance(num_chordwise_panels_bounds, tuple)
        and len(num_chordwise_panels_bounds) == 2
    ):
        raise TypeError("num_chordwise_panels_bounds must be a tuple with length 2.")
    if not all(isinstance(bound, int) for bound in num_chordwise_panels_bounds):
        raise TypeError("Both values in num_chordwise_panels_bounds must be ints.")
    if num_chordwise_panels_bounds[1] < num_chordwise_panels_bounds[0]:
        raise ValueError(
            "The first value in num_chordwise_panels_bounds must be less than or "
            "equal to the second value."
        )
    if num_chordwise_panels_bounds[0] <= 0:
        raise ValueError("Both values in num_chordwise_panels_bounds must be positive.")

    # Validate rtol
    rtol = _parameter_validation.number_in_range_return_float(
        rtol, "rtol", min_val=0.0, min_inclusive=False
    )

    # Validate atol
    atol = _parameter_validation.number_in_range_return_float(
        atol, "atol", min_val=0.0, min_inclusive=False
    )

    # Validate show_solver_progress
    show_solver_progress = _parameter_validation.boolLike_return_bool(
        show_solver_progress, "show_solver_progress"
    )

    # Validate visualization parameters
    visualize_meshes = _parameter_validation.boolLike_return_bool(
        visualize_meshes, "visualize_meshes"
    )
    if visualize_meshes and visualization_dir is None:
        raise ValueError("visualization_dir is required when visualize_meshes is True.")
    if visualize_meshes:
        assert visualization_dir is not None
        Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    # Validate delta_time and delta_time_bounds mutual exclusivity
    if delta_time is not None and delta_time_bounds is not None:
        raise ValueError("delta_time and delta_time_bounds cannot both be provided.")

    # Validate delta_time_bounds
    if delta_time_bounds is not None:
        if not (isinstance(delta_time_bounds, tuple) and len(delta_time_bounds) == 2):
            raise TypeError("delta_time_bounds must be a tuple with length 2.")
        if not all(isinstance(bound, (int, float)) for bound in delta_time_bounds):
            raise TypeError("Both values in delta_time_bounds must be numbers.")
        if not all(bound > 0 for bound in delta_time_bounds):
            raise ValueError("Both values in delta_time_bounds must be positive.")
        if delta_time_bounds[0] < delta_time_bounds[1]:
            raise ValueError(
                "The first value in delta_time_bounds must be greater than or "
                "equal to the second value."
            )

    # ==========================================================================
    # SETUP
    # ==========================================================================

    convergence_logger.info("Beginning non-trapezoidal convergence analysis...")

    ref_airplane_movement = ref_movement.airplane_movements[0]  # Single airplane
    ref_operating_point_movement = ref_movement.operating_point_movement

    # Pre-calculate span and average chord for each wing using high-res geometry
    wing_geometry_info: list[tuple[float, float]] = []  # [(span, avg_chord), ...]
    num_wings = len(ref_airplane_movement.wing_movements)

    for wing_id in range(num_wings):
        high_res_data = wing_geometry_resampler(wing_id, 100)
        span, avg_chord = _calculate_wing_span_and_avg_chord(high_res_data)
        wing_geometry_info.append((span, avg_chord))
        convergence_logger.info(
            f"\tWing {wing_id}: span={span:.4f}, avg_chord={avg_chord:.4f}"
        )

    # Create iteration lists
    wake_list: list[bool] = []
    if prescribed_wake:
        wake_list.append(True)
    if free_wake:
        wake_list.append(False)

    if static:
        assert num_chords_bounds is not None
        wake_lengths_list = list(range(num_chords_bounds[0], num_chords_bounds[1] + 1))
    else:
        assert num_cycles_bounds is not None
        wake_lengths_list = list(range(num_cycles_bounds[0], num_cycles_bounds[1] + 1))

    panel_aspect_ratios_list = list(
        range(panel_aspect_ratio_bounds[0], panel_aspect_ratio_bounds[1] - 1, -1)
    )
    num_chordwise_panels_list = list(
        range(num_chordwise_panels_bounds[0], num_chordwise_panels_bounds[1] + 1)
    )

    # Create delta_time iteration list using halving pattern
    if delta_time_bounds is not None:
        delta_time_list: list[float | None] = []
        current_dt = float(delta_time_bounds[0])
        target_dt = float(delta_time_bounds[1])
        while current_dt >= target_dt:
            delta_time_list.append(current_dt)
            current_dt = current_dt / 2.0
        if delta_time_list[-1] != target_dt:
            delta_time_list.append(target_dt)
    else:
        delta_time_list = [delta_time]

    # Initialize result storage arrays
    iter_times = np.zeros(
        (
            len(delta_time_list),
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
        ),
        dtype=float,
    )
    finalCoefficients = np.zeros(
        (
            len(delta_time_list),
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
            1,  # Single airplane
            6,  # cFX, cFY, cFZ, cMX, cMY, cMZ
        ),
        dtype=float,
    )

    # Caches
    num_cross_sections_cache: dict[tuple[int, int, int], int] = {}
    geometry_cache: dict[tuple[int, int], np.ndarray] = {}

    iteration = 0
    num_iterations = (
        len(delta_time_list)
        * len(wake_list)
        * len(wake_lengths_list)
        * len(panel_aspect_ratios_list)
        * len(num_chordwise_panels_list)
    )

    # ==========================================================================
    # MAIN ITERATION LOOPS
    # ==========================================================================

    for dt_id, this_delta_time in enumerate(delta_time_list):
        if this_delta_time is not None:
            convergence_logger.info(f"\tDelta time: {this_delta_time:.6f} s")
        else:
            convergence_logger.info("\tDelta time: auto-calculated")

        for wake_id, wake in enumerate(wake_list):
            if wake:
                convergence_logger.info("\t\tWake type: prescribed")
            else:
                convergence_logger.info("\t\tWake type: free")

            for length_id, wake_length in enumerate(wake_lengths_list):
                if static:
                    convergence_logger.info("\t\t\tChord lengths: " + str(wake_length))
                else:
                    convergence_logger.info("\t\t\tCycles: " + str(wake_length))

                for ar_id, panel_aspect_ratio in enumerate(panel_aspect_ratios_list):
                    convergence_logger.info(
                        "\t\t\t\tPanel aspect ratio: " + str(panel_aspect_ratio)
                    )

                    for chord_id, num_chordwise_panels in enumerate(
                        num_chordwise_panels_list
                    ):
                        convergence_logger.info(
                            "\t\t\t\t\tChordwise Panels: " + str(num_chordwise_panels)
                        )

                        iteration += 1
                        convergence_logger.info(
                            f"\t\t\t\t\t\tIteration {iteration}/{num_iterations}"
                        )

                        # ------------------------------------------------------
                        # BUILD GEOMETRY FOR THIS ITERATION
                        # ------------------------------------------------------

                        these_base_wings = []
                        these_wing_movements = []

                        for wing_id in range(num_wings):
                            ref_wing_movement = ref_airplane_movement.wing_movements[
                                wing_id
                            ]
                            ref_base_wing = ref_wing_movement.base_wing

                            # Get span and avg_chord for this wing
                            span, avg_chord = wing_geometry_info[wing_id]

                            # Calculate number of cross sections needed
                            cache_key = (ar_id, chord_id, wing_id)
                            if cache_key in num_cross_sections_cache:
                                num_sections = num_cross_sections_cache[cache_key]
                            else:
                                num_sections = _get_num_cross_sections_for_panel_ar(
                                    span,
                                    avg_chord,
                                    panel_aspect_ratio,
                                    num_chordwise_panels,
                                )
                                num_cross_sections_cache[cache_key] = num_sections

                            convergence_logger.debug(
                                f"\t\t\t\t\t\t\tWing {wing_id}: {num_sections} sections"
                            )

                            # Get resampled geometry (with caching)
                            geom_cache_key = (wing_id, num_sections)
                            if geom_cache_key in geometry_cache:
                                wing_section_data = geometry_cache[geom_cache_key]
                            else:
                                wing_section_data = wing_geometry_resampler(
                                    wing_id, num_sections
                                )
                                geometry_cache[geom_cache_key] = wing_section_data

                            # Create WingCrossSections
                            these_base_wcs: list[
                                geometry.wing_cross_section.WingCrossSection
                            ] = []
                            these_wcs_movements: list[
                                movements.wing_cross_section_movement.WingCrossSectionMovement
                            ] = []
                            num_wcs = num_sections + 1

                            for wcs_id in range(num_wcs):
                                this_num_spanwise_panels: int | None = (
                                    1 if wcs_id < num_sections else None
                                )

                                # Get reference WCS for non-geometry properties
                                ref_wcs_movement = (
                                    ref_wing_movement.wing_cross_section_movements[
                                        0 if wcs_id == 0 else -1
                                    ]
                                )
                                ref_base_wcs = ref_wcs_movement.base_wing_cross_section

                                this_base_wcs = geometry.wing_cross_section.WingCrossSection(
                                    Lp_Wcsp_Lpp=tuple(wing_section_data[wcs_id, :3]),
                                    chord=float(wing_section_data[wcs_id, 3]),
                                    num_spanwise_panels=this_num_spanwise_panels,
                                    angles_Wcsp_to_Wcs_ixyz=ref_base_wcs.angles_Wcsp_to_Wcs_ixyz,
                                    airfoil=geometry.airfoil.Airfoil(
                                        name=ref_base_wcs.airfoil.name,
                                        outline_A_lp=ref_base_wcs.airfoil.outline_A_lp,
                                        resample=ref_base_wcs.airfoil.resample,
                                        n_points_per_side=ref_base_wcs.airfoil.n_points_per_side,
                                    ),
                                    control_surface_symmetry_type=ref_base_wcs.control_surface_symmetry_type,
                                    control_surface_hinge_point=ref_base_wcs.control_surface_hinge_point,
                                    control_surface_deflection=ref_base_wcs.control_surface_deflection,
                                    spanwise_spacing=ref_base_wcs.spanwise_spacing,
                                )
                                these_base_wcs.append(this_base_wcs)

                                # Create WingCrossSectionMovement (no individual motion)
                                this_wcs_movement = movements.wing_cross_section_movement.WingCrossSectionMovement(
                                    base_wing_cross_section=this_base_wcs,
                                )
                                these_wcs_movements.append(this_wcs_movement)

                            # Create Wing
                            this_base_wing = geometry.wing.Wing(
                                wing_cross_sections=these_base_wcs,
                                num_chordwise_panels=num_chordwise_panels,
                                name=ref_base_wing.name,
                                Ler_Gs_Cgs=ref_base_wing.Ler_Gs_Cgs,
                                angles_Gs_to_Wn_ixyz=ref_base_wing.angles_Gs_to_Wn_ixyz,
                                symmetric=ref_base_wing.symmetric,
                                mirror_only=ref_base_wing.mirror_only,
                                symmetryNormal_G=ref_base_wing.symmetryNormal_G,
                                symmetryPoint_G_Cg=ref_base_wing.symmetryPoint_G_Cg,
                                chordwise_spacing=ref_base_wing.chordwise_spacing,
                            )
                            these_base_wings.append(this_base_wing)

                            # Create WingMovement
                            this_wing_movement = movements.wing_movement.WingMovement(
                                base_wing=this_base_wing,
                                wing_cross_section_movements=these_wcs_movements,
                                rotationPointOffset_Gs_Ler=ref_wing_movement.rotationPointOffset_Gs_Ler,
                                ampLer_Gs_Cgs=ref_wing_movement.ampLer_Gs_Cgs,
                                periodLer_Gs_Cgs=ref_wing_movement.periodLer_Gs_Cgs,
                                spacingLer_Gs_Cgs=ref_wing_movement.spacingLer_Gs_Cgs,
                                phaseLer_Gs_Cgs=ref_wing_movement.phaseLer_Gs_Cgs,
                                ampAngles_Gs_to_Wn_ixyz=ref_wing_movement.ampAngles_Gs_to_Wn_ixyz,
                                periodAngles_Gs_to_Wn_ixyz=ref_wing_movement.periodAngles_Gs_to_Wn_ixyz,
                                spacingAngles_Gs_to_Wn_ixyz=ref_wing_movement.spacingAngles_Gs_to_Wn_ixyz,
                                phaseAngles_Gs_to_Wn_ixyz=ref_wing_movement.phaseAngles_Gs_to_Wn_ixyz,
                            )
                            these_wing_movements.append(this_wing_movement)

                        # Create Airplane
                        ref_base_airplane = ref_airplane_movement.base_airplane
                        this_base_airplane = geometry.airplane.Airplane(
                            wings=these_base_wings,
                            name=ref_base_airplane.name,
                            Cg_GP1_CgP1=ref_base_airplane.Cg_GP1_CgP1,
                            weight=ref_base_airplane.weight,
                            s_ref=None,
                            c_ref=None,
                            b_ref=None,
                        )

                        # ------------------------------------------------------
                        # OPTIONAL: VISUALIZE MESH
                        # ------------------------------------------------------

                        if visualize_meshes:
                            ar_ok, actual_ar = _verify_panel_aspect_ratio(
                                this_base_airplane, panel_aspect_ratio
                            )
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\tTarget AR: {panel_aspect_ratio}, "
                                f"Actual AR: {actual_ar:.2f}, OK: {ar_ok}"
                            )

                            vis_filename = f"mesh_ar{panel_aspect_ratio}_chord{num_chordwise_panels}.png"

                            assert visualization_dir is not None
                            vis_path = Path(visualization_dir) / vis_filename
                            _visualize_wing_mesh(
                                this_base_airplane,
                                title=f"AR={panel_aspect_ratio}, Chordwise={num_chordwise_panels}",
                                show=False,
                                save_path=str(vis_path),
                            )

                        # ------------------------------------------------------
                        # CREATE MOVEMENT AND PROBLEM
                        # ------------------------------------------------------

                        this_airplane_movement = movements.airplane_movement.AirplaneMovement(
                            base_airplane=this_base_airplane,
                            wing_movements=these_wing_movements,
                            ampCg_GP1_CgP1=ref_airplane_movement.ampCg_GP1_CgP1,
                            periodCg_GP1_CgP1=ref_airplane_movement.periodCg_GP1_CgP1,
                            spacingCg_GP1_CgP1=ref_airplane_movement.spacingCg_GP1_CgP1,
                            phaseCg_GP1_CgP1=ref_airplane_movement.phaseCg_GP1_CgP1,
                        )

                        if static:
                            this_movement = movements.movement.Movement(
                                airplane_movements=[this_airplane_movement],
                                operating_point_movement=ref_operating_point_movement,
                                num_chords=wake_length,
                                delta_time=this_delta_time,
                            )
                        else:
                            this_movement = movements.movement.Movement(
                                airplane_movements=[this_airplane_movement],
                                operating_point_movement=ref_operating_point_movement,
                                num_cycles=wake_length,
                                delta_time=this_delta_time,
                            )

                        this_problem = problems.UnsteadyProblem(
                            movement=this_movement,
                            only_final_results=True,
                        )

                        # ------------------------------------------------------
                        # RUN SOLVER
                        # ------------------------------------------------------

                        this_solver = unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                            unsteady_problem=this_problem
                        )

                        convergence_logger.info("\t\t\t\t\t\t\tStarting simulation...")

                        iter_start = time.time()
                        this_solver.run(
                            prescribed_wake=wake,
                            calculate_streamlines=False,
                            show_progress=show_solver_progress,
                        )
                        iter_stop = time.time()
                        this_iter_time = iter_stop - iter_start

                        convergence_logger.info(
                            f"\t\t\t\t\t\t\tSimulation completed in "
                            f"{this_iter_time:.3f} s"
                        )

                        # ------------------------------------------------------
                        # EXTRACT AND STORE RESULTS
                        # ------------------------------------------------------

                        theseFinalCoefficients = np.zeros((1, 6), dtype=float)
                        theseFinalLoads = np.zeros(6, dtype=float)

                        if static:
                            theseFinalCoefficients[0, :3] = (
                                this_problem.finalForceCoefficients_W[0]
                            )
                            theseFinalCoefficients[0, 3:] = (
                                this_problem.finalMomentCoefficients_W_CgP1[0]
                            )
                            theseFinalLoads[:3] = this_problem.finalForces_W[0]
                            theseFinalLoads[3:] = this_problem.finalMoments_W_CgP1[0]
                        else:
                            theseFinalCoefficients[0, :3] = (
                                this_problem.finalMeanForceCoefficients_W[0]
                            )
                            theseFinalCoefficients[0, 3:] = (
                                this_problem.finalMeanMomentCoefficients_W_CgP1[0]
                            )
                            theseFinalLoads[:3] = this_problem.finalMeanForces_W[0]
                            theseFinalLoads[3:] = this_problem.finalMeanMoments_W_CgP1[
                                0
                            ]

                        finalCoefficients[
                            dt_id, wake_id, length_id, ar_id, chord_id, :, :
                        ] = theseFinalCoefficients
                        iter_times[dt_id, wake_id, length_id, ar_id, chord_id] = (
                            this_iter_time
                        )

                        # ------------------------------------------------------
                        # CHECK CONVERGENCE
                        # ------------------------------------------------------

                        # Get the current coefficients as a flat (6,) array.
                        current_coefficients = theseFinalCoefficients[0, :]

                        dt_converged = False
                        wake_converged = False
                        length_converged = False
                        ar_converged = False
                        chord_converged = False

                        # Delta time convergence check.
                        if dt_id > 0:
                            coarser_dt_coefficients = finalCoefficients[
                                dt_id - 1,
                                wake_id,
                                length_id,
                                ar_id,
                                chord_id,
                                0,
                                :,
                            ]
                            (
                                dt_converged,
                                dt_min_metric,
                                dt_errors,
                                dt_tols,
                                dt_metrics,
                            ) = _check_coefficient_convergence(
                                current_coefficients,
                                coarser_dt_coefficients,
                                rtol,
                                atol,
                            )
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - delta time:"
                            )
                            for i, label in enumerate(_COEFFICIENT_LABELS):
                                convergence_logger.info(
                                    f"\t\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                    f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                    f" {_LOAD_UNITS[i]}"
                                    f", error={dt_errors[i]:.3e}"
                                    f", tol={dt_tols[i]:.3e}"
                                    f", metric={dt_metrics[i]:.2f}"
                                )
                            min_label = _COEFFICIENT_LABELS[int(np.argmin(dt_metrics))]
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\t    Minimum metric: {dt_min_metric:.2f}"
                                f" ({min_label})"
                            )
                        else:
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - delta time: "
                                "not yet checked"
                            )

                        # Wake state convergence check.
                        if wake_id > 0:
                            coarser_wake_coefficients = finalCoefficients[
                                dt_id,
                                wake_id - 1,
                                length_id,
                                ar_id,
                                chord_id,
                                0,
                                :,
                            ]
                            (
                                wake_converged,
                                wake_min_metric,
                                wake_errors,
                                wake_tols,
                                wake_metrics,
                            ) = _check_coefficient_convergence(
                                current_coefficients,
                                coarser_wake_coefficients,
                                rtol,
                                atol,
                            )
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - wake type:"
                            )
                            for i, label in enumerate(_COEFFICIENT_LABELS):
                                convergence_logger.info(
                                    f"\t\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                    f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                    f" {_LOAD_UNITS[i]}"
                                    f", error={wake_errors[i]:.3e}"
                                    f", tol={wake_tols[i]:.3e}"
                                    f", metric={wake_metrics[i]:.2f}"
                                )
                            min_label = _COEFFICIENT_LABELS[
                                int(np.argmin(wake_metrics))
                            ]
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\t    Minimum metric: {wake_min_metric:.2f}"
                                f" ({min_label})"
                            )
                        else:
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - wake type: "
                                "not yet checked"
                            )

                        # Wake length convergence check.
                        if length_id > 0:
                            coarser_length_coefficients = finalCoefficients[
                                dt_id,
                                wake_id,
                                length_id - 1,
                                ar_id,
                                chord_id,
                                0,
                                :,
                            ]
                            (
                                length_converged,
                                length_min_metric,
                                length_errors,
                                length_tols,
                                length_metrics,
                            ) = _check_coefficient_convergence(
                                current_coefficients,
                                coarser_length_coefficients,
                                rtol,
                                atol,
                            )
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - wake length:"
                            )
                            for i, label in enumerate(_COEFFICIENT_LABELS):
                                convergence_logger.info(
                                    f"\t\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                    f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                    f" {_LOAD_UNITS[i]}"
                                    f", error={length_errors[i]:.3e}"
                                    f", tol={length_tols[i]:.3e}"
                                    f", metric={length_metrics[i]:.2f}"
                                )
                            min_label = _COEFFICIENT_LABELS[
                                int(np.argmin(length_metrics))
                            ]
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\t    Minimum metric: "
                                f"{length_min_metric:.2f} ({min_label})"
                            )
                        else:
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - wake length: "
                                "not yet checked"
                            )

                        # Panel aspect ratio convergence check.
                        if ar_id > 0:
                            coarser_ar_coefficients = finalCoefficients[
                                dt_id,
                                wake_id,
                                length_id,
                                ar_id - 1,
                                chord_id,
                                0,
                                :,
                            ]
                            (
                                ar_converged,
                                ar_min_metric,
                                ar_errors,
                                ar_tols,
                                ar_metrics,
                            ) = _check_coefficient_convergence(
                                current_coefficients,
                                coarser_ar_coefficients,
                                rtol,
                                atol,
                            )
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - Panel AR:"
                            )
                            for i, label in enumerate(_COEFFICIENT_LABELS):
                                convergence_logger.info(
                                    f"\t\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                    f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                    f" {_LOAD_UNITS[i]}"
                                    f", error={ar_errors[i]:.3e}"
                                    f", tol={ar_tols[i]:.3e}"
                                    f", metric={ar_metrics[i]:.2f}"
                                )
                            min_label = _COEFFICIENT_LABELS[int(np.argmin(ar_metrics))]
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\t    Minimum metric: {ar_min_metric:.2f}"
                                f" ({min_label})"
                            )
                        else:
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - Panel AR: "
                                "not yet checked"
                            )

                        # Chordwise Panels convergence check.
                        if chord_id > 0:
                            coarser_chord_coefficients = finalCoefficients[
                                dt_id,
                                wake_id,
                                length_id,
                                ar_id,
                                chord_id - 1,
                                0,
                                :,
                            ]
                            (
                                chord_converged,
                                chord_min_metric,
                                chord_errors,
                                chord_tols,
                                chord_metrics,
                            ) = _check_coefficient_convergence(
                                current_coefficients,
                                coarser_chord_coefficients,
                                rtol,
                                atol,
                            )
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - chordwise Panels:"
                            )
                            for i, label in enumerate(_COEFFICIENT_LABELS):
                                convergence_logger.info(
                                    f"\t\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                    f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                    f" {_LOAD_UNITS[i]}"
                                    f", error={chord_errors[i]:.3e}"
                                    f", tol={chord_tols[i]:.3e}"
                                    f", metric={chord_metrics[i]:.2f}"
                                )
                            min_label = _COEFFICIENT_LABELS[
                                int(np.argmin(chord_metrics))
                            ]
                            convergence_logger.info(
                                f"\t\t\t\t\t\t\t    Minimum metric: "
                                f"{chord_min_metric:.2f} ({min_label})"
                            )
                        else:
                            convergence_logger.info(
                                "\t\t\t\t\t\t\tConvergence check - chordwise Panels: "
                                "not yet checked"
                            )

                        # Check convergence conditions.
                        dt_saturated = (
                            delta_time_bounds is not None
                            and this_delta_time == delta_time_bounds[1]
                        )
                        wake_saturated = not wake
                        ar_saturated = panel_aspect_ratio == 1

                        single_dt = len(delta_time_list) == 1
                        single_wake = len(wake_list) == 1
                        single_length = len(wake_lengths_list) == 1
                        single_ar = len(panel_aspect_ratios_list) == 1
                        single_chord = len(num_chordwise_panels_list) == 1

                        dt_passed = dt_converged or single_dt or dt_saturated
                        wake_passed = wake_converged or single_wake or wake_saturated
                        length_passed = length_converged or single_length
                        ar_passed = ar_converged or single_ar or ar_saturated
                        chord_passed = chord_converged or single_chord

                        # If all passed, return converged parameters
                        if (
                            dt_passed
                            and wake_passed
                            and length_passed
                            and ar_passed
                            and chord_passed
                        ):
                            # Determine converged_dt_id
                            if single_dt:
                                converged_dt_id = dt_id
                            elif dt_converged:
                                converged_dt_id = dt_id - 1
                            else:
                                converged_dt_id = dt_id

                            if single_wake:
                                converged_wake_id = wake_id
                            elif wake_converged:
                                converged_wake_id = wake_id - 1
                            else:
                                converged_wake_id = wake_id

                            if single_length:
                                converged_length_id = length_id
                            else:
                                converged_length_id = length_id - 1

                            if single_ar:
                                converged_ar_id = ar_id
                            elif ar_converged:
                                converged_ar_id = ar_id - 1
                            else:
                                converged_ar_id = ar_id

                            if single_chord:
                                converged_chord_id = chord_id
                            else:
                                converged_chord_id = chord_id - 1

                            converged_delta_time = delta_time_list[converged_dt_id]
                            converged_wake = wake_list[converged_wake_id]
                            converged_wake_length = wake_lengths_list[
                                converged_length_id
                            ]
                            converged_chordwise_panels = num_chordwise_panels_list[
                                converged_chord_id
                            ]
                            converged_aspect_ratio = panel_aspect_ratios_list[
                                converged_ar_id
                            ]
                            converged_iter_time = float(
                                iter_times[
                                    converged_dt_id,
                                    converged_wake_id,
                                    converged_length_id,
                                    converged_ar_id,
                                    converged_chord_id,
                                ]
                            )

                            # Log results
                            if (
                                single_dt
                                or single_wake
                                or single_length
                                or single_ar
                                or single_chord
                            ):
                                convergence_logger.info(
                                    "The analysis found a semi-converged case:"
                                )
                                if single_dt:
                                    convergence_logger.warning(
                                        "Delta time convergence not checked"
                                    )
                                if single_wake:
                                    convergence_logger.warning(
                                        "Wake type convergence not checked"
                                    )
                                if single_length:
                                    convergence_logger.warning(
                                        "Wake length convergence not checked"
                                    )
                                if single_ar:
                                    convergence_logger.warning(
                                        "Panel aspect ratio convergence not checked"
                                    )
                                if single_chord:
                                    convergence_logger.warning(
                                        "Chordwise Panels convergence not checked"
                                    )
                            else:
                                convergence_logger.info(
                                    "The analysis found a converged case:"
                                )

                            if converged_delta_time is not None:
                                convergence_logger.info(
                                    f"\tDelta time: {converged_delta_time:.6f} s"
                                )
                            else:
                                convergence_logger.info("\tDelta time: auto-calculated")

                            if converged_wake:
                                convergence_logger.info("\tWake type: prescribed")
                            else:
                                convergence_logger.info("\tWake type: free")

                            if static:
                                convergence_logger.info(
                                    "\tChord lengths: " + str(converged_wake_length)
                                )
                            else:
                                convergence_logger.info(
                                    "\tCycles: " + str(converged_wake_length)
                                )

                            convergence_logger.info(
                                "\tPanel aspect ratio: " + str(converged_aspect_ratio)
                            )
                            convergence_logger.info(
                                "\tChordwise Panels: " + str(converged_chordwise_panels)
                            )
                            convergence_logger.info(
                                "\tSimulation completed in "
                                + str(round(converged_iter_time, 3))
                                + " s"
                            )

                            # Log spanwise sections for each wing
                            convergence_logger.info("\tSpanwise sections per wing:")
                            for wing_id in range(num_wings):
                                cache_key = (
                                    converged_ar_id,
                                    converged_chord_id,
                                    wing_id,
                                )
                                num_sections = num_cross_sections_cache.get(
                                    cache_key, 0
                                )
                                convergence_logger.info(
                                    f"\t\tWing {wing_id}: {num_sections} sections"
                                )

                            return (
                                converged_delta_time,
                                converged_wake,
                                converged_wake_length,
                                converged_aspect_ratio,
                                converged_chordwise_panels,
                            )

    # No convergence found
    convergence_logger.info(
        "The analysis did not find a converged case within the given bounds"
    )
    return None, None, None, None, None


# TEST: Consider adding unit tests for this function.
# TEST: Assess how comprehensive this function's integration tests are and update or
#  extend them if needed.
# TODO: If a converged mesh was found, consider also returning the converged solver.
def analyze_unsteady_convergence_non_trapezoidal_optimized_dt(
    ref_problem: problems.UnsteadyProblem,
    wing_geometry_resampler: Callable[[int, int], np.ndarray],
    prescribed_wake: bool | np.bool_ = True,
    free_wake: bool | np.bool_ = True,
    num_cycles_bounds: tuple[int, int] | None = None,
    num_chords_bounds: tuple[int, int] | None = None,
    panel_aspect_ratio_bounds: tuple[int, int] = (4, 1),
    num_chordwise_panels_bounds: tuple[int, int] = (3, 12),
    rtol: float | int = 0.05,
    atol: float | int = 0.001,
    coefficient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    show_solver_progress: bool | np.bool_ = True,
    visualize_meshes: bool | np.bool_ = False,
    visualization_dir: str | None = None,
    cache_file: str | Path | None = None,
) -> tuple[bool, int, int, int] | tuple[None, None, None, None]:
    """Finds the converged parameters of an UnsteadyProblem with non-trapezoidal wings,
    using Movement's "optimize" option for delta_time.

    This function is like analyze_unsteady_convergence_non_trapezoidal except that it
    does not sweep delta_time as a convergence parameter. Instead, it always uses
    Movement's "optimize" option for delta_time, which finds the delta_time that
    minimizes the area mismatch between wake RingVortices and their parent bound
    trailing edge RingVortices.

    **Key Difference from analyze_unsteady_convergence_non_trapezoidal:**

    Instead of iterating over a range of delta_time values (or using a fixed
    delta_time), this function passes delta_time="optimize" to Movement for every
    iteration. This removes delta_time as a convergence parameter, leaving four
    parameters to converge: wake state, wake length, Panel aspect ratio, and number of
    chordwise Panels.

    **Restrictions:**

    - Only supports UnsteadyProblems with exactly one Airplane. - All Wings must use
    num_spanwise_panels=1 for all WingCrossSections (except last).

    **Procedure:**

    Convergence is found by varying the UnsteadyRingVortexLatticeMethodSolver's wake
    state (prescribed or free), the final length of the UnsteadyProblem's wake (in
    number of chord lengths for static geometry or number of maximum period motion
    cycles for variable geometry), the Airplanes' Wings' Panels' aspect ratios (by
    resampling the wing geometry at different resolutions), and the Airplanes' Wings'
    numbers of chordwise Panels. These values are iterated over via four nested loops.
    The outermost loop is the wake state. The next loop is the wake length. The loop
    after that is the Panel aspect ratios, and the innermost loop is the number of
    chordwise Panels.

    With each new combination of these values, delta_time is automatically optimized via
    Movement's "optimize" option. Then the UnsteadyProblem is solved, and each
    Airplane's 6 individual final load coefficients (cFX, cFY, cFZ, cMX, cMY, cMZ) are
    stored. Convergence is checked per coefficient using an absolute plus relative
    tolerance: a coefficient is converged when abs(current - coarser) <= atol + rtol *
    max(abs(current), abs(coarser)).

    :param ref_problem: The UnsteadyProblem whose converged parameters will be found.
        Must contain exactly one Airplane with non-trapezoidal wings.
    :param wing_geometry_resampler: A callable that takes (wing_id, num_sections) and
        returns an (N+1, 4) ndarray with [dx, dy, dz, chord] data for each cross
        section. This allows the function to resample the wing geometry at different
        spanwise resolutions while preserving the planform shape.
    :param prescribed_wake: Determines if a prescribed wake state should be analyzed.
        Can be a bool or a numpy bool and will be converted internally to a bool. The
        default is True.
    :param free_wake: Determines if a free wake state should be analyzed. Can be a bool
        or a numpy bool and will be converted internally to a bool. The default is True.
    :param num_cycles_bounds: For problems with variable geometry, the range of wake
        lengths in cycles. Must be a tuple of two ints in ascending order, or None for
        static geometry problems. The default is None.
    :param num_chords_bounds: For problems with static geometry, the range of wake
        lengths in chord lengths. Must be a tuple of two ints in ascending order, or
        None for variable geometry problems. The default is None.
    :param panel_aspect_ratio_bounds: A tuple of two ints, in descending order, that
        determines the range of Panel aspect ratios to test, from coarsest to finest.
        The default is (4, 1).
    :param num_chordwise_panels_bounds: A tuple of two ints, in ascending order, that
        determines the range of chordwise Panel counts to test. The default is (3, 12).
    :param rtol: The relative tolerance for convergence checking. A coefficient is
        converged when its absolute change is within atol + rtol * max(abs(current),
        abs(coarser)). Must be a positive number (int or float). Values are converted to
        floats internally. The default is 0.05 (5%).
    :param atol: The absolute tolerance for convergence checking. Provides a floor
        tolerance for coefficients near zero. Must be a positive number (int or float).
        Values are converted to floats internally. The default is 0.001.
    :param coefficient_mask: A tuple of 6 bools that determines which of the 6 load
        coefficients (cFX, cFY, cFZ, cMX, cMY, cMZ) are checked for convergence. True
        means the coefficient is checked; False means it is ignored. At least one
        element must be True. If None, all 6 coefficients are checked. The default is
        None.
    :param show_solver_progress: Show TQDM progress bar during solver runs. Can be a
        bool or a numpy bool and will be converted internally to a bool. The default is
        True.
    :param visualize_meshes: If True, save mesh visualizations for each Panel aspect
        ratio and chordwise Panel combination. Useful for verifying mesh quality. Can be
        a bool or a numpy bool and will be converted internally to a bool. The default
        is False.
    :param visualization_dir: Directory to save mesh visualizations. Required if
        visualize_meshes is True. The default is None.
    :param cache_file: Path to a JSON file for caching simulation results. When
        provided, previously computed results are loaded from this file and reused,
        skipping redundant simulations. New results are saved to this file after each
        simulation. If None (default), no caching is performed.
    :return: A tuple of (converged_wake, converged_wake_length, converged_panel_ar,
        converged_num_chordwise_panels), or (None, None, None, None) if not converged.
    """
    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    # Validate ref_problem is an UnsteadyProblem.
    if not isinstance(ref_problem, problems.UnsteadyProblem):
        raise TypeError("ref_problem must be an UnsteadyProblem.")

    # Validate non-trapezoidal requirements (single airplane, num_spanwise_panels=1).
    _validate_non_trapezoidal_problem(ref_problem)

    # Validate wing_geometry_resampler is callable.
    if not callable(wing_geometry_resampler):
        raise TypeError("wing_geometry_resampler must be a callable.")

    # Validate wake type parameters.
    prescribed_wake = _parameter_validation.boolLike_return_bool(
        prescribed_wake, "prescribed_wake"
    )
    free_wake = _parameter_validation.boolLike_return_bool(free_wake, "free_wake")
    if not (prescribed_wake or free_wake):
        raise ValueError("At least one of prescribed_wake or free_wake must be True.")

    # Validate wake length bounds parameters.
    ref_movement: movements.movement.Movement = ref_problem.movement
    static = ref_movement.static
    if static:
        if num_cycles_bounds is not None:
            raise ValueError(
                "num_cycles_bounds must be None for UnsteadyProblems "
                "with static geometry."
            )
        if not (isinstance(num_chords_bounds, tuple) and len(num_chords_bounds) == 2):
            raise TypeError("num_chords_bounds must be a tuple with length 2.")
        if not all(isinstance(bound, int) for bound in num_chords_bounds):
            raise TypeError("Both values in num_chords_bounds must be ints.")
        if num_chords_bounds[1] < num_chords_bounds[0]:
            raise ValueError(
                "The second value in num_chords_bounds must be greater than or equal "
                "to the first value."
            )
        if num_chords_bounds[1] <= 0:
            raise ValueError("Both values in num_chords_bounds must be positive.")
    else:
        if num_chords_bounds is not None:
            raise ValueError(
                "num_chords_bounds must be None for UnsteadyProblems "
                "with variable geometry."
            )
        if not (isinstance(num_cycles_bounds, tuple) and len(num_cycles_bounds) == 2):
            raise TypeError("num_cycles_bounds must be a tuple with length 2.")
        if not all(isinstance(bound, int) for bound in num_cycles_bounds):
            raise TypeError("Both values in num_cycles_bounds must be ints.")
        if num_cycles_bounds[1] < num_cycles_bounds[0]:
            raise ValueError(
                "The second value in num_cycles_bounds must be greater than or equal "
                "to the first value."
            )
        if num_cycles_bounds[1] <= 0:
            raise ValueError("Both values in num_cycles_bounds must be positive.")

    # Validate panel_aspect_ratio_bounds.
    if not (
        isinstance(panel_aspect_ratio_bounds, tuple)
        and len(panel_aspect_ratio_bounds) == 2
    ):
        raise TypeError("panel_aspect_ratio_bounds must be a tuple with length 2.")
    if not all(isinstance(bound, int) for bound in panel_aspect_ratio_bounds):
        raise TypeError("Both values in panel_aspect_ratio_bounds must be ints.")
    if panel_aspect_ratio_bounds[0] < panel_aspect_ratio_bounds[1]:
        raise ValueError(
            "The first value in panel_aspect_ratio_bounds must be greater than or "
            "equal to the second value."
        )
    if panel_aspect_ratio_bounds[1] <= 0:
        raise ValueError("Both values in panel_aspect_ratio_bounds must be positive.")

    # Validate num_chordwise_panels_bounds.
    if not (
        isinstance(num_chordwise_panels_bounds, tuple)
        and len(num_chordwise_panels_bounds) == 2
    ):
        raise TypeError("num_chordwise_panels_bounds must be a tuple with length 2.")
    if not all(isinstance(bound, int) for bound in num_chordwise_panels_bounds):
        raise TypeError("Both values in num_chordwise_panels_bounds must be ints.")
    if num_chordwise_panels_bounds[1] < num_chordwise_panels_bounds[0]:
        raise ValueError(
            "The first value in num_chordwise_panels_bounds must be less than or "
            "equal to the second value."
        )
    if num_chordwise_panels_bounds[0] <= 0:
        raise ValueError("Both values in num_chordwise_panels_bounds must be positive.")

    # Validate rtol.
    rtol = _parameter_validation.number_in_range_return_float(
        rtol, "rtol", min_val=0.0, min_inclusive=False
    )

    # Validate atol.
    atol = _parameter_validation.number_in_range_return_float(
        atol, "atol", min_val=0.0, min_inclusive=False
    )

    # Validate coefficient_mask.
    if coefficient_mask is None:
        coefficient_mask = (True, True, True, True, True, True)
    if not isinstance(coefficient_mask, tuple):
        raise TypeError("coefficient_mask must be a tuple or None.")
    if len(coefficient_mask) != 6:
        raise ValueError("coefficient_mask must have exactly 6 elements.")
    if not all(isinstance(elem, bool) for elem in coefficient_mask):
        raise TypeError("All elements of coefficient_mask must be bools.")
    if not any(coefficient_mask):
        raise ValueError("At least one element of coefficient_mask must be True.")
    coefficient_mask_array = np.array(coefficient_mask, dtype=bool)

    # Validate show_solver_progress.
    show_solver_progress = _parameter_validation.boolLike_return_bool(
        show_solver_progress, "show_solver_progress"
    )

    # Validate visualization parameters.
    visualize_meshes = _parameter_validation.boolLike_return_bool(
        visualize_meshes, "visualize_meshes"
    )
    if visualize_meshes and visualization_dir is None:
        raise ValueError("visualization_dir is required when visualize_meshes is True.")
    if visualize_meshes:
        assert visualization_dir is not None
        Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    # Validate and load cache_file.
    simulation_cache: dict[str, dict] = {}
    persisted_dt_cache: dict[str, float] = {}
    if cache_file is not None:
        cache_file = Path(cache_file)
        if cache_file.exists():
            with _lock_cache_file(cache_file):
                with open(cache_file, "r") as f:
                    loaded_cache = json.load(f)
            persisted_dt_cache = loaded_cache.pop("_optimized_dt", {})
            simulation_cache = loaded_cache

    # ==========================================================================
    # SETUP
    # ==========================================================================

    convergence_logger.info(
        "Beginning non-trapezoidal convergence analysis (optimized delta_time)..."
    )

    ref_airplane_movement = ref_movement.airplane_movements[0]  # Single airplane.
    ref_operating_point_movement = ref_movement.operating_point_movement

    # Pre-calculate span and average chord for each wing using high resolution geometry.
    wing_geometry_info: list[tuple[float, float]] = []  # [(span, avg_chord), ...]
    num_wings = len(ref_airplane_movement.wing_movements)

    for wing_id in range(num_wings):
        high_res_data = wing_geometry_resampler(wing_id, 100)
        span, avg_chord = _calculate_wing_span_and_avg_chord(high_res_data)
        wing_geometry_info.append((span, avg_chord))
        convergence_logger.info(
            f"\tWing {wing_id}: span={span:.4f}, avg_chord={avg_chord:.4f}"
        )

    active_labels = ", ".join(
        label for label, active in zip(_COEFFICIENT_LABELS, coefficient_mask) if active
    )
    convergence_logger.info(f"\tActive coefficients for convergence: {active_labels}")

    # Create iteration lists.
    wake_list: list[bool] = []
    if prescribed_wake:
        wake_list.append(True)
    if free_wake:
        wake_list.append(False)

    if static:
        assert num_chords_bounds is not None
        wake_lengths_list = list(range(num_chords_bounds[0], num_chords_bounds[1] + 1))
    else:
        assert num_cycles_bounds is not None
        wake_lengths_list = list(range(num_cycles_bounds[0], num_cycles_bounds[1] + 1))

    panel_aspect_ratios_list = list(
        range(panel_aspect_ratio_bounds[0], panel_aspect_ratio_bounds[1] - 1, -1)
    )
    num_chordwise_panels_list = list(
        range(num_chordwise_panels_bounds[0], num_chordwise_panels_bounds[1] + 1)
    )

    # Initialize result storage arrays.
    iter_times = np.zeros(
        (
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
        ),
        dtype=float,
    )
    finalCoefficients = np.zeros(
        (
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
            1,  # Single airplane.
            6,  # cFX, cFY, cFZ, cMX, cMY, cMZ.
        ),
        dtype=float,
    )

    # Caches.
    num_cross_sections_cache: dict[tuple[int, int, int], int] = {}
    geometry_cache: dict[tuple[int, int], np.ndarray] = {}
    optimized_dt_cache: dict[tuple[int, int], float] = {
        tuple(int(x) for x in k.split(",")): v  # type: ignore[misc]
        for k, v in persisted_dt_cache.items()
    }

    # Pre-populate result arrays from the simulation cache so that convergence
    # checks can compare against values computed in previous runs.
    num_prepopulated = 0
    for wake_id, wake in enumerate(wake_list):
        for length_id, wake_length in enumerate(wake_lengths_list):
            for ar_id, panel_aspect_ratio in enumerate(panel_aspect_ratios_list):
                for chord_id, num_chordwise_panels in enumerate(
                    num_chordwise_panels_list
                ):
                    key = (
                        f"{wake},{wake_length},"
                        f"{panel_aspect_ratio},{num_chordwise_panels}"
                    )
                    if key in simulation_cache:
                        cached = simulation_cache[key]
                        finalCoefficients[wake_id, length_id, ar_id, chord_id, 0, :] = (
                            cached["coefficients"]
                        )
                        iter_times[wake_id, length_id, ar_id, chord_id] = cached["time"]
                        num_prepopulated += 1
    if num_prepopulated > 0:
        convergence_logger.info(
            f"\tPre-populated {num_prepopulated} entries from cache"
        )

    iteration = 0
    num_iterations = (
        len(wake_list)
        * len(wake_lengths_list)
        * len(panel_aspect_ratios_list)
        * len(num_chordwise_panels_list)
    )

    # ==========================================================================
    # MAIN ITERATION LOOPS
    # ==========================================================================

    for wake_id, wake in enumerate(wake_list):
        if wake:
            convergence_logger.info("\tWake type: prescribed")
        else:
            convergence_logger.info("\tWake type: free")

        for length_id, wake_length in enumerate(wake_lengths_list):
            if static:
                convergence_logger.info("\t\tChord lengths: " + str(wake_length))
            else:
                convergence_logger.info("\t\tCycles: " + str(wake_length))

            for ar_id, panel_aspect_ratio in enumerate(panel_aspect_ratios_list):
                convergence_logger.info(
                    "\t\t\tPanel aspect ratio: " + str(panel_aspect_ratio)
                )

                for chord_id, num_chordwise_panels in enumerate(
                    num_chordwise_panels_list
                ):
                    convergence_logger.info(
                        "\t\t\t\tChordwise Panels: " + str(num_chordwise_panels)
                    )

                    iteration += 1
                    convergence_logger.info(
                        f"\t\t\t\t\tIteration {iteration}/{num_iterations}"
                    )

                    # ----------------------------------------------------------
                    # CHECK SIMULATION CACHE
                    # ----------------------------------------------------------

                    sim_cache_key = (
                        f"{wake},{wake_length},"
                        f"{panel_aspect_ratio},{num_chordwise_panels}"
                    )

                    if sim_cache_key in simulation_cache:
                        # Cache hit: restore results from cache.
                        cached = simulation_cache[sim_cache_key]
                        theseFinalCoefficients = np.zeros((1, 6), dtype=float)
                        theseFinalCoefficients[0, :] = cached["coefficients"]
                        theseFinalLoads = np.array(cached["loads"], dtype=float)
                        this_iter_time = cached["time"]

                        # Populate num_cross_sections_cache (needed for final
                        # convergence logging).
                        for wing_id in range(num_wings):
                            cs_cache_key = (ar_id, chord_id, wing_id)
                            if cs_cache_key not in num_cross_sections_cache:
                                span, avg_chord = wing_geometry_info[wing_id]
                                num_cross_sections_cache[cs_cache_key] = (
                                    _get_num_cross_sections_for_panel_ar(
                                        span,
                                        avg_chord,
                                        panel_aspect_ratio,
                                        num_chordwise_panels,
                                    )
                                )

                        convergence_logger.info(
                            f"\t\t\t\t\t\tCache hit (original time: "
                            f"{this_iter_time:.3f} s)"
                        )
                    else:
                        # Cache miss: build geometry, run solver, extract results.

                        # ------------------------------------------------------
                        # BUILD GEOMETRY FOR THIS ITERATION
                        # ------------------------------------------------------

                        these_base_wings = []
                        these_wing_movements = []

                        for wing_id in range(num_wings):
                            ref_wing_movement = ref_airplane_movement.wing_movements[
                                wing_id
                            ]
                            ref_base_wing = ref_wing_movement.base_wing

                            # Get span and avg_chord for this wing.
                            span, avg_chord = wing_geometry_info[wing_id]

                            # Calculate number of cross sections needed.
                            cache_key = (ar_id, chord_id, wing_id)
                            if cache_key in num_cross_sections_cache:
                                num_sections = num_cross_sections_cache[cache_key]
                            else:
                                num_sections = _get_num_cross_sections_for_panel_ar(
                                    span,
                                    avg_chord,
                                    panel_aspect_ratio,
                                    num_chordwise_panels,
                                )
                                num_cross_sections_cache[cache_key] = num_sections

                            convergence_logger.debug(
                                f"\t\t\t\t\t\tWing {wing_id}: {num_sections} sections"
                            )

                            # Get resampled geometry (with caching).
                            geom_cache_key = (wing_id, num_sections)
                            if geom_cache_key in geometry_cache:
                                wing_section_data = geometry_cache[geom_cache_key]
                            else:
                                wing_section_data = wing_geometry_resampler(
                                    wing_id, num_sections
                                )
                                geometry_cache[geom_cache_key] = wing_section_data

                            # Create WingCrossSections.
                            these_base_wing_cross_sections: list[
                                geometry.wing_cross_section.WingCrossSection
                            ] = []
                            these_wing_cross_section_movements: list[
                                movements.wing_cross_section_movement.WingCrossSectionMovement
                            ] = []
                            num_wing_cross_sections = num_sections + 1

                            for wing_cross_section_id in range(num_wing_cross_sections):
                                this_num_spanwise_panels: int | None = (
                                    1 if wing_cross_section_id < num_sections else None
                                )

                                # Get reference WingCrossSection for non-geometry
                                # properties.
                                ref_wing_cross_section_movement = (
                                    ref_wing_movement.wing_cross_section_movements[
                                        0 if wing_cross_section_id == 0 else -1
                                    ]
                                )
                                ref_base_wing_cross_section = (
                                    ref_wing_cross_section_movement.base_wing_cross_section
                                )

                                this_base_wing_cross_section = geometry.wing_cross_section.WingCrossSection(
                                    Lp_Wcsp_Lpp=tuple(
                                        wing_section_data[wing_cross_section_id, :3]
                                    ),
                                    chord=float(
                                        wing_section_data[wing_cross_section_id, 3]
                                    ),
                                    num_spanwise_panels=this_num_spanwise_panels,
                                    angles_Wcsp_to_Wcs_ixyz=ref_base_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
                                    airfoil=geometry.airfoil.Airfoil(
                                        name=ref_base_wing_cross_section.airfoil.name,
                                        outline_A_lp=ref_base_wing_cross_section.airfoil.outline_A_lp,
                                        resample=ref_base_wing_cross_section.airfoil.resample,
                                        n_points_per_side=ref_base_wing_cross_section.airfoil.n_points_per_side,
                                    ),
                                    control_surface_symmetry_type=ref_base_wing_cross_section.control_surface_symmetry_type,
                                    control_surface_hinge_point=ref_base_wing_cross_section.control_surface_hinge_point,
                                    control_surface_deflection=ref_base_wing_cross_section.control_surface_deflection,
                                    spanwise_spacing=ref_base_wing_cross_section.spanwise_spacing,
                                )
                                these_base_wing_cross_sections.append(
                                    this_base_wing_cross_section
                                )

                                # Create WingCrossSectionMovement (no individual
                                # motion).
                                this_wing_cross_section_movement = movements.wing_cross_section_movement.WingCrossSectionMovement(
                                    base_wing_cross_section=this_base_wing_cross_section,
                                )
                                these_wing_cross_section_movements.append(
                                    this_wing_cross_section_movement
                                )

                            # Create Wing.
                            this_base_wing = geometry.wing.Wing(
                                wing_cross_sections=these_base_wing_cross_sections,
                                num_chordwise_panels=num_chordwise_panels,
                                name=ref_base_wing.name,
                                Ler_Gs_Cgs=ref_base_wing.Ler_Gs_Cgs,
                                angles_Gs_to_Wn_ixyz=ref_base_wing.angles_Gs_to_Wn_ixyz,
                                symmetric=ref_base_wing.symmetric,
                                mirror_only=ref_base_wing.mirror_only,
                                symmetryNormal_G=ref_base_wing.symmetryNormal_G,
                                symmetryPoint_G_Cg=ref_base_wing.symmetryPoint_G_Cg,
                                chordwise_spacing=ref_base_wing.chordwise_spacing,
                            )
                            these_base_wings.append(this_base_wing)

                            # Create WingMovement.
                            this_wing_movement = movements.wing_movement.WingMovement(
                                base_wing=this_base_wing,
                                wing_cross_section_movements=these_wing_cross_section_movements,
                                rotationPointOffset_Gs_Ler=ref_wing_movement.rotationPointOffset_Gs_Ler,
                                ampLer_Gs_Cgs=ref_wing_movement.ampLer_Gs_Cgs,
                                periodLer_Gs_Cgs=ref_wing_movement.periodLer_Gs_Cgs,
                                spacingLer_Gs_Cgs=ref_wing_movement.spacingLer_Gs_Cgs,
                                phaseLer_Gs_Cgs=ref_wing_movement.phaseLer_Gs_Cgs,
                                ampAngles_Gs_to_Wn_ixyz=ref_wing_movement.ampAngles_Gs_to_Wn_ixyz,
                                periodAngles_Gs_to_Wn_ixyz=ref_wing_movement.periodAngles_Gs_to_Wn_ixyz,
                                spacingAngles_Gs_to_Wn_ixyz=ref_wing_movement.spacingAngles_Gs_to_Wn_ixyz,
                                phaseAngles_Gs_to_Wn_ixyz=ref_wing_movement.phaseAngles_Gs_to_Wn_ixyz,
                            )
                            these_wing_movements.append(this_wing_movement)

                        # Create Airplane.
                        ref_base_airplane = ref_airplane_movement.base_airplane
                        this_base_airplane = geometry.airplane.Airplane(
                            wings=these_base_wings,
                            name=ref_base_airplane.name,
                            Cg_GP1_CgP1=ref_base_airplane.Cg_GP1_CgP1,
                            weight=ref_base_airplane.weight,
                            s_ref=None,
                            c_ref=None,
                            b_ref=None,
                        )

                        # ------------------------------------------------------
                        # OPTIONAL: VISUALIZE MESH
                        # ------------------------------------------------------

                        if visualize_meshes:
                            ar_ok, actual_ar = _verify_panel_aspect_ratio(
                                this_base_airplane, panel_aspect_ratio
                            )
                            convergence_logger.info(
                                f"\t\t\t\t\t\tTarget AR: {panel_aspect_ratio}, "
                                f"Actual AR: {actual_ar:.2f}, OK: {ar_ok}"
                            )

                            vis_filename = (
                                f"mesh_ar{panel_aspect_ratio}"
                                f"_chord{num_chordwise_panels}.png"
                            )

                            assert visualization_dir is not None
                            vis_path = Path(visualization_dir) / vis_filename
                            _visualize_wing_mesh(
                                this_base_airplane,
                                title=(
                                    f"AR={panel_aspect_ratio}, "
                                    f"Chordwise={num_chordwise_panels}"
                                ),
                                show=False,
                                save_path=str(vis_path),
                            )

                        # ------------------------------------------------------
                        # CREATE MOVEMENT AND PROBLEM
                        # ------------------------------------------------------

                        this_airplane_movement = movements.airplane_movement.AirplaneMovement(
                            base_airplane=this_base_airplane,
                            wing_movements=these_wing_movements,
                            ampCg_GP1_CgP1=ref_airplane_movement.ampCg_GP1_CgP1,
                            periodCg_GP1_CgP1=ref_airplane_movement.periodCg_GP1_CgP1,
                            spacingCg_GP1_CgP1=ref_airplane_movement.spacingCg_GP1_CgP1,
                            phaseCg_GP1_CgP1=ref_airplane_movement.phaseCg_GP1_CgP1,
                        )

                        # Use cached optimized delta_time if available,
                        # since it depends only on panel AR and chordwise
                        # panel count.
                        dt_cache_key = (panel_aspect_ratio, num_chordwise_panels)
                        this_delta_time: str | float
                        if dt_cache_key in optimized_dt_cache:
                            this_delta_time = optimized_dt_cache[dt_cache_key]
                        else:
                            this_delta_time = "optimize"

                        if static:
                            this_movement = movements.movement.Movement(
                                airplane_movements=[this_airplane_movement],
                                operating_point_movement=ref_operating_point_movement,
                                num_chords=wake_length,
                                delta_time=this_delta_time,
                            )
                        else:
                            this_movement = movements.movement.Movement(
                                airplane_movements=[this_airplane_movement],
                                operating_point_movement=ref_operating_point_movement,
                                num_cycles=wake_length,
                                delta_time=this_delta_time,
                            )

                        # Cache the optimized delta_time for reuse.
                        if dt_cache_key not in optimized_dt_cache:
                            optimized_dt_cache[dt_cache_key] = this_movement.delta_time

                        this_problem = problems.UnsteadyProblem(
                            movement=this_movement,
                            only_final_results=True,
                        )

                        # ------------------------------------------------------
                        # RUN SOLVER
                        # ------------------------------------------------------

                        this_solver = unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                            unsteady_problem=this_problem
                        )

                        if isinstance(this_delta_time, str):
                            convergence_logger.info(
                                f"\t\t\t\t\t\tOptimized delta_time: "
                                f"{this_movement.delta_time:.6f} s"
                            )
                        else:
                            convergence_logger.info(
                                f"\t\t\t\t\t\tCached delta_time: "
                                f"{this_movement.delta_time:.6f} s"
                            )
                        convergence_logger.info("\t\t\t\t\t\tStarting simulation...")

                        iter_start = time.time()
                        this_solver.run(
                            prescribed_wake=wake,
                            calculate_streamlines=False,
                            show_progress=show_solver_progress,
                        )
                        iter_stop = time.time()
                        this_iter_time = iter_stop - iter_start

                        convergence_logger.info(
                            f"\t\t\t\t\t\tSimulation completed in "
                            f"{this_iter_time:.3f} s"
                        )

                        # ------------------------------------------------------
                        # EXTRACT RESULTS
                        # ------------------------------------------------------

                        theseFinalCoefficients = np.zeros((1, 6), dtype=float)
                        theseFinalLoads = np.zeros(6, dtype=float)

                        if static:
                            theseFinalCoefficients[0, :3] = (
                                this_problem.finalForceCoefficients_W[0]
                            )
                            theseFinalCoefficients[0, 3:] = (
                                this_problem.finalMomentCoefficients_W_CgP1[0]
                            )
                            theseFinalLoads[:3] = this_problem.finalForces_W[0]
                            theseFinalLoads[3:] = this_problem.finalMoments_W_CgP1[0]
                        else:
                            theseFinalCoefficients[0, :3] = (
                                this_problem.finalMeanForceCoefficients_W[0]
                            )
                            theseFinalCoefficients[0, 3:] = (
                                this_problem.finalMeanMomentCoefficients_W_CgP1[0]
                            )
                            theseFinalLoads[:3] = this_problem.finalMeanForces_W[0]
                            theseFinalLoads[3:] = this_problem.finalMeanMoments_W_CgP1[
                                0
                            ]

                        # Save to simulation cache.
                        if cache_file is not None:
                            simulation_cache[sim_cache_key] = {
                                "coefficients": theseFinalCoefficients[0, :].tolist(),
                                "loads": theseFinalLoads.tolist(),
                                "time": this_iter_time,
                            }
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            with _lock_cache_file(cache_file):
                                # Re-read to merge entries from other processes.
                                if cache_file.exists():
                                    with open(cache_file, "r") as f:
                                        disk_cache = json.load(f)
                                else:
                                    disk_cache = {}
                                disk_cache.update(simulation_cache)
                                disk_cache["_optimized_dt"] = {
                                    f"{k[0]},{k[1]}": v
                                    for k, v in optimized_dt_cache.items()
                                }
                                with open(cache_file, "w") as f:
                                    json.dump(disk_cache, f, indent=2)

                    # ----------------------------------------------------------
                    # STORE RESULTS
                    # ----------------------------------------------------------

                    finalCoefficients[wake_id, length_id, ar_id, chord_id, :, :] = (
                        theseFinalCoefficients
                    )
                    iter_times[wake_id, length_id, ar_id, chord_id] = this_iter_time

                    # ----------------------------------------------------------
                    # CHECK CONVERGENCE
                    # ----------------------------------------------------------

                    # Get the current coefficients as a flat (6,) array.
                    current_coefficients = theseFinalCoefficients[0, :]

                    wake_converged = False
                    length_converged = False
                    ar_converged = False
                    chord_converged = False

                    # Wake state convergence check.
                    if wake_id > 0:
                        coarser_wake_coefficients = finalCoefficients[
                            wake_id - 1,
                            length_id,
                            ar_id,
                            chord_id,
                            0,
                            :,
                        ]
                        (
                            wake_converged,
                            wake_min_metric,
                            wake_errors,
                            wake_tols,
                            wake_metrics,
                        ) = _check_coefficient_convergence(
                            current_coefficients,
                            coarser_wake_coefficients,
                            rtol,
                            atol,
                            mask=coefficient_mask_array,
                        )
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - wake type:"
                        )
                        for i, label in enumerate(_COEFFICIENT_LABELS):
                            if not coefficient_mask_array[i]:
                                continue
                            convergence_logger.info(
                                f"\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                f" {_LOAD_UNITS[i]}"
                                f", error={wake_errors[i]:.3e}"
                                f", tol={wake_tols[i]:.3e}"
                                f", metric={wake_metrics[i]:.2f}"
                            )
                        masked_wake = np.where(
                            coefficient_mask_array, wake_metrics, np.inf
                        )
                        min_label = _COEFFICIENT_LABELS[int(np.argmin(masked_wake))]
                        convergence_logger.info(
                            f"\t\t\t\t\t\t    Minimum metric: "
                            f"{wake_min_metric:.2f} ({min_label})"
                        )
                    else:
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - wake type: "
                            "not yet checked"
                        )

                    # Wake length convergence check.
                    if length_id > 0:
                        coarser_length_coefficients = finalCoefficients[
                            wake_id,
                            length_id - 1,
                            ar_id,
                            chord_id,
                            0,
                            :,
                        ]
                        (
                            length_converged,
                            length_min_metric,
                            length_errors,
                            length_tols,
                            length_metrics,
                        ) = _check_coefficient_convergence(
                            current_coefficients,
                            coarser_length_coefficients,
                            rtol,
                            atol,
                            mask=coefficient_mask_array,
                        )
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - wake length:"
                        )
                        for i, label in enumerate(_COEFFICIENT_LABELS):
                            if not coefficient_mask_array[i]:
                                continue
                            convergence_logger.info(
                                f"\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                f" {_LOAD_UNITS[i]}"
                                f", error={length_errors[i]:.3e}"
                                f", tol={length_tols[i]:.3e}"
                                f", metric={length_metrics[i]:.2f}"
                            )
                        masked_length = np.where(
                            coefficient_mask_array, length_metrics, np.inf
                        )
                        min_label = _COEFFICIENT_LABELS[int(np.argmin(masked_length))]
                        convergence_logger.info(
                            f"\t\t\t\t\t\t    Minimum metric: "
                            f"{length_min_metric:.2f} ({min_label})"
                        )
                    else:
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - wake length: "
                            "not yet checked"
                        )

                    # Panel aspect ratio convergence check.
                    if ar_id > 0:
                        coarser_ar_coefficients = finalCoefficients[
                            wake_id,
                            length_id,
                            ar_id - 1,
                            chord_id,
                            0,
                            :,
                        ]
                        (
                            ar_converged,
                            ar_min_metric,
                            ar_errors,
                            ar_tols,
                            ar_metrics,
                        ) = _check_coefficient_convergence(
                            current_coefficients,
                            coarser_ar_coefficients,
                            rtol,
                            atol,
                            mask=coefficient_mask_array,
                        )
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - Panel AR:"
                        )
                        for i, label in enumerate(_COEFFICIENT_LABELS):
                            if not coefficient_mask_array[i]:
                                continue
                            convergence_logger.info(
                                f"\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                f" {_LOAD_UNITS[i]}"
                                f", error={ar_errors[i]:.3e}"
                                f", tol={ar_tols[i]:.3e}"
                                f", metric={ar_metrics[i]:.2f}"
                            )
                        masked_ar = np.where(coefficient_mask_array, ar_metrics, np.inf)
                        min_label = _COEFFICIENT_LABELS[int(np.argmin(masked_ar))]
                        convergence_logger.info(
                            f"\t\t\t\t\t\t    Minimum metric: "
                            f"{ar_min_metric:.2f} ({min_label})"
                        )
                    else:
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - Panel AR: "
                            "not yet checked"
                        )

                    # Chordwise Panels convergence check.
                    if chord_id > 0:
                        coarser_chord_coefficients = finalCoefficients[
                            wake_id,
                            length_id,
                            ar_id,
                            chord_id - 1,
                            0,
                            :,
                        ]
                        (
                            chord_converged,
                            chord_min_metric,
                            chord_errors,
                            chord_tols,
                            chord_metrics,
                        ) = _check_coefficient_convergence(
                            current_coefficients,
                            coarser_chord_coefficients,
                            rtol,
                            atol,
                            mask=coefficient_mask_array,
                        )
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - chordwise Panels:"
                        )
                        for i, label in enumerate(_COEFFICIENT_LABELS):
                            if not coefficient_mask_array[i]:
                                continue
                            convergence_logger.info(
                                f"\t\t\t\t\t\t    {label}={current_coefficients[i]:.6e}"
                                f", {_LOAD_LABELS[i]}={theseFinalLoads[i]:.6e}"
                                f" {_LOAD_UNITS[i]}"
                                f", error={chord_errors[i]:.3e}"
                                f", tol={chord_tols[i]:.3e}"
                                f", metric={chord_metrics[i]:.2f}"
                            )
                        masked_chord = np.where(
                            coefficient_mask_array, chord_metrics, np.inf
                        )
                        min_label = _COEFFICIENT_LABELS[int(np.argmin(masked_chord))]
                        convergence_logger.info(
                            f"\t\t\t\t\t\t    Minimum metric: "
                            f"{chord_min_metric:.2f} ({min_label})"
                        )
                    else:
                        convergence_logger.info(
                            "\t\t\t\t\t\tConvergence check - chordwise Panels: "
                            "not yet checked"
                        )

                    # Check convergence conditions.
                    wake_saturated = not wake
                    ar_saturated = panel_aspect_ratio == 1

                    single_wake = len(wake_list) == 1
                    single_length = len(wake_lengths_list) == 1
                    single_ar = len(panel_aspect_ratios_list) == 1
                    single_chord = len(num_chordwise_panels_list) == 1

                    wake_passed = wake_converged or single_wake or wake_saturated
                    length_passed = length_converged or single_length
                    ar_passed = ar_converged or single_ar or ar_saturated
                    chord_passed = chord_converged or single_chord

                    # If all passed, return converged parameters.
                    if wake_passed and length_passed and ar_passed and chord_passed:
                        if single_wake:
                            converged_wake_id = wake_id
                        elif wake_converged:
                            converged_wake_id = wake_id - 1
                        else:
                            converged_wake_id = wake_id

                        if single_length:
                            converged_length_id = length_id
                        else:
                            converged_length_id = length_id - 1

                        if single_ar:
                            converged_ar_id = ar_id
                        elif ar_converged:
                            converged_ar_id = ar_id - 1
                        else:
                            converged_ar_id = ar_id

                        if single_chord:
                            converged_chord_id = chord_id
                        else:
                            converged_chord_id = chord_id - 1

                        converged_wake = wake_list[converged_wake_id]
                        converged_wake_length = wake_lengths_list[converged_length_id]
                        converged_chordwise_panels = num_chordwise_panels_list[
                            converged_chord_id
                        ]
                        converged_aspect_ratio = panel_aspect_ratios_list[
                            converged_ar_id
                        ]
                        converged_iter_time = float(
                            iter_times[
                                converged_wake_id,
                                converged_length_id,
                                converged_ar_id,
                                converged_chord_id,
                            ]
                        )

                        # Log results.
                        if single_wake or single_length or single_ar or single_chord:
                            convergence_logger.info(
                                "The analysis found a semi-converged case:"
                            )
                            if single_wake:
                                convergence_logger.warning(
                                    "Wake type convergence not checked"
                                )
                            if single_length:
                                convergence_logger.warning(
                                    "Wake length convergence not checked"
                                )
                            if single_ar:
                                convergence_logger.warning(
                                    "Panel aspect ratio convergence not checked"
                                )
                            if single_chord:
                                convergence_logger.warning(
                                    "Chordwise Panels convergence not checked"
                                )
                        else:
                            convergence_logger.info(
                                "The analysis found a converged case:"
                            )

                        convergence_logger.info("\tDelta time: optimized per iteration")

                        if converged_wake:
                            convergence_logger.info("\tWake type: prescribed")
                        else:
                            convergence_logger.info("\tWake type: free")

                        if static:
                            convergence_logger.info(
                                "\tChord lengths: " + str(converged_wake_length)
                            )
                        else:
                            convergence_logger.info(
                                "\tCycles: " + str(converged_wake_length)
                            )

                        convergence_logger.info(
                            "\tPanel aspect ratio: " + str(converged_aspect_ratio)
                        )
                        convergence_logger.info(
                            "\tChordwise Panels: " + str(converged_chordwise_panels)
                        )
                        convergence_logger.info(
                            "\tSimulation completed in "
                            + str(round(converged_iter_time, 3))
                            + " s"
                        )

                        # Log spanwise sections for each wing.
                        convergence_logger.info("\tSpanwise sections per wing:")
                        for wing_id in range(num_wings):
                            cache_key = (
                                converged_ar_id,
                                converged_chord_id,
                                wing_id,
                            )
                            num_sections = num_cross_sections_cache.get(cache_key, 0)
                            convergence_logger.info(
                                f"\t\tWing {wing_id}: {num_sections} sections"
                            )

                        return (
                            converged_wake,
                            converged_wake_length,
                            converged_aspect_ratio,
                            converged_chordwise_panels,
                        )

    # No convergence found.
    convergence_logger.info(
        "The analysis did not find a converged case within the given bounds"
    )
    return None, None, None, None
