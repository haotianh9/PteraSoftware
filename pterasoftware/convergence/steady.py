"""Contains the steady convergence analysis function.

**Contains the following classes:**

None

**Contains the following functions:**

analyze_steady_convergence: Finds the converged parameters of a SteadyProblem solved
using a given steady solver.
"""

from __future__ import annotations

import time

import numpy as np

from .. import (
    _parameter_validation,
    geometry,
    problems,
    steady_horseshoe_vortex_lattice_method,
    steady_ring_vortex_lattice_method,
)
from ._functions import convergence_logger
from .unsteady import _get_wing_section_num_spanwise_panels


# TEST: Consider adding unit tests for this function.
# TEST: Assess how comprehensive this function's integration tests are and update or
#  extend them if needed.
# TODO: If a converged mesh was found, consider also returning the converged solver.
def analyze_steady_convergence(
    ref_problem: problems.SteadyProblem,
    solver_type: str,
    panel_aspect_ratio_bounds: tuple[int, int] = (4, 1),
    num_chordwise_panels_bounds: tuple[int, int] = (3, 12),
    convergence_criteria: float | int = 5.0,
) -> tuple[int, int] | tuple[None, None]:
    """Finds the converged parameters of a SteadyProblem solved using a given steady
    solver.

    **Procedure:**

    Convergence is found by varying the SteadyProblem's Airplanes' Panels' aspect ratios
    their Wings' numbers of chordwise Panels. These values are iterated over via two
    nested for loops (with the number of chordwise Panels as the inner loop).

    With each new combination of these values, the SteadyProblem is solved, and its
    resultant load coefficients are stored. Each Airplanes' three force coefficients are
    combined by taking their root-sum-square to find the resultant force coefficient.
    Then, the absolute percent change (APE) of each Airplanes' resultant force
    coefficient is found between this iteration, and the iterations with incrementally
    coarser meshes in the two parameter directions (panel aspect ratio and number of
    chordwise panels). These two steps are repeated for the three moment coefficients.

    The maximums of the resultant force coefficient APEs and resultant moment
    coefficient APEs are found. This leaves us with two maximum APEs, one for each
    parameter direction, per Airplane. Next, we take the maximum of each parameter
    directions' APEs across all Airplanes, leaving us with two maximum APEs total. If
    either of the parameter direction APEs is below the convergence criteria, then this
    iteration has found a converged solution for that parameter direction.

    If an iteration's two APEs are both below the converged criteria, then we exit the
    nested for loops and return the converged parameters. However, the converged
    parameters are actually the values incrementally coarser than the final values
    (because the incrementally coarser values were found to be within the convergence
    criteria percent difference from the final values).

    **Notes:**

    There are two edge cases to this function. The first is if the user inputs equal
    values for the coarsest and finest values of either the Panel aspect ratio or the
    number of chordwise Panels (e.g. panel_aspect_ratio_bounds=(2, 2)). Then, this
    parameter will not be iterated over, and convergence will only be checked for the
    other parameter.

    The second edge case happens if the Panel aspect ratio has not converged at a value
    of 1. This is the gold standard value for Panel aspect ratio, so this function will
    return 1 for the converged value of Panel aspect ratio. In the code below, this
    state is referred to as a "saturated" Panel aspect ratio case.

    :param ref_problem: The SteadyProblem whose converged parameters will be found.
    :param solver_type: Determines what type of steady solver will be used to analyze
        the SteadyProblem. The options are "steady horseshoe vortex lattice method" and
        "steady ring vortex lattice method".
    :param panel_aspect_ratio_bounds: A tuple of two ints, in descending order, that
        determines the range of Panel aspect ratios to consider, from largest to
        smallest. This value dictates the Panels' average y component length (in wing
        cross section parent axes) divided their average x component width (in wing
        cross section parent axes). Historically, these values range between 5 and 1.
        Values above 5 can be used for a coarser mesh, but the minimum value cannot be
        less than 1. The default is (4, 1).
    :param num_chordwise_panels_bounds: A tuple of two ints, in ascending order, that
        determines the range of values to use for the Wings' numbers of chordwise
        panels. The default is (3, 12).
    :param convergence_criteria: A positive number (int or float) that determines the
        point at which the function considers the simulation to have converged.
        Specifically, it is the maximum absolute percent change in the combined load
        coefficients. Therefore, it is in units of percent. Refer to the description in
        this function's docstring for more details on how it affects the solver. In
        short, set this value to 5.0 for a lenient convergence, and 1.0 for a strict
        convergence. Values are converted to floats internally. The default is 5.0.
    :return: A tuple of two ints or a tuple of two Nones. In order, they are the
        converged of Panel aspect ratio and the converged number of chordwise Panels. If
        the function could not find a set of converged parameters, it returns (None,
        None).
    """
    # Validate the ref_problem parameter.
    if not isinstance(ref_problem, problems.SteadyProblem):
        raise TypeError("ref_problem must be a SteadyProblem.")

    # Validate the solver_type parameter.
    if solver_type not in (
        "steady horseshoe vortex lattice method",
        "steady ring vortex lattice method",
    ):
        raise ValueError(
            'solver_type must be either "steady horseshoe vortex lattice method" or '
            '"steady ring vortex lattice method".'
        )

    # Validate the panel_aspect_ratio_bounds parameter.
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

    # Validate the num_chordwise_panels_bounds parameter.
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

    # Validate the convergence_criteria parameter.
    convergence_criteria = _parameter_validation.number_in_range_return_float(
        convergence_criteria, "convergence_criteria", min_val=0.0, min_inclusive=False
    )

    convergence_logger.info("Beginning convergence analysis...")

    ref_operating_point = ref_problem.operating_point
    ref_airplanes = ref_problem.airplanes

    # Create lists containing each Panel aspect ratio and each number of chordwise
    # Panels to test.
    panel_aspect_ratios_list = list(
        range(panel_aspect_ratio_bounds[0], panel_aspect_ratio_bounds[1] - 1, -1)
    )
    num_chordwise_panels_list = list(
        range(num_chordwise_panels_bounds[0], num_chordwise_panels_bounds[1] + 1)
    )

    # Initialize some empty ndarrays to hold variables related to each iteration.
    # Going forward, an "iteration" refers to a SteadyProblem containing one of the
    # combinations of Panel aspect ratio and number of chordwise Panels.
    iter_times = np.zeros(
        (len(panel_aspect_ratios_list), len(num_chordwise_panels_list)), dtype=float
    )
    combinedForceCoefficients = np.zeros(
        (
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
            len(ref_airplanes),
        ),
        dtype=float,
    )
    combinedMomentCoefficients = np.zeros(
        (
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
            len(ref_airplanes),
        ),
        dtype=float,
    )

    iteration = 0
    num_iterations = len(panel_aspect_ratios_list) * len(num_chordwise_panels_list)

    # This is a cache to store previously calculated numbers of spanwise Panels for
    # specific combinations of parameters to avoid redundant calculations. The key is
    # a tuple of 5 ints: ar_id, chord_id, ref_airplane_id, ref_wing_id,
    # ref_wing_cross_section_id,
    num_spanwise_panels_cache: dict[tuple[int, int, int, int, int], int] = {}

    # Begin iterating through the outer loop of Panel aspect ratios.
    for ar_id, panel_aspect_ratio in enumerate(panel_aspect_ratios_list):
        convergence_logger.info("\tPanel aspect ratio: " + str(panel_aspect_ratio))

        # Begin iterating through the inner loop of number of chordwise Panels.
        for chord_id, num_chordwise_panels in enumerate(num_chordwise_panels_list):
            convergence_logger.info(
                "\t\tChordwise Panels: " + str(num_chordwise_panels)
            )

            iteration += 1
            convergence_logger.info(
                "\t\t\tIteration number: " + str(iteration) + "/" + str(num_iterations)
            )

            # Initialize an empty list to hold this iteration's Airplanes. Then,
            # fill the list by making new copies of each of the Airplanes with
            # modified values for Panel aspect ratio and number of chordwise Panels.
            these_airplanes = []
            for ref_airplane_id, ref_airplane in enumerate(ref_airplanes):
                ref_wings = ref_airplane.wings
                these_wings = []

                for ref_wing_id, ref_wing in enumerate(ref_wings):
                    ref_wing_cross_sections = ref_wing.wing_cross_sections
                    these_wing_cross_sections = []

                    for (
                        ref_wing_cross_section_id,
                        ref_wing_cross_section,
                    ) in enumerate(ref_wing_cross_sections):

                        # If this is not the last WingCrossSection, find the number
                        # of spanwise Panels to use for this section of the Wing,
                        # based on the desired Panel aspect ratio and number of
                        # chordwise Panels.
                        if ref_wing_cross_section_id < (
                            len(ref_wing_cross_sections) - 1
                        ):
                            # Check if we've already calculated the number of
                            # spanwise Panels for this case/combination of parameters.
                            num_spanwise_panels_key = (
                                ar_id,
                                chord_id,
                                ref_airplane_id,
                                ref_wing_id,
                                ref_wing_cross_section_id,
                            )
                            if num_spanwise_panels_key in num_spanwise_panels_cache:
                                convergence_logger.debug(
                                    f"\t\t\t\tGetting the cached number of spanwise "
                                    f"Panels calculated for the #"
                                    f"{ref_wing_cross_section_id + 1} "
                                    f"WingCrossSection of {ref_airplane.name}'s "
                                    f"{ref_wing.name}..."
                                )

                                this_num_spanwise_panels = num_spanwise_panels_cache[
                                    num_spanwise_panels_key
                                ]
                            else:
                                # The way we calculate the correct number of spanwise
                                # Panels is to make skeleton Airplanes containing
                                # only one Wing with only the two WingCrossSections
                                # that make up the current Wing section. During
                                # initialization, the Airplane meshes its Wing,
                                # and we can then access the Wing's
                                # average_panel_aspect_ratio property. We repeat this
                                # process with increasing numbers of spanwise Panels,
                                # until we find the value that results in
                                # average_panel_aspect_ratio most closely matches the
                                # desired Panel aspect ratio. Initially, the first
                                # skeleton Airplane uses num_spanwise_panels=1.
                                # However, if we've already calculated a number of
                                # spanwise Panels for this Wing section with a
                                # coarser mesh (either in Panel aspect ratio,
                                # number of chordwise Panels, or both), then we know
                                # the current mesh must use at least this many
                                # spanwise Panels. Therefore, we can start the
                                # iterations with a higher number of spanwise Panels.
                                starting_num_spanwise_panels = 1

                                # Get the keys for the three coarser cases.
                                last_ar_key = (
                                    ar_id - 1,
                                    chord_id,
                                    ref_airplane_id,
                                    ref_wing_id,
                                    ref_wing_cross_section_id,
                                )
                                last_chord_key = (
                                    ar_id,
                                    chord_id - 1,
                                    ref_airplane_id,
                                    ref_wing_id,
                                    ref_wing_cross_section_id,
                                )
                                last_ar_and_chord_key = (
                                    ar_id - 1,
                                    chord_id - 1,
                                    ref_airplane_id,
                                    ref_wing_id,
                                    ref_wing_cross_section_id,
                                )

                                # Initialize the three coarser cases number of
                                # spanwise to be infinity, and update them if they
                                # exist in the cache.
                                last_ar_cache_val = np.inf
                                if last_ar_key in num_spanwise_panels_cache:
                                    last_ar_cache_val = num_spanwise_panels_cache[
                                        last_ar_key
                                    ]
                                last_chord_cache_val = np.inf
                                if last_chord_key in num_spanwise_panels_cache:
                                    last_chord_cache_val = num_spanwise_panels_cache[
                                        last_chord_key
                                    ]
                                last_ar_and_chord_cache_val = np.inf
                                if last_ar_and_chord_key in num_spanwise_panels_cache:
                                    last_ar_and_chord_cache_val = (
                                        num_spanwise_panels_cache[last_ar_and_chord_key]
                                    )

                                # To be conservative, take the minimum
                                # num_spanwise_panels of the three coarser cases. If
                                # at least one of the three cases has already been
                                # calculated, use that num_spanwise_panels as the
                                # starting value instead of 1.
                                last_cache_val = min(
                                    last_ar_cache_val,
                                    last_chord_cache_val,
                                    last_ar_and_chord_cache_val,
                                )
                                if last_cache_val != np.inf:
                                    starting_num_spanwise_panels = int(last_cache_val)

                                next_ref_wing_cross_section = ref_wing_cross_sections[
                                    ref_wing_cross_section_id + 1
                                ]

                                convergence_logger.debug(
                                    f"\t\t\t\tCalculating the number of spanwise "
                                    f"Panels for the #{ref_wing_cross_section_id + 1} "
                                    f"WingCrossSection of {ref_airplane.name}'s "
                                    f"{ref_wing.name}, with a starting value of "
                                    f"{starting_num_spanwise_panels}..."
                                )

                                # Iteratively find the correct number of spanwise
                                # Panels.
                                this_num_spanwise_panels = (
                                    _get_wing_section_num_spanwise_panels(
                                        panel_aspect_ratio,
                                        num_chordwise_panels,
                                        ref_wing.chordwise_spacing,
                                        ref_wing_cross_section,
                                        next_ref_wing_cross_section,
                                        starting_num_spanwise_panels,
                                    )
                                )

                                # Cache the calculated number of spanwise Panels for
                                # future use.
                                num_spanwise_panels_cache[num_spanwise_panels_key] = (
                                    this_num_spanwise_panels
                                )

                            convergence_logger.debug(
                                f"\t\t\t\tNumber of spanwise Panels: "
                                f"{this_num_spanwise_panels}"
                            )
                        else:
                            this_num_spanwise_panels = None

                        these_wing_cross_sections.append(
                            geometry.wing_cross_section.WingCrossSection(
                                # These values are copied from the reference
                                # WingCrossSection.
                                chord=ref_wing_cross_section.chord,
                                Lp_Wcsp_Lpp=ref_wing_cross_section.Lp_Wcsp_Lpp,
                                angles_Wcsp_to_Wcs_ixyz=ref_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
                                control_surface_symmetry_type=ref_wing_cross_section.control_surface_symmetry_type,
                                control_surface_hinge_point=ref_wing_cross_section.control_surface_hinge_point,
                                control_surface_deflection=ref_wing_cross_section.control_surface_deflection,
                                spanwise_spacing=ref_wing_cross_section.spanwise_spacing,
                                # These values change.
                                num_spanwise_panels=this_num_spanwise_panels,
                                airfoil=geometry.airfoil.Airfoil(
                                    name=ref_wing_cross_section.airfoil.name,
                                    outline_A_lp=ref_wing_cross_section.airfoil.outline_A_lp,
                                    resample=ref_wing_cross_section.airfoil.resample,
                                    n_points_per_side=ref_wing_cross_section.airfoil.n_points_per_side,
                                ),
                            )
                        )

                    these_wings.append(
                        geometry.wing.Wing(
                            # These values are copied from the reference Wing.
                            name=ref_wing.name,
                            Ler_Gs_Cgs=ref_wing.Ler_Gs_Cgs,
                            angles_Gs_to_Wn_ixyz=ref_wing.angles_Gs_to_Wn_ixyz,
                            symmetric=ref_wing.symmetric,
                            mirror_only=ref_wing.mirror_only,
                            symmetryNormal_G=ref_wing.symmetryNormal_G,
                            symmetryPoint_G_Cg=ref_wing.symmetryPoint_G_Cg,
                            chordwise_spacing=ref_wing.chordwise_spacing,
                            # These values change.
                            wing_cross_sections=these_wing_cross_sections,
                            num_chordwise_panels=num_chordwise_panels,
                        )
                    )

                these_airplanes.append(
                    geometry.airplane.Airplane(
                        # These values are copied from the reference Airplane.
                        name=ref_airplane.name,
                        Cg_GP1_CgP1=ref_airplane.Cg_GP1_CgP1,
                        weight=ref_airplane.weight,
                        # These values change.
                        wings=these_wings,
                        s_ref=None,
                        c_ref=None,
                        b_ref=None,
                    )
                )

            # Create a new SteadyProblem for this iteration.
            this_problem = problems.SteadyProblem(
                airplanes=these_airplanes, operating_point=ref_operating_point
            )

            # Create this iteration's steady solver based on the type specified.
            this_solver: (
                steady_horseshoe_vortex_lattice_method.SteadyHorseshoeVortexLatticeMethodSolver
                | steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver
            )
            if solver_type == "steady horseshoe vortex lattice method":
                this_solver = steady_horseshoe_vortex_lattice_method.SteadyHorseshoeVortexLatticeMethodSolver(
                    steady_problem=this_problem,
                )
            else:
                this_solver = steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver(
                    steady_problem=this_problem,
                )

            convergence_logger.info("\t\t\tStarting simulation...")

            # Run the steady solver and time how long it takes to execute.
            iter_start = time.time()
            this_solver.run()
            iter_stop = time.time()
            this_iter_time = iter_stop - iter_start

            # Create and fill ndarrays with each of this iteration's Airplanes'
            # combined load coefficients.
            theseCombinedForceCoefficients = np.zeros(len(these_airplanes), dtype=float)
            theseCombinedMomentCoefficients = np.zeros(
                len(these_airplanes), dtype=float
            )

            for airplane_id, airplane in enumerate(these_airplanes):
                _forceCoefficients_W = airplane.forceCoefficients_W
                assert _forceCoefficients_W is not None

                theseCombinedForceCoefficients[airplane_id] = np.linalg.norm(
                    _forceCoefficients_W
                )

                _momentCoefficients_W_CgP1 = airplane.momentCoefficients_W_CgP1
                assert _momentCoefficients_W_CgP1 is not None

                theseCombinedMomentCoefficients[airplane_id] = np.linalg.norm(
                    _momentCoefficients_W_CgP1
                )

            # Populate the ndarrays that store information from all the iterations with
            # the data from this iteration.
            combinedForceCoefficients[ar_id, chord_id, :] = (
                theseCombinedForceCoefficients
            )
            combinedMomentCoefficients[ar_id, chord_id, :] = (
                theseCombinedMomentCoefficients
            )
            iter_times[ar_id, chord_id] = this_iter_time

            convergence_logger.info(
                "\t\t\tSimulation completed in " + str(round(this_iter_time, 3)) + " s"
            )

            max_ar_pc = np.inf
            max_chord_pc = np.inf

            # If this isn't the first Panel aspect ratio, calculate the Panel aspect
            # ratio APE.
            if ar_id > 0:
                lastArCombinedForceCoefficients = combinedForceCoefficients[
                    ar_id - 1, chord_id, :
                ]
                lastArCombinedMomentCoefficients = combinedMomentCoefficients[
                    ar_id - 1, chord_id, :
                ]
                max_ar_force_pc = max(
                    100
                    * np.abs(
                        (
                            theseCombinedForceCoefficients
                            - lastArCombinedForceCoefficients
                        )
                        / lastArCombinedForceCoefficients
                    )
                )
                max_ar_moment_pc = max(
                    100
                    * np.abs(
                        (
                            theseCombinedMomentCoefficients
                            - lastArCombinedMomentCoefficients
                        )
                        / lastArCombinedMomentCoefficients
                    )
                )
                max_ar_pc = max(max_ar_force_pc, max_ar_moment_pc)

                convergence_logger.info(
                    "\t\t\tMaximum combined coefficient change from Panel aspect "
                    "ratio: " + str(round(max_ar_pc, 2)) + "%"
                )
            else:
                convergence_logger.info(
                    "\t\t\tMaximum combined coefficient change from Panel aspect "
                    "ratio: " + str(max_ar_pc)
                )

            # If this isn't the first number of chordwise Panels, calculate the
            # number of chordwise Panels APE.
            if chord_id > 0:
                lastChordCombinedForceCoefficients = combinedForceCoefficients[
                    ar_id, chord_id - 1, :
                ]
                lastChordCombinedMomentCoefficients = combinedMomentCoefficients[
                    ar_id, chord_id - 1, :
                ]
                max_chord_force_pc = max(
                    100
                    * np.abs(
                        (
                            theseCombinedForceCoefficients
                            - lastChordCombinedForceCoefficients
                        )
                        / lastChordCombinedForceCoefficients
                    )
                )
                max_chord_moment_pc = max(
                    100
                    * np.abs(
                        (
                            theseCombinedMomentCoefficients
                            - lastChordCombinedMomentCoefficients
                        )
                        / lastChordCombinedMomentCoefficients
                    )
                )
                max_chord_pc = max(max_chord_force_pc, max_chord_moment_pc)

                convergence_logger.info(
                    "\t\t\tMaximum combined coefficient change from number of "
                    "chordwise Panels: " + str(round(max_chord_pc, 2)) + "%"
                )
            else:
                convergence_logger.info(
                    "\t\t\tMaximum combined coefficient change from number of "
                    "chordwise Panels: " + str(max_chord_pc)
                )

            # Consider the Panel aspect ratio value to be saturated if it is equal to
            # 1. This is because a Panel aspect ratio of 1 is considered the maximum
            # degree of fineness.
            ar_saturated = panel_aspect_ratio == 1

            # Check if only one value for either the Panel aspect ratio or the number
            # of chordwise Panels were specified.
            single_ar = len(panel_aspect_ratios_list) == 1
            single_chord = len(num_chordwise_panels_list) == 1

            # Check if this iteration is converged with respect to the Panel aspect
            # ratio and/or the number of chordwise Panels.
            ar_converged = max_ar_pc < convergence_criteria
            chord_converged = max_chord_pc < convergence_criteria

            # Consider each convergence parameter to have "passed" if it is
            # converged, single, or saturated.
            ar_passed = ar_converged or single_ar or ar_saturated
            chord_passed = chord_converged or single_chord

            # If both convergence parameters have passed, then a converged or
            # semi-converged combination of parameters has been found and will be
            # returned.
            if ar_passed and chord_passed:
                if single_ar:
                    converged_ar_id = ar_id
                else:
                    # More than one Panel aspect ratio was tested.
                    if ar_converged:
                        # There is no big difference between this Panel aspect ratio
                        # and the last (coarser) Panel aspect ratio. Therefore,
                        # the last (coarser) Panel aspect ratio is converged.
                        converged_ar_id = ar_id - 1
                    else:
                        # There is a big difference between this Panel aspect ratio
                        # and the last (coarser) Panel aspect ratio. However,
                        # the Panel aspect ratio is one, so it's saturated.
                        # Therefore, this Panel aspect ratio is converged.
                        converged_ar_id = ar_id

                if single_chord:
                    converged_chord_id = chord_id
                else:
                    converged_chord_id = chord_id - 1

                converged_aspect_ratio = panel_aspect_ratios_list[converged_ar_id]
                converged_chordwise_panels = num_chordwise_panels_list[
                    converged_chord_id
                ]
                converged_iter_time = float(
                    iter_times[converged_ar_id, converged_chord_id]
                )

                if single_ar or single_chord:
                    convergence_logger.info("The analysis found a semi-converged case:")
                    if single_ar:
                        convergence_logger.warning(
                            "Panel aspect ratio convergence was not checked"
                        )
                    if single_chord:
                        convergence_logger.warning(
                            "Chordwise panels convergence was not checked"
                        )
                else:
                    convergence_logger.info("The analysis found a converged case:")

                convergence_logger.info(
                    "\tPanel aspect ratio: " + str(converged_aspect_ratio)
                )
                convergence_logger.info(
                    "\tChordwise Panels: " + str(converged_chordwise_panels)
                )
                convergence_logger.info(
                    "\tSimulation time: " + str(round(converged_iter_time, 3)) + " s"
                )
                convergence_logger.info("\tSpanwise Panels:")
                for airplane_id, airplane in enumerate(ref_airplanes):
                    convergence_logger.info("\t\t" + airplane.name + ":")
                    for wing_id, wing in enumerate(airplane.wings):
                        convergence_logger.info("\t\t\t" + wing.name + ":")
                        for wing_cross_section_id, wing_cross_section in enumerate(
                            wing.wing_cross_sections
                        ):
                            if (
                                wing_cross_section_id
                                < len(wing.wing_cross_sections) - 1
                            ):
                                # Not the last WingCrossSection, retrieve from cache.
                                num_spanwise_panels_key = (
                                    converged_ar_id,
                                    converged_chord_id,
                                    airplane_id,
                                    wing_id,
                                    wing_cross_section_id,
                                )
                                num_spanwise_panels = num_spanwise_panels_cache[
                                    num_spanwise_panels_key
                                ]
                            else:
                                # Last WingCrossSection.
                                num_spanwise_panels = None
                            convergence_logger.info(
                                "\t\t\t\tWingCrossSection "
                                + str(wing_cross_section_id + 1)
                                + ": "
                                + str(num_spanwise_panels)
                            )

                return (
                    converged_aspect_ratio,
                    converged_chordwise_panels,
                )

    # If all iterations have been checked and none of them resulted in both
    # convergence parameters passing, then indicate that no converged case was found
    # and return values of None for the converged parameters.
    convergence_logger.info(
        "The analysis did not find a converged case within the given bounds"
    )
    return None, None
