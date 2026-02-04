"""This module contains a testing case for the unsteady convergence function."""

import unittest

import numpy as np

import pterasoftware as ps
from tests.integration.fixtures import problem_fixtures


class TestUnsteadyConvergence(unittest.TestCase):
    """This is a class for testing the unsteady convergence function."""

    def setUp(self):
        """This method sets up the test.

        :return: None
        """
        self.unsteady_validation_problem = (
            problem_fixtures.make_unsteady_validation_problem_with_static_geometry()
        )

    def test_unsteady_convergence(self):
        """This method tests that the function finds pre-known convergence parameters
        for an UnsteadyRingVortexLatticeMethodSolver.

        :return: None
        """
        converged_parameters = ps.convergence.analyze_unsteady_convergence(
            ref_problem=self.unsteady_validation_problem,
            prescribed_wake=True,
            free_wake=True,
            num_chords_bounds=(1, 4),
            panel_aspect_ratio_bounds=(4, 2),
            num_chordwise_panels_bounds=(1, 5),
            convergence_criteria=5.0,
            show_solver_progress=False,
        )

        converged_wake_state = converged_parameters[0]
        converged_num_chords = converged_parameters[1]
        converged_panel_ar = converged_parameters[2]
        converged_num_chordwise = converged_parameters[3]

        wake_state_ans = True
        num_chords_ans = 2
        panel_ar_ans = 4
        num_chordwise_ans = 3

        self.assertTrue(converged_wake_state == wake_state_ans)
        self.assertTrue(abs(converged_num_chords - num_chords_ans) <= 1)
        self.assertTrue(abs(converged_panel_ar - panel_ar_ans) <= 1)
        self.assertTrue(abs(converged_num_chordwise - num_chordwise_ans) <= 1)


class TestUnsteadyConvergenceNonTrapezoidal(unittest.TestCase):
    """This is a class for testing the non-trapezoidal unsteady convergence function."""

    def setUp(self):
        """This method sets up the test.

        :return: None
        """
        # Create a simple non-trapezoidal problem with a single wing
        # that has num_spanwise_panels=1 per WCS
        wing_span = 5.0
        chord = 1.0

        # Create many WCS with num_spanwise_panels=1
        num_sections = 10
        wing_cross_sections = []
        airfoil = ps.geometry.airfoil.Airfoil(name="naca2412")
        for i in range(num_sections + 1):
            y_pos = (i / num_sections) * wing_span
            wcs = ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=airfoil,
                Lp_Wcsp_Lpp=(0.0, y_pos, 0.0),
                chord=chord,
                num_spanwise_panels=1 if i < num_sections else None,
            )
            wing_cross_sections.append(wcs)

        wing = ps.geometry.wing.Wing(
            wing_cross_sections=wing_cross_sections,
            num_chordwise_panels=3,
        )

        airplane = ps.geometry.airplane.Airplane(wings=[wing])

        # Create movements
        wcs_movements = [
            ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
                base_wing_cross_section=wcs
            )
            for wcs in wing_cross_sections
        ]
        wing_movement = ps.movements.wing_movement.WingMovement(
            base_wing=wing,
            wing_cross_section_movements=wcs_movements,
        )
        airplane_movement = ps.movements.airplane_movement.AirplaneMovement(
            base_airplane=airplane,
            wing_movements=[wing_movement],
        )
        operating_point = ps.operating_point.OperatingPoint()
        operating_point_movement = (
            ps.movements.operating_point_movement.OperatingPointMovement(
                base_operating_point=operating_point
            )
        )
        movement = ps.movements.movement.Movement(
            airplane_movements=[airplane_movement],
            operating_point_movement=operating_point_movement,
            num_chords=2,
        )

        self.non_trapezoidal_problem = ps.problems.UnsteadyProblem(movement=movement)

        # Store geometry info for the resampler
        self.wing_span = wing_span
        self.chord = chord

    def _wing_geometry_resampler(self, wing_id: int, num_sections: int) -> np.ndarray:
        """Simple resampler for a rectangular wing.

        :param wing_id: The wing ID (unused since there's only one wing).
        :param num_sections: The number of spanwise sections.
        :return: An (N+1, 4) array with [dx, dy, dz, chord] for each cross section.
        """
        result = np.zeros((num_sections + 1, 4))
        for i in range(num_sections + 1):
            y_pos = (i / num_sections) * self.wing_span
            result[i, :] = [0.0, y_pos, 0.0, self.chord]
        return result

    def test_non_trapezoidal_convergence_with_delta_time_bounds(self):
        """This method tests the non-trapezoidal convergence function with
        delta_time_bounds parameter.

        :return: None
        """
        converged_parameters = (
            ps.convergence.analyze_unsteady_convergence_non_trapezoidal(
                ref_problem=self.non_trapezoidal_problem,
                wing_geometry_resampler=self._wing_geometry_resampler,
                prescribed_wake=True,
                free_wake=False,
                num_chords_bounds=(1, 2),
                panel_aspect_ratio_bounds=(4, 2),
                num_chordwise_panels_bounds=(2, 4),
                delta_time_bounds=(0.04, 0.02),
                rtol=0.20,
                atol=0.001,
                show_solver_progress=False,
            )
        )

        # Verify the return type has 5 elements
        self.assertEqual(len(converged_parameters), 5)

        converged_delta_time = converged_parameters[0]
        converged_wake_state = converged_parameters[1]
        converged_num_chords = converged_parameters[2]
        converged_panel_ar = converged_parameters[3]
        converged_num_chordwise = converged_parameters[4]

        # Check that results are not None (convergence was found)
        self.assertIsNotNone(converged_delta_time)
        self.assertIsNotNone(converged_wake_state)
        self.assertIsNotNone(converged_num_chords)
        self.assertIsNotNone(converged_panel_ar)
        self.assertIsNotNone(converged_num_chordwise)

        # Check that converged_delta_time is in the expected range
        self.assertGreaterEqual(converged_delta_time, 0.02)
        self.assertLessEqual(converged_delta_time, 0.04)

    def test_non_trapezoidal_convergence_without_delta_time_bounds(self):
        """This method tests the non-trapezoidal convergence function without
        delta_time_bounds parameter (backward compatibility).

        :return: None
        """
        converged_parameters = (
            ps.convergence.analyze_unsteady_convergence_non_trapezoidal(
                ref_problem=self.non_trapezoidal_problem,
                wing_geometry_resampler=self._wing_geometry_resampler,
                prescribed_wake=True,
                free_wake=False,
                num_chords_bounds=(1, 2),
                panel_aspect_ratio_bounds=(4, 2),
                num_chordwise_panels_bounds=(2, 4),
                rtol=0.10,
                atol=0.001,
                show_solver_progress=False,
            )
        )

        # Verify the return type has 5 elements
        self.assertEqual(len(converged_parameters), 5)

        converged_delta_time = converged_parameters[0]

        # When delta_time_bounds is not provided, converged_delta_time should be None
        self.assertIsNone(converged_delta_time)

    def test_delta_time_and_delta_time_bounds_mutual_exclusion(self):
        """This method tests that delta_time and delta_time_bounds cannot both be
        provided.

        :return: None
        """
        with self.assertRaises(ValueError) as context:
            ps.convergence.analyze_unsteady_convergence_non_trapezoidal(
                ref_problem=self.non_trapezoidal_problem,
                wing_geometry_resampler=self._wing_geometry_resampler,
                prescribed_wake=True,
                free_wake=False,
                num_chords_bounds=(1, 2),
                panel_aspect_ratio_bounds=(4, 2),
                num_chordwise_panels_bounds=(2, 4),
                delta_time=0.01,
                delta_time_bounds=(0.04, 0.02),
                rtol=0.10,
                atol=0.001,
                show_solver_progress=False,
            )

        self.assertIn(
            "delta_time and delta_time_bounds cannot both be provided",
            str(context.exception),
        )
