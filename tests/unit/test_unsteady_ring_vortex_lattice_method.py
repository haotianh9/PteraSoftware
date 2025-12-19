"""This module contains a class to test the UnsteadyRingVortexLatticeMethodSolver."""

import unittest

import pterasoftware as ps
from tests.unit.fixtures import problem_fixtures


class TestUnsteadyRingVortexLatticeMethodSolver(unittest.TestCase):
    """This is a class with functions to test the UnsteadyRingVortexLatticeMethodSolver."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.basic_unsteady_problem = (
            problem_fixtures.make_basic_unsteady_problem_fixture()
        )

    def test_force_method_parameter_default(self):
        """Test that force_method defaults to 'joukowski'."""
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.basic_unsteady_problem,
        )
        self.assertEqual(solver._force_method, "joukowski")

    def test_force_method_parameter_joukowski(self):
        """Test that force_method accepts 'joukowski'."""
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.basic_unsteady_problem,
        )
        solver.run(force_method="joukowski", show_progress=False)
        self.assertEqual(solver._force_method, "joukowski")
        self.assertTrue(solver.ran)

    def test_force_method_parameter_katz(self):
        """Test that force_method accepts 'katz'."""
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.basic_unsteady_problem,
        )
        solver.run(force_method="katz", show_progress=False)
        self.assertEqual(solver._force_method, "katz")
        self.assertTrue(solver.ran)

    def test_force_method_parameter_invalid_string(self):
        """Test that force_method raises ValueError for invalid strings."""
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.basic_unsteady_problem,
        )
        invalid_values = ["invalid", "JOUKOWSKI", "Katz", "both", ""]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    solver.run(force_method=invalid, show_progress=False)

    def test_force_method_parameter_invalid_type(self):
        """Test that force_method raises TypeError for non strings."""
        solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
            unsteady_problem=self.basic_unsteady_problem,
        )
        invalid_types = [123, 1.0, None, True, ["joukowski"], {"method": "joukowski"}]
        for invalid in invalid_types:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    solver.run(force_method=invalid, show_progress=False)
