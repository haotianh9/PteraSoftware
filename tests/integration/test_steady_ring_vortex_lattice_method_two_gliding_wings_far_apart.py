"""Validate that two far-separated gliders reproduce the single-glider result."""

from __future__ import annotations

import unittest

from examples import multibody_two_gliding_wings_validation as validation_case


class TestTwoGlidingWingsFarApart(unittest.TestCase):
    """This class tests the phase-0 multibody far-field equivalence check."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the single- and two-glider validation solves once for all tests."""
        cls.summary = validation_case.run_validation(
            output_dir=validation_case.DEFAULT_OUTPUT_DIR / "_test_artifacts",
            separation_m=validation_case.DEFAULT_SEPARATION_M,
            speed_mps=validation_case.DEFAULT_SPEED_MPS,
            alpha_deg=validation_case.DEFAULT_ALPHA_DEG,
            beta_deg=validation_case.DEFAULT_BETA_DEG,
        )

    def test_far_field_equivalence_flag(self) -> None:
        """The summary should mark the far-field equivalence check as passing."""
        self.assertTrue(self.summary["passes_far_field_equivalence_check"])

    def test_force_coefficients_match_single_case(self) -> None:
        """Each glider in the far-separated pair should match the single-glider load."""
        for pair_glider in self.summary["pair_gliders"]:
            self.assertLess(
                pair_glider["force_coefficients_relative_error_norm_vs_single"], 1e-3
            )

    def test_moment_coefficients_match_single_case(self) -> None:
        """Each glider in the far-separated pair should match the single-glider moment."""
        for pair_glider in self.summary["pair_gliders"]:
            self.assertLess(
                pair_glider["moment_coefficients_relative_error_norm_vs_single"], 1e-3
            )


if __name__ == "__main__":
    unittest.main()
