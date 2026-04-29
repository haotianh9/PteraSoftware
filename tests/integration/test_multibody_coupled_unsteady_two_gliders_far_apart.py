"""Validate far-field multibody free flight against the single-glider case."""

from __future__ import annotations

import unittest

from examples import multibody_two_gliding_wings_free_flight as validation_case


class TestMultiBodyTwoGlidersFarApart(unittest.TestCase):
    """The free-wake multibody path should preserve two-body symmetry."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the short validation solves once for all tests."""
        cls.summary = validation_case.run_validation(
            output_dir=validation_case.DEFAULT_OUTPUT_DIR / "_test_artifacts",
            lateral_separation_m=validation_case.DEFAULT_LATERAL_SEPARATION_M,
            prescribed_num_steps=8,
            free_num_steps=16,
            steps_per_reference_period=8,
        )

    def test_body_symmetry_flag(self) -> None:
        """The summary should mark the two-body symmetry check as passing."""
        self.assertTrue(self.summary["passes_body_symmetry_check"])

    def test_velocity_histories_match_between_bodies(self) -> None:
        """The two bodies should remain dynamically symmetric."""
        self.assertLess(self.summary["max_interbody_velocity_delta_mps"], 1e-3)

    def test_position_offsets_remain_constant_between_bodies(self) -> None:
        """The lateral separation should remain constant to tight tolerance."""
        self.assertLess(self.summary["max_interbody_offset_drift_m"], 1e-5)


if __name__ == "__main__":
    unittest.main()
