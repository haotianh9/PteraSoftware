"""Smoke test for two-body inline flapping free-flight gap sweep."""

from __future__ import annotations

import unittest

from examples import multibody_two_flapping_forward_inline_gap_sweep as sweep_case


class TestMultiBodyTwoFlappingInlineGapSweep(unittest.TestCase):
    """Validate that the two-flapping-body inline sweep runs and stays symmetric."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one short far-apart case once for all tests."""
        cls.sweep_summary = sweep_case.run_gap_sweep(
            output_root=sweep_case.DEFAULT_OUTPUT_ROOT / "_test_artifacts",
            gaps_m=(2.0,),
            gap_axis="x",
            prescribed_num_steps=4,
            free_num_steps=8,
            steps_per_flap=8,
            show_progress=False,
            prescribed_wake=False,
            history_stride=1,
            save_every_n_steps=None,
            history_save_dir=None,
            standard_render_wake=False,
            standard_render_follow_body_index=0,
            clamp_yaw_deg=0.0,
            trajectory_render=False,
        )
        cls.summary = cls.sweep_summary["gap_results"][0]

    def test_sweep_returns_single_gap_result(self) -> None:
        """The requested one-gap sweep should return one summary."""
        self.assertEqual(len(self.sweep_summary["gap_results"]), 1)

    def test_body_symmetry_flag(self) -> None:
        """The short far-apart case should preserve body symmetry."""
        self.assertTrue(self.summary["passes_body_symmetry_check"])

    def test_interbody_offset_drift_is_small(self) -> None:
        """The body-to-body offset should stay almost constant."""
        self.assertLess(self.summary["max_interbody_offset_drift_m"], 5e-4)


if __name__ == "__main__":
    unittest.main()
