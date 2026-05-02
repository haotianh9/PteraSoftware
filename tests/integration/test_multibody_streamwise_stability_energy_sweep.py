"""Smoke test for the streamwise-only multibody formation sweep."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from examples import multibody_streamwise_stability_energy_sweep as sweep_case


class TestStreamwiseStabilityEnergySweep(unittest.TestCase):
    """Validate that a tiny free-wake streamwise-only case runs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one short case once for all assertions."""
        cls._temp_dir = TemporaryDirectory()
        cls.summary = sweep_case.run_streamwise_case(
            output_dir=Path(cls._temp_dir.name) / "streamwise_smoke",
            x_over_span=1.0,
            y_over_span=1.0,
            z_over_span=0.0,
            prescribed_num_steps=2,
            free_num_steps=4,
            steps_per_flap=4,
            show_progress=False,
            history_stride=1,
            save_every_n_steps=None,
            history_save_dir=None,
            render_wake_movie=False,
            baseline_power_W=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary output artifacts."""
        cls._temp_dir.cleanup()

    def test_case_uses_free_wake(self) -> None:
        """The new validation case must not silently switch to prescribed wake."""
        self.assertFalse(self.summary["prescribed_wake"])
        self.assertEqual(self.summary["wake_model"], "free")

    def test_constrained_coordinates_remain_clamped(self) -> None:
        """Y/Z and attitude should remain fixed under the streamwise clamp."""
        self.assertLess(self.summary["max_yz_drift_m"], 1e-10)
        self.assertLess(self.summary["max_euler_deviation_deg"], 1e-8)

    def test_clamp_reactions_are_reported(self) -> None:
        """Every streamwise run should expose clamp force and torque diagnostics."""
        self.assertIn("clamp_force_yz_stats_N", self.summary)
        self.assertIn("clamp_torque_stats_Nm", self.summary)
        self.assertIn("body_0_Fz_rms", self.summary["clamp_force_yz_stats_N"])
        self.assertIn("body_1_My_rms", self.summary["clamp_torque_stats_Nm"])


if __name__ == "__main__":
    unittest.main()
