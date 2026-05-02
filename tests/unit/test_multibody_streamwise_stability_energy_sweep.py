"""Unit tests for the streamwise formation stability example helpers."""

from __future__ import annotations

import unittest

import numpy as np

from examples import multibody_streamwise_stability_energy_sweep as sweep_case


class _FakeMuJoCoModel:
    """Minimal fake model for diagnostics-only tests."""

    num_bodies = 2


class TestStreamwiseStabilityEnergyHelpers(unittest.TestCase):
    """Validate coordinate and clamp-diagnostic helper behavior."""

    def test_body_positions_follow_paper_coordinate_convention(self) -> None:
        """X=x2-x1, Y=y1-y2, Z=z1-z2 maps body 2 to [X, -Y, -Z]."""
        body_1, body_2 = sweep_case.body_positions_from_paper_offsets(
            x_over_span=2.0,
            y_over_span=0.75,
            z_over_span=-0.25,
            span_m=0.4,
        )

        np.testing.assert_allclose(body_1, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(body_2, np.array([0.8, -0.3, 0.1]))

    def test_fixed_wing_theory_geometry_uses_regular_rectangular_numbers(self) -> None:
        """The formation-study fixed wing should use the clean requested geometry."""
        self.assertAlmostEqual(sweep_case.FULL_SPAN_M, 1.0)
        self.assertAlmostEqual(sweep_case.SEMI_SPAN_M, 0.5)
        self.assertAlmostEqual(sweep_case.ROOT_CHORD_M, 0.1)
        self.assertAlmostEqual(sweep_case.TIP_CHORD_M, 0.1)
        self.assertAlmostEqual(sweep_case.ASPECT_RATIO, 10.0)
        self.assertEqual(sweep_case.NUM_CHORDWISE_PANELS, 2)
        self.assertEqual(sweep_case.NUM_SPANWISE_PANELS_PER_HALF, 3)

        airplane = sweep_case.build_rectangular_fixed_wing_airplane()
        self.assertAlmostEqual(float(airplane.s_ref), 0.1)
        self.assertAlmostEqual(float(airplane.c_ref), 0.1)
        self.assertAlmostEqual(float(airplane.b_ref), 1.0)

    def test_initial_collision_guard_rejects_planform_overlap(self) -> None:
        """Initial overlapping rectangular wings should be skipped before launch."""
        self.assertFalse(
            sweep_case.initial_condition_is_collision_free(
                x_over_span=0.0,
                y_over_span=0.5,
                z_over_span=0.0,
            )
        )
        self.assertTrue(
            sweep_case.initial_condition_is_collision_free(
                x_over_span=0.0,
                y_over_span=0.5,
                z_over_span=0.1,
            )
        )
        self.assertTrue(
            sweep_case.initial_condition_is_collision_free(
                x_over_span=1.0,
                y_over_span=0.25,
                z_over_span=0.0,
            )
        )

    def test_clamp_history_uses_negative_removed_loads(self) -> None:
        """Clamp forces and torques should cancel constrained loads."""
        diagnostics = sweep_case.StreamwiseClampDiagnostics(
            mujoco_model=_FakeMuJoCoModel(),
            target_positions_E_m=np.zeros((2, 3)),
            target_angles_deg=np.array([0.0, 5.0, 0.0]),
        )
        diagnostics.raw_forces_E.append(
            np.array(
                [
                    [1.0, 2.0, -3.0],
                    [-4.0, -5.0, 6.0],
                ],
                dtype=float,
            )
        )
        diagnostics.raw_moments_E_Cg.append(
            np.array(
                [
                    [0.1, -0.2, 0.3],
                    [-0.4, 0.5, -0.6],
                ],
                dtype=float,
            )
        )
        diagnostics.projected_forces_E.append(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-4.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        )
        diagnostics.projected_moments_E_Cg.append(np.zeros((2, 3), dtype=float))
        diagnostics.preclamp_yz_velocities_E.append(
            np.array(
                [
                    [0.5, -0.25],
                    [0.1, 0.2],
                ],
                dtype=float,
            )
        )
        diagnostics.preclamp_angular_rates_rad_s.append(
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=float,
            )
        )

        arrays = diagnostics.history_arrays()

        np.testing.assert_allclose(
            arrays["clamp_forces_yz_E_N"][0],
            np.array([[-2.0, 3.0], [5.0, -6.0]]),
        )
        np.testing.assert_allclose(
            arrays["clamp_moments_E_Cg_Nm"][0],
            np.array([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]),
        )
        self.assertGreater(arrays["clamp_power_proxy_W"][0, 0], 0.0)
        self.assertGreater(arrays["clamp_power_proxy_W"][0, 1], 0.0)

    def test_add_x_perturbations(self) -> None:
        """The helper should add +/- perturbations around every requested X/B."""
        self.assertEqual(
            sweep_case.add_x_perturbations((0.0, 1.0), 0.1),
            (-0.1, 0.0, 0.1, 0.9, 1.0, 1.1),
        )


if __name__ == "__main__":
    unittest.main()
