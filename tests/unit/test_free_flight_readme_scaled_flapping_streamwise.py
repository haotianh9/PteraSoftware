"""Tests for the README-scaled streamwise-free flapping example."""

from __future__ import annotations

import unittest

import numpy as np

from examples import free_flight_readme_scaled_flapping_streamwise as case


class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.qvel = np.array([case.INITIAL_SPEED_MPS, 0.0, 0.0, 0.0, 0.0, 0.0])


class _FakeMuJoCoModel:
    def __init__(self) -> None:
        self.model = object()
        self.data = _FakeData()
        self.applied_forces_E: list[np.ndarray] = []
        self.applied_moments_E: list[np.ndarray] = []

    def apply_loads(self, forces_E: np.ndarray, moments_E_Cg: np.ndarray) -> None:
        self.applied_forces_E.append(np.asarray(forces_E, dtype=float).copy())
        self.applied_moments_E.append(np.asarray(moments_E_Cg, dtype=float).copy())

    def step(self) -> None:
        self.data.qpos[:] = np.array([9.0, 8.0, 7.0, 0.2, 0.3, 0.4, 0.5])
        self.data.qvel[:] = np.array([1.25, 2.0, 3.0, 4.0, 5.0, 6.0])


class TestReadmeScaledFlappingGeometry(unittest.TestCase):
    """Validate the requested scaled geometry and kinematics constants."""

    def test_scaled_geometry_constants(self) -> None:
        self.assertAlmostEqual(case.FULL_SPAN_M, 1.0)
        self.assertAlmostEqual(case.MEAN_CHORD_M, 0.1)
        self.assertAlmostEqual(case.ROOT_CHORD_M, 0.11666666666666667)
        self.assertAlmostEqual(case.TIP_CHORD_M, 0.08333333333333333)
        self.assertAlmostEqual(case.FLAPPING_FREQUENCY_HZ, 1.0)
        self.assertAlmostEqual(case.FLAPPING_PERIOD_S, 1.0)

    def test_build_airplane_uses_naca0012_and_readme_angles(self) -> None:
        airplane = case.build_airplane()

        self.assertEqual(len(airplane.wings), 2)
        self.assertEqual(airplane.name, "README-Scaled Flapping Streamwise Wing")
        for wing in airplane.wings:
            np.testing.assert_allclose(
                wing.angles_Gs_to_Wn_ixyz,
                np.array([0.0, case.WING_INCIDENCE_DEG, 0.0]),
            )
            self.assertAlmostEqual(wing.wing_cross_sections[0].chord, case.ROOT_CHORD_M)
            self.assertAlmostEqual(wing.wing_cross_sections[1].chord, case.TIP_CHORD_M)
            np.testing.assert_allclose(
                wing.wing_cross_sections[1].Lp_Wcsp_Lpp,
                np.array(
                    [
                        case.TIP_X_FROM_ROOT_M,
                        case.TIP_Y_FROM_ROOT_M,
                        case.TIP_Z_FROM_ROOT_M,
                    ]
                ),
            )
            np.testing.assert_allclose(
                wing.wing_cross_sections[1].angles_Wcsp_to_Wcs_ixyz,
                np.array([0.0, case.TIP_TWIST_DEG, 0.0]),
            )
            self.assertEqual(
                wing.wing_cross_sections[0].airfoil.name, case.AIRFOIL_NAME
            )
            self.assertEqual(
                wing.wing_cross_sections[1].airfoil.name, case.AIRFOIL_NAME
            )

    def test_build_airplane_movement_uses_single_axis_one_hz_flapping(self) -> None:
        airplane_movement = case.build_airplane_movement(case.build_airplane())

        self.assertEqual(len(airplane_movement.wing_movements), 2)
        for wing_movement in airplane_movement.wing_movements:
            np.testing.assert_allclose(
                wing_movement.ampAngles_Gs_to_Wn_ixyz,
                np.array([case.FLAPPING_AMPLITUDE_DEG, 0.0, 0.0]),
            )
            np.testing.assert_allclose(
                wing_movement.periodAngles_Gs_to_Wn_ixyz,
                np.array([case.FLAPPING_PERIOD_S, 0.0, 0.0]),
            )
            np.testing.assert_allclose(
                wing_movement.rotationPointOffset_Gs_Ler,
                np.zeros(3),
            )


class TestStreamwiseFreeClampDiagnostics(unittest.TestCase):
    """Validate the x-only free-motion projection and reactions."""

    @staticmethod
    def _make_diagnostics() -> (
        tuple[_FakeMuJoCoModel, case.StreamwiseFreeClampDiagnostics]
    ):
        fake_model = _FakeMuJoCoModel()
        diagnostics = case.StreamwiseFreeClampDiagnostics(
            mujoco_model=fake_model,
            target_position_E_m=np.array([0.0, -0.25, 0.5]),
            target_angles_deg=np.zeros(3),
            initial_streamwise_speed_mps=case.INITIAL_SPEED_MPS,
            forward_fn=lambda model, data: None,
        )
        diagnostics.install()
        return fake_model, diagnostics

    def test_load_projection_passes_only_streamwise_force(self) -> None:
        fake_model, diagnostics = self._make_diagnostics()

        fake_model.apply_loads(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        )
        fake_model.step()

        np.testing.assert_allclose(fake_model.applied_forces_E[-1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(fake_model.applied_moments_E[-1], np.zeros(3))

        history = diagnostics.history_arrays()
        np.testing.assert_allclose(history["raw_forces_E_N"][0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(history["raw_moments_E_Cg_Nm"][0], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(history["projected_forces_E_N"][0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(history["projected_moments_E_Cg_Nm"][0], np.zeros(3))
        np.testing.assert_allclose(history["clamp_forces_E_N"][0], [0.0, -2.0, -3.0])
        np.testing.assert_allclose(
            history["clamp_moments_E_Cg_Nm"][0], [-4.0, -5.0, -6.0]
        )

    def test_step_clamps_everything_except_x_and_ux(self) -> None:
        fake_model, diagnostics = self._make_diagnostics()

        fake_model.step()

        np.testing.assert_allclose(fake_model.data.qpos[0], 9.0)
        np.testing.assert_allclose(fake_model.data.qpos[1:3], [-0.25, 0.5])
        np.testing.assert_allclose(fake_model.data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(fake_model.data.qvel[0], 1.25)
        np.testing.assert_allclose(fake_model.data.qvel[1:6], np.zeros(5))

        np.testing.assert_allclose(
            diagnostics.preclamp_velocities_E[0],
            np.array([1.25, 2.0, 3.0]),
        )
        np.testing.assert_allclose(
            diagnostics.preclamp_angular_rates_rad_s[0],
            np.array([4.0, 5.0, 6.0]),
        )


class TestSpeedConvergence(unittest.TestCase):
    """Validate conservative streamwise-speed convergence reporting."""

    def test_convergence_requires_at_least_five_free_cycles(self) -> None:
        velocities_E = np.zeros((4, 3), dtype=float)
        velocities_E[:, 0] = 1.0

        convergence = case.compute_speed_convergence(
            times_s=np.arange(4, dtype=float),
            velocities_E__E=velocities_E,
            prescribed_num_steps=0,
            steps_per_flap=2,
        )

        self.assertFalse(convergence["converged"])
        self.assertIsNone(convergence["converged_speed_mps"])
        self.assertEqual(len(convergence["cycle_mean_ux_mps"]), 2)


if __name__ == "__main__":
    unittest.main()
