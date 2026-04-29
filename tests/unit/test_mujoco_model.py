"""Unit tests for the MuJoCo wrapper classes."""

import unittest

import numpy as np

import pterasoftware as ps
from pterasoftware import _mujoco_model
from tests.unit.fixtures import geometry_fixtures, operating_point_fixtures


class TestMultiBodyMuJoCoModel(unittest.TestCase):
    """Tests for the true multibody MuJoCo wrapper."""

    @staticmethod
    def _make_model() -> _mujoco_model.MultiBodyMuJoCoModel:
        """Create a two-body MuJoCo model with distinct body states."""
        airplane_1 = geometry_fixtures.make_basic_airplane_fixture()
        airplane_2 = geometry_fixtures.make_basic_airplane_fixture()

        coupled_operating_point_1 = (
            operating_point_fixtures.make_basic_coupled_operating_point_fixture()
        )
        coupled_operating_point_2 = (
            operating_point_fixtures.make_with_attitude_angles_coupled_operating_point_fixture()
        )

        return _mujoco_model.MultiBodyMuJoCoModel(
            airplanes=[airplane_1, airplane_2],
            coupled_operating_points=[
                coupled_operating_point_1,
                coupled_operating_point_2,
            ],
            I_BP1_CgP1s=[
                np.diag([10.0, 20.0, 30.0]),
                np.diag([11.0, 21.0, 31.0]),
            ],
            initial_positions_E_E=[
                np.array([0.0, 0.0, 0.0]),
                np.array([4.0, -2.0, 1.5]),
            ],
            delta_time=0.01,
        )

    def test_initialization_builds_two_unique_bodies(self) -> None:
        """The wrapper should build two freejoint bodies with unique names."""
        model = self._make_model()

        self.assertEqual(model.num_bodies, 2)
        self.assertEqual(model.body_names[0], "Basic Test Airplane")
        self.assertEqual(model.body_names[1], "Basic Test Airplane_2")
        self.assertEqual(model.joint_names[0], "Basic Test Airplane_freejoint")
        self.assertEqual(model.joint_names[1], "Basic Test Airplane_2_freejoint")

        np.testing.assert_array_equal(model.body_qposadrs, np.array([0, 7]))
        np.testing.assert_array_equal(model.body_qveladrs, np.array([0, 6]))

    def test_get_states_returns_expected_shapes(self) -> None:
        """State extraction should return one row per body."""
        model = self._make_model()
        states = model.get_states()

        self.assertEqual(states["positions_E_E"].shape, (2, 3))
        self.assertEqual(states["R_pas_E_to_BPs"].shape, (2, 3, 3))
        self.assertEqual(states["velocities_E__E"].shape, (2, 3))
        self.assertEqual(states["omegas_BPs__E"].shape, (2, 3))
        self.assertIsInstance(states["time"], float)

    def test_get_state_matches_get_states_slice(self) -> None:
        """Single-body access should match the corresponding row slice."""
        model = self._make_model()
        states = model.get_states()

        for body_index in range(model.num_bodies):
            with self.subTest(body_index=body_index):
                state = model.get_state(body_index)
                np.testing.assert_allclose(
                    state["position_E_E"], states["positions_E_E"][body_index]
                )
                np.testing.assert_allclose(
                    state["R_pas_E_to_BP"], states["R_pas_E_to_BPs"][body_index]
                )
                np.testing.assert_allclose(
                    state["velocity_E__E"], states["velocities_E__E"][body_index]
                )
                np.testing.assert_allclose(
                    state["omegas_BP__E"], states["omegas_BPs__E"][body_index]
                )
                self.assertEqual(state["time"], states["time"])

    def test_apply_loads_writes_one_row_per_body(self) -> None:
        """Per-body loads should populate the matching MuJoCo rows only."""
        model = self._make_model()
        forces_E = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        moments_E_Cg = np.array([[4.0, 5.0, 6.0], [-4.0, -5.0, -6.0]])

        model.apply_loads(forces_E=forces_E, moments_E_Cg=moments_E_Cg)

        for body_index, body_id in enumerate(model.body_ids):
            with self.subTest(body_index=body_index):
                np.testing.assert_allclose(
                    model.data.xfrc_applied[body_id],
                    np.hstack([forces_E[body_index], moments_E_Cg[body_index]]),
                )

    def test_step_advances_positions_with_initial_velocity(self) -> None:
        """With zero applied loads, free bodies should advect at constant velocity."""
        model = self._make_model()
        initial_states = model.get_states()
        delta_time = model.model.opt.timestep

        model.step()
        stepped_states = model.get_states()

        expected_positions = (
            initial_states["positions_E_E"]
            + initial_states["velocities_E__E"] * delta_time
        )
        np.testing.assert_allclose(
            stepped_states["positions_E_E"], expected_positions, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            stepped_states["velocities_E__E"],
            initial_states["velocities_E__E"],
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertAlmostEqual(stepped_states["time"], delta_time)

    def test_reset_restores_initial_state_and_clears_loads(self) -> None:
        """Reset should restore the initial qpos/qvel, time, and applied loads."""
        model = self._make_model()
        initial_states = model.get_states()

        model.apply_loads(
            forces_E=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            moments_E_Cg=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        )
        model.step()
        model.reset()

        reset_states = model.get_states()
        np.testing.assert_allclose(
            reset_states["positions_E_E"], initial_states["positions_E_E"]
        )
        np.testing.assert_allclose(
            reset_states["R_pas_E_to_BPs"], initial_states["R_pas_E_to_BPs"]
        )
        np.testing.assert_allclose(
            reset_states["velocities_E__E"], initial_states["velocities_E__E"]
        )
        np.testing.assert_allclose(
            reset_states["omegas_BPs__E"], initial_states["omegas_BPs__E"]
        )
        self.assertEqual(reset_states["time"], 0.0)
        np.testing.assert_allclose(model.data.xfrc_applied, 0.0)

    def test_requires_at_least_two_bodies(self) -> None:
        """The multibody wrapper should reject a single-body configuration."""
        airplane = geometry_fixtures.make_basic_airplane_fixture()
        coupled_operating_point = (
            operating_point_fixtures.make_basic_coupled_operating_point_fixture()
        )

        with self.assertRaises(ValueError):
            _mujoco_model.MultiBodyMuJoCoModel(
                airplanes=[airplane],
                coupled_operating_points=[coupled_operating_point],
                I_BP1_CgP1s=[np.diag([10.0, 20.0, 30.0])],
                initial_positions_E_E=[np.zeros(3)],
                delta_time=0.01,
            )

    def test_apply_loads_validates_body_row_count(self) -> None:
        """Load arrays should have one row per body."""
        model = self._make_model()

        with self.assertRaises(ValueError):
            model.apply_loads(
                forces_E=np.array([[1.0, 2.0, 3.0]]),
                moments_E_Cg=np.array([[4.0, 5.0, 6.0]]),
            )

    def test_get_state_validates_body_index(self) -> None:
        """Single-body state extraction should reject an out-of-range body index."""
        model = self._make_model()

        with self.assertRaises(ValueError):
            model.get_state(2)


class TestBodyStateFromAirplaneAndOperatingPoint(unittest.TestCase):
    """Tests for the multibody body-state packing helper."""

    def test_zero_gravity_rejected(self) -> None:
        """MuJoCo mass inference should reject zero-gravity operating points."""
        airplane = geometry_fixtures.make_basic_airplane_fixture()
        coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
            rho=1.225,
            vCg__E=10.0,
            alpha=5.0,
            beta=0.0,
            externalFX_W=0.0,
            nu=15.06e-6,
            omegas_BP1__E=(0.0, 0.0, 0.0),
            angles_E_to_BP1_izyx=(0.0, 0.0, 0.0),
            g_E=(0.0, 0.0, 0.0),
        )

        with self.assertRaises(ValueError):
            _mujoco_model._body_state_from_airplane_and_operating_point(
                airplane=airplane,
                coupled_operating_point=coupled_operating_point,
                inertia_BP1_CgP1=np.diag([1.0, 2.0, 3.0]),
                initial_position_E_E=np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
