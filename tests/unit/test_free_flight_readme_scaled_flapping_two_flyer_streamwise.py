"""Tests for the README-scaled two-flyer streamwise-free flapping example."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples import free_flight_readme_scaled_flapping_two_flyer_streamwise as case
from pterasoftware import _transformations


class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(14, dtype=float)
        self.qvel = np.zeros(12, dtype=float)


class _FakeMuJoCoModel:
    """Minimal multibody model for clamp tests."""

    num_bodies = 2
    body_qposadrs = np.array([0, 7], dtype=int)
    body_qveladrs = np.array([0, 6], dtype=int)
    model = object()

    def __init__(self) -> None:
        self.data = _FakeData()
        self.applied_forces: list[np.ndarray] = []
        self.applied_moments: list[np.ndarray] = []

    def apply_loads(self, forces_E: np.ndarray, moments_E_Cg: np.ndarray) -> None:
        self.applied_forces.append(np.asarray(forces_E, dtype=float).copy())
        self.applied_moments.append(np.asarray(moments_E_Cg, dtype=float).copy())

    def step(self) -> None:
        self.data.qpos[0] += self.data.qvel[0] * 0.1
        self.data.qpos[7] += self.data.qvel[6] * 0.1


class TestTwoFlyerStreamwiseFreeCase(unittest.TestCase):
    """Validate coordinates and independent streamwise clamp behavior."""

    def test_body_positions_use_front_rear_convention(self) -> None:
        front, rear = case.body_positions_from_offsets(
            x_over_span=2.0,
            y_over_span=0.25,
            z_over_span=0.5,
            span_m=1.0,
        )

        np.testing.assert_allclose(front, [2.0, -0.25, -0.5])
        np.testing.assert_allclose(rear, [0.0, 0.0, 0.0])

    def test_requested_grid_is_collision_free(self) -> None:
        for x_over_span in case.GRID_X_OVER_SPAN:
            for y_over_span in case.GRID_Y_OVER_SPAN:
                self.assertTrue(
                    case.initial_condition_is_collision_free(
                        x_over_span=x_over_span,
                        y_over_span=y_over_span,
                        z_over_span=0.0,
                    )
                )

    def test_load_projection_passes_only_each_body_streamwise_force(self) -> None:
        fake = _FakeMuJoCoModel()
        diagnostics = case.MultiBodyStreamwiseFreeClampDiagnostics(
            mujoco_model=fake,
            target_positions_E_m=np.array([[0.5, -0.25, 0.0], [0.0, 0.0, 0.0]]),
            target_angles_deg=np.zeros(3),
            forward_fn=lambda model, data: None,
        )
        diagnostics.install()

        raw_forces = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0]])
        raw_moments = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
        fake.apply_loads(raw_forces, raw_moments)

        np.testing.assert_allclose(
            fake.applied_forces[-1], [[1.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(fake.applied_moments[-1], np.zeros((2, 3)))
        history = diagnostics.history_arrays()
        np.testing.assert_allclose(
            history["clamp_forces_E_N"][0],
            [[0.0, -2.0, -3.0], [0.0, 5.0, -6.0]],
        )
        np.testing.assert_allclose(
            history["clamp_moments_E_Cg_Nm"][0],
            [[-0.1, -0.2, -0.3], [0.4, -0.5, 0.6]],
        )

    def test_step_preserves_independent_x_and_ux_only(self) -> None:
        fake = _FakeMuJoCoModel()
        diagnostics = case.MultiBodyStreamwiseFreeClampDiagnostics(
            mujoco_model=fake,
            target_positions_E_m=np.array([[0.5, -0.25, 0.0], [0.0, 0.0, 0.0]]),
            target_angles_deg=np.zeros(3),
            initial_streamwise_speed_mps=0.8,
            forward_fn=lambda model, data: None,
        )
        diagnostics.install()

        fake.data.qpos[0:3] = [1.2, 9.0, 8.0]
        fake.data.qpos[7:10] = [-0.4, 7.0, 6.0]
        fake.data.qvel[0:6] = [0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        fake.data.qvel[6:12] = [0.7, -1.0, -2.0, -3.0, -4.0, -5.0]

        fake.step()

        np.testing.assert_allclose(fake.data.qpos[[0, 7]], [1.29, -0.33])
        np.testing.assert_allclose(fake.data.qvel[[0, 6]], [0.9, 0.7])
        np.testing.assert_allclose(fake.data.qpos[1:3], [-0.25, 0.0])
        np.testing.assert_allclose(fake.data.qpos[8:10], [0.0, 0.0])
        np.testing.assert_allclose(fake.data.qvel[1:6], np.zeros(5))
        np.testing.assert_allclose(fake.data.qvel[7:12], np.zeros(5))
        np.testing.assert_allclose(
            diagnostics.preclamp_velocities_E[0][0], [0.9, 1.0, 2.0]
        )
        np.testing.assert_allclose(
            diagnostics.preclamp_velocities_E[0][1], [0.7, -1.0, -2.0]
        )

    def test_restart_state_loads_checkpoint_arrays(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "restart_step_000048.npz"
            np.savez_compressed(
                path,
                restart_compatible=np.array([True]),
                global_step=np.array([48]),
                positions_E_E=np.ones((2, 3)),
                R_pas_E_to_BPs=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
                velocities_E__E=np.full((2, 3), 0.5),
                omegas_BPs__E=np.zeros((2, 3)),
                wake_strengths=np.arange(4.0),
                wake_ages=np.arange(4.0) + 1.0,
                wake_rc0s=np.full(4, 0.003),
                wake_br=np.ones((4, 3)),
                wake_fr=np.ones((4, 3)) * 2.0,
                wake_fl=np.ones((4, 3)) * 3.0,
                wake_bl=np.ones((4, 3)) * 4.0,
                previous_body_positions_E_E=np.zeros((2, 3)),
                previous_body_R_pas_GP_to_Es=np.repeat(
                    np.eye(3)[None, :, :], 2, axis=0
                ),
                previous_bound_vortex_strengths=np.ones(6),
                previous_stack_cpp=np.ones((6, 3)),
                previous_stack_cblvpr=np.ones((6, 3)) * 2.0,
                previous_stack_cblvpf=np.ones((6, 3)) * 3.0,
                previous_stack_cblvpl=np.ones((6, 3)) * 4.0,
                previous_stack_cblvpb=np.ones((6, 3)) * 5.0,
                previous_panel_blpp=np.ones((6, 3)) * 6.0,
                previous_panel_brpp=np.ones((6, 3)) * 7.0,
            )

            restart_state = case.load_restart_state(path)

        self.assertEqual(restart_state.global_step, 48)
        np.testing.assert_allclose(restart_state.positions_E_E, np.ones((2, 3)))
        np.testing.assert_allclose(restart_state.wake_strengths, np.arange(4.0))
        np.testing.assert_allclose(
            restart_state.previous_panel_brpp,
            np.ones((6, 3)) * 7.0,
        )

    def test_flapping_phase_offset_stays_in_wing_movement_range(self) -> None:
        self.assertEqual(case.flapping_phase_offset_deg(0, 48), 0.0)
        self.assertEqual(case.flapping_phase_offset_deg(24, 48), 180.0)
        self.assertEqual(case.flapping_phase_offset_deg(47, 48), -7.5)
        self.assertEqual(case.flapping_phase_offset_deg(1103, 48), -7.5)

    def test_distance_convergence_uses_final_cycle_slope(self) -> None:
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dense.csv"
            with csv_path.open("w") as f:
                f.write("x_over_span\n")
                for value in [1.0] * 10:
                    f.write(f"{value}\n")
                for value in [1.001] * 10:
                    f.write(f"{value}\n")
                for value in [1.002] * 10:
                    f.write(f"{value}\n")

            diagnostics = case.distance_convergence_from_csv(
                csv_path,
                steps_per_flap=5,
                slope_tol_xb_per_period=0.005,
                window_periods=5,
            )

        self.assertTrue(diagnostics["converged"])
        self.assertLess(abs(diagnostics["final_window_slope_xb_per_period"]), 0.005)

    def test_far_pair_first_free_wake_matches_single_body(self) -> None:
        """A far-separated pair should reduce to two independent single flyers."""
        steps_per_flap = 48
        initial_speed_mps = 0.644276526
        single_problem, single_solver, single_diagnostics = (
            case.single_case.build_problem(
                prescribed_num_steps=1,
                free_num_steps=2,
                steps_per_flap=steps_per_flap,
                initial_speed_mps=initial_speed_mps,
            )
        )
        single_solver.run(
            prescribed_wake=False,
            show_progress=False,
            history_stride=1,
        )

        _, pair_solver, pair_diagnostics = case.build_problem(
            x_over_span=1.0,
            y_over_span=100.0,
            z_over_span=0.0,
            prescribed_num_steps=1,
            free_num_steps=2,
            steps_per_flap=steps_per_flap,
            initial_speed_mps=initial_speed_mps,
        )
        pair_solver.run(
            prescribed_wake=False,
            show_progress=False,
            history_stride=1,
        )

        transform_gp_to_e = (
            single_solver.current_coupled_operating_point.T_pas_GP1_CgP1_to_E_CgP1
        )
        single_position_E = single_solver.stackPosition_E_E[single_solver._current_step]
        single_wake_rows_E = []
        pair_rear_wake_rows_E = []
        for single_wing, rear_wing in zip(
            single_solver.current_airplane.wings,
            pair_solver.current_airplanes[1].wings,
            strict=True,
        ):
            single_grid = single_wing.gridWrvp_GP1_CgP1
            assert single_grid is not None
            rear_grid = rear_wing.gridWrvp_GP1_CgP1
            assert rear_grid is not None
            single_wake_rows_E.append(
                single_position_E
                + _transformations.apply_T_to_vectors(
                    transform_gp_to_e,
                    single_grid.reshape((-1, 3)),
                    has_point=True,
                ).reshape(single_grid.shape)
            )
            pair_rear_wake_rows_E.append(rear_grid)

        np.testing.assert_allclose(
            np.concatenate(single_wake_rows_E, axis=1),
            np.concatenate(pair_rear_wake_rows_E, axis=1),
            atol=1.0e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            single_diagnostics.raw_forces_E[-1],
            pair_diagnostics.raw_forces_E[-1][1],
            atol=1.0e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            pair_diagnostics.raw_forces_E[-1][0],
            pair_diagnostics.raw_forces_E[-1][1],
            atol=1.0e-8,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
