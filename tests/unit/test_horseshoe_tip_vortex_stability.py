"""Tests for the horseshoe/tip-vortex streamwise stability model."""

from __future__ import annotations

import unittest

import numpy as np

from examples import horseshoe_tip_vortex_stability as model


class TestHorseshoeWakeModel(unittest.TestCase):
    """Validate the analytical model used by the fixed-wing figures."""

    def setUp(self) -> None:
        """Use enough quadrature for sign-sensitive finite-wing checks."""
        self.params = model.HorseshoeWakeParams(
            quadrature_order=96,
            core_radius_m=1.0e-4,
        )
        self.singular_free_params = model.HorseshoeWakeParams(
            quadrature_order=128,
            core_radius_m=0.0,
        )

    def test_horseshoe_circulation_matches_lift_relation(self) -> None:
        """The constant circulation must satisfy L = rho U Gamma_t B."""
        gamma = model.gamma0_m2_s(self.params)
        reconstructed_lift = (
            self.params.rho_kg_m3 * self.params.ubar_mps * gamma * self.params.span_m
        )
        self.assertAlmostEqual(reconstructed_lift, model.lift_n(self.params))

    def test_outer_tip_pair_upwash_has_expected_sign(self) -> None:
        """The tip-vortex pair should produce upwash outside the source wing."""
        w_right = model.point_trailing_vertical_velocity_mps(
            1.0 * self.params.span_m,
            1.0 * self.params.span_m,
            0.0,
            self.params,
        )
        w_left = model.point_trailing_vertical_velocity_mps(
            1.0 * self.params.span_m,
            -1.0 * self.params.span_m,
            0.0,
            self.params,
        )
        self.assertGreater(float(w_right), 0.0)
        self.assertGreater(float(w_left), 0.0)

    def test_point_tip_pair_boundary_changes_streamwise_gradient_sign(self) -> None:
        """The closed-form R_c boundary separates stable and unstable signs."""
        y = 1.0 * self.params.span_m
        rc = float(model.critical_radius_tip_pair_m(y, self.params))
        stable_gradient = model.point_trailing_dwdx_mps_per_m(
            0.5 * rc,
            y,
            0.0,
            self.params,
        )
        unstable_gradient = model.point_trailing_dwdx_mps_per_m(
            1.5 * rc,
            y,
            0.0,
            self.params,
        )
        self.assertGreater(float(stable_gradient), 0.0)
        self.assertLess(float(unstable_gradient), 0.0)

    def test_tip_pair_neutral_boundary_obeys_x2_plus_z2(self) -> None:
        """The plotted boundary should implement X^2 + Z^2 = R_c^2."""
        y_over_span = np.array([0.75, 1.0, 1.5])
        x0 = model.tip_pair_neutral_x_over_span(y_over_span, 0.0, self.params)
        xz = model.tip_pair_neutral_x_over_span(y_over_span, 0.25, self.params)
        self.assertTrue(np.all(np.isfinite(x0)))
        self.assertTrue(np.all(xz < x0))

    def test_full_horseshoe_gradient_matches_finite_difference(self) -> None:
        """The analytic full-horseshoe gradient should match Wbar finite diff."""
        x = 0.8 * self.params.span_m
        y = 1.25 * self.params.span_m
        z = 0.2 * self.params.span_m
        analytic = model.lift_weighted_dwbardx_mps_per_m(
            x, y, z, self.singular_free_params
        )
        finite_difference = model.lift_weighted_dwbardx_biot_savart_fd_mps_per_m(
            x, y, z, self.singular_free_params
        )
        self.assertAlmostEqual(float(finite_difference), float(analytic), places=5)

    def test_full_horseshoe_gradient_is_not_tip_pair_surrogate(self) -> None:
        """The active model must retain the bound-vortex correction."""
        x = 0.8 * self.params.span_m
        y = 1.25 * self.params.span_m
        z = 0.2 * self.params.span_m
        full_horseshoe = model.lift_weighted_dwbardx_mps_per_m(x, y, z, self.params)
        tip_pair_only = model.lift_weighted_tip_pair_dwbardx_mps_per_m(
            x,
            y,
            z,
            self.params,
        )
        self.assertNotAlmostEqual(float(full_horseshoe), float(tip_pair_only), places=6)

    def test_coordinate_convention_for_pair_terms(self) -> None:
        """Wminus and its X derivative should follow W(-X,-Y,-Z)."""
        x = 1.4 * self.params.span_m
        y = 0.75 * self.params.span_m
        z = 0.2 * self.params.span_m
        terms = model.pair_wake_terms(x, y, z, self.params)
        expected_wminus = model.lift_weighted_wbar_mps(-x, -y, -z, self.params)
        expected_dwminus = -model.lift_weighted_dwbardx_mps_per_m(
            -x, -y, -z, self.params
        )
        self.assertAlmostEqual(float(terms["w_minus_mps"]), float(expected_wminus))
        self.assertAlmostEqual(
            float(terms["dw_minus_dX_mps_per_m"]), float(expected_dwminus)
        )


if __name__ == "__main__":
    unittest.main()
