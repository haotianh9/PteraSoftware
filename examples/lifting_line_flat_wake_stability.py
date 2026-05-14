"""Horseshoe/tip-vortex model for streamwise formation stability.

This module implements the analytical closure used for the fixed-wing manuscript
figures: a constant-circulation horseshoe vortex evaluated with Biot-Savart.
The full model contains the spanwise bound segment plus two semi-infinite
trailing tip vortices.  The tip-vortex-pair limit is also exposed for the
closed-form streamwise-neutral boundary.

The historical module name is kept so existing plotting scripts continue to
import one shared analytical source of truth.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FlatWakeParams:
    """Parameters for the constant-circulation horseshoe wake model."""

    span_m: float = 1.0
    chord_m: float = 0.1
    rho_kg_m3: float = 1.225
    ubar_mps: float = 1.0
    aoa_deg: float = 5.0
    gravity_mps2: float = 9.81
    speed_damping_n_per_mps: float = 1.0
    oswald_efficiency: float = 1.0
    quadrature_order: int = 64
    core_radius_m: float = 1.0e-4
    gradient_step_over_span: float = 1.0e-3
    wake_length_over_span: float = 120.0
    wake_num_segments: int = 128


HorseshoeWakeParams = FlatWakeParams

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "streamwise_stability_energy"
    / "curated_fixed_wing_dataset"
    / "figures"
)
DEFAULT_OUTPUT_FIGURE = DEFAULT_OUTPUT_DIR / "streamwise_stability_9panel.png"


def aspect_ratio(params: FlatWakeParams) -> float:
    """Return the rectangular-wing aspect ratio."""
    return params.span_m / params.chord_m


def finite_wing_lift_slope_per_rad(params: FlatWakeParams) -> float:
    """Return a simple finite-wing lift slope used to set circulation scale."""
    ar = aspect_ratio(params)
    return 2.0 * np.pi * ar / (ar + 2.0)


def lift_coefficient(params: FlatWakeParams) -> float:
    """Return the reference lift coefficient at the selected angle of attack."""
    return finite_wing_lift_slope_per_rad(params) * np.deg2rad(params.aoa_deg)


def lift_n(params: FlatWakeParams) -> float:
    """Return reference lift from the rectangular planform and AOA."""
    area = params.span_m * params.chord_m
    dynamic_pressure = 0.5 * params.rho_kg_m3 * params.ubar_mps**2
    return dynamic_pressure * area * lift_coefficient(params)


def mass_kg(params: FlatWakeParams) -> float:
    """Return the mass that would trim the reference lift in level flight."""
    return lift_n(params) / params.gravity_mps2


def gamma0_m2_s(params: FlatWakeParams) -> float:
    """Return the horseshoe circulation Gamma_t from L = rho U Gamma_t B."""
    return lift_n(params) / (params.rho_kg_m3 * params.ubar_mps * params.span_m)


def baseline_thrust_n(params: FlatWakeParams) -> float:
    """Return the analytical baseline drag scale used for stability damping."""
    return params.speed_damping_n_per_mps * params.ubar_mps


def circulation_m2_s(eta_m: np.ndarray | float, params: FlatWakeParams) -> np.ndarray:
    """Return the constant bound circulation along the rectangular wing."""
    eta = np.asarray(eta_m, dtype=float)
    semispan = 0.5 * params.span_m
    return np.where(np.abs(eta) <= semispan, gamma0_m2_s(params), 0.0)


def minus_dgamma_deta_mps(
    eta_m: np.ndarray | float, params: FlatWakeParams
) -> np.ndarray:
    """Return the distributed trailing-sheet strength.

    The horseshoe closure collapses the trailing sheet to two tip vortices, so
    this compatibility helper is identically zero away from the singular tips.
    """
    return np.zeros_like(np.asarray(eta_m, dtype=float))


@functools.lru_cache(maxsize=None)
def _quadrature(span_m: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on [-span/2, span/2]."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    semispan = 0.5 * span_m
    return semispan * nodes, semispan * weights


def _broadcast_xyz(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast coordinate arrays to a shared shape."""
    return np.broadcast_arrays(
        np.asarray(x_m, dtype=float),
        np.asarray(y_m, dtype=float),
        np.asarray(z_m, dtype=float),
    )


def _tip_offsets(
    y_m: np.ndarray,
    z_m: np.ndarray,
    params: FlatWakeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return right/left tip offsets and regularized squared radii."""
    semispan = 0.5 * params.span_m
    core2 = params.core_radius_m**2
    delta_r = y_m - semispan
    delta_l = y_m + semispan
    r_r2 = delta_r**2 + z_m**2 + core2
    r_l2 = delta_l**2 + z_m**2 + core2
    return delta_r, delta_l, r_r2, r_l2


def point_trailing_vertical_velocity_mps(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return W_tr from the two semi-infinite trailing tip vortices."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    gamma = gamma0_m2_s(params)
    delta_r, delta_l, r_r2, r_l2 = _tip_offsets(y, z, params)
    r_r = np.sqrt(x**2 + r_r2)
    r_l = np.sqrt(x**2 + r_l2)
    return (
        gamma
        / (4.0 * np.pi)
        * (delta_r / r_r2 * (1.0 + x / r_r) - delta_l / r_l2 * (1.0 + x / r_l))
    )


def point_bound_vertical_velocity_mps(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return W_b from the finite spanwise bound vortex segment."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    eta, weights = _quadrature(params.span_m, params.quadrature_order)
    gamma = gamma0_m2_s(params)
    core2 = params.core_radius_m**2
    integral = np.zeros_like(x, dtype=float)
    for this_eta, this_weight in zip(eta, weights, strict=True):
        r2 = x**2 + (y - this_eta) ** 2 + z**2 + core2
        integral += this_weight / (r2**1.5)
    return -gamma * x * integral / (4.0 * np.pi)


def point_vertical_velocity_mps(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return full horseshoe vertical velocity W(X,Y,Z)."""
    return point_trailing_vertical_velocity_mps(
        x_m, y_m, z_m, params
    ) + point_bound_vertical_velocity_mps(x_m, y_m, z_m, params)


@functools.lru_cache(maxsize=None)
def _horseshoe_segments(
    span_m: float,
    wake_length_over_span: float,
    bound_segments: int,
    wake_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return midpoint-rule line segments for a full horseshoe vortex."""
    semispan = 0.5 * span_m
    wake_length = wake_length_over_span * span_m

    starts: list[list[float]] = []
    ends: list[list[float]] = []
    signs: list[float] = []

    y_edges = np.linspace(-semispan, semispan, max(1, bound_segments) + 1)
    for y0, y1 in zip(y_edges[:-1], y_edges[1:], strict=True):
        starts.append([0.0, y0, 0.0])
        ends.append([0.0, y1, 0.0])
        signs.append(1.0)

    x_edges = np.linspace(0.0, wake_length, max(1, wake_segments) + 1)
    for x0, x1 in zip(x_edges[:-1], x_edges[1:], strict=True):
        starts.append([x0, semispan, 0.0])
        ends.append([x1, semispan, 0.0])
        signs.append(1.0)
        starts.append([x0, -semispan, 0.0])
        ends.append([x1, -semispan, 0.0])
        signs.append(-1.0)

    return (
        np.asarray(starts, dtype=float),
        np.asarray(ends, dtype=float),
        np.asarray(signs, dtype=float),
    )


def point_velocity_mps(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return full induced velocity from a Biot-Savart horseshoe vortex.

    The vector field for quivers/streamlines is evaluated by midpoint-rule line
    integration of the bound and trailing filaments.  The vertical component is
    then replaced with the exact regularized horseshoe expression above so the
    scalar panels and stability model use the same analytical W.
    """
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    shape = x.shape
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    velocity = np.zeros_like(points)
    starts, ends, signs = _horseshoe_segments(
        params.span_m,
        params.wake_length_over_span,
        params.quadrature_order,
        params.wake_num_segments,
    )
    gamma = gamma0_m2_s(params)
    core2 = params.core_radius_m**2
    for start, end, sign in zip(starts, ends, signs, strict=True):
        dl = end - start
        midpoint = 0.5 * (start + end)
        r_vec = points - midpoint
        r2 = np.einsum("ij,ij->i", r_vec, r_vec) + core2
        velocity += (
            sign * gamma / (4.0 * np.pi) * np.cross(dl, r_vec) / (r2[:, None] ** 1.5)
        )
    velocity = velocity.reshape(shape + (3,))
    velocity[..., 2] = point_vertical_velocity_mps(x, y, z, params)
    return velocity


def point_trailing_dwdx_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return dW_tr/dX for the tip-vortex-pair limit."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    gamma = gamma0_m2_s(params)
    delta_r, delta_l, r_r2, r_l2 = _tip_offsets(y, z, params)
    r_r = np.sqrt(x**2 + r_r2)
    r_l = np.sqrt(x**2 + r_l2)
    return gamma / (4.0 * np.pi) * (delta_r / r_r**3 - delta_l / r_l**3)


def point_bound_dwdx_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return dW_b/dX for the finite bound segment."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    eta, weights = _quadrature(params.span_m, params.quadrature_order)
    gamma = gamma0_m2_s(params)
    core2 = params.core_radius_m**2
    total = np.zeros_like(x, dtype=float)
    for this_eta, this_weight in zip(eta, weights, strict=True):
        delta = y - this_eta
        r2 = x**2 + delta**2 + z**2 + core2
        numerator = 2.0 * x**2 - delta**2 - z**2 - core2
        total += this_weight * numerator / (r2**2.5)
    return gamma * total / (4.0 * np.pi)


def point_dwdx_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return dW/dX for the full horseshoe model."""
    return point_trailing_dwdx_mps_per_m(
        x_m, y_m, z_m, params
    ) + point_bound_dwdx_mps_per_m(x_m, y_m, z_m, params)


def _circulation_integral(params: FlatWakeParams) -> float:
    """Return int Gamma dspan for the constant-circulation wing."""
    return gamma0_m2_s(params) * params.span_m


def lift_weighted_wbar_mps(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return the lift-weighted mean upwash received by a second wing."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    xi, weights = _quadrature(params.span_m, params.quadrature_order)
    gamma = gamma0_m2_s(params)
    gamma_integral = _circulation_integral(params)
    total = np.zeros_like(x, dtype=float)
    for this_xi, this_weight in zip(xi, weights, strict=True):
        total += (
            this_weight * gamma * point_vertical_velocity_mps(x, y + this_xi, z, params)
        )
    return total / gamma_integral


def lift_weighted_dwbardx_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Wbar)/dX for the full horseshoe model."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    xi, weights = _quadrature(params.span_m, params.quadrature_order)
    gamma = gamma0_m2_s(params)
    gamma_integral = _circulation_integral(params)
    total = np.zeros_like(x, dtype=float)
    for this_xi, this_weight in zip(xi, weights, strict=True):
        total += this_weight * gamma * point_dwdx_mps_per_m(x, y + this_xi, z, params)
    return total / gamma_integral


def lift_weighted_tip_pair_dwbardx_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Wbar)/dX using only the trailing tip-vortex pair."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    xi, weights = _quadrature(params.span_m, params.quadrature_order)
    gamma = gamma0_m2_s(params)
    gamma_integral = _circulation_integral(params)
    total = np.zeros_like(x, dtype=float)
    for this_xi, this_weight in zip(xi, weights, strict=True):
        total += (
            this_weight
            * gamma
            * point_trailing_dwdx_mps_per_m(x, y + this_xi, z, params)
        )
    return total / gamma_integral


def lift_weighted_dwbardx_biot_savart_fd_mps_per_m(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Wbar)/dX by finite differencing Biot-Savart-evaluated Wbar."""
    step = max(
        params.gradient_step_over_span * params.span_m,
        10.0 * params.core_radius_m,
        1.0e-8,
    )
    return (
        lift_weighted_wbar_mps(
            np.asarray(x_m, dtype=float) + step,
            y_m,
            z_m,
            params,
        )
        - lift_weighted_wbar_mps(
            np.asarray(x_m, dtype=float) - step,
            y_m,
            z_m,
            params,
        )
    ) / (2.0 * step)


def pair_wake_terms(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
    *,
    gradient_method: str = "analytic",
) -> dict[str, np.ndarray]:
    """Return plus/minus wake values and their derivatives for pair dynamics."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    w_plus = lift_weighted_wbar_mps(x, y, z, params)
    w_minus = lift_weighted_wbar_mps(-x, -y, -z, params)
    if gradient_method in {"analytic", "compact_kernel"}:
        gradient_function = lift_weighted_dwbardx_mps_per_m
    elif gradient_method == "biot_savart_fd":
        gradient_function = lift_weighted_dwbardx_biot_savart_fd_mps_per_m
    else:
        raise ValueError(
            "gradient_method must be 'analytic', 'compact_kernel', or "
            "'biot_savart_fd'."
        )
    dw_plus = gradient_function(x, y, z, params)
    dw_minus = -gradient_function(-x, -y, -z, params)
    return {
        "w_plus_mps": w_plus,
        "w_minus_mps": w_minus,
        "dw_plus_dX_mps_per_m": dw_plus,
        "dw_minus_dX_mps_per_m": dw_minus,
    }


def power_change_w(
    wbar_mps: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return Delta P = -L Wbar."""
    return -lift_n(params) * np.asarray(wbar_mps, dtype=float)


def thrust_change_n(
    wbar_mps: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return Delta T = Delta P / U = -(L/U) Wbar."""
    return power_change_w(wbar_mps, params) / params.ubar_mps


def thrust_gradient_n_per_m(
    dwbardx_mps_per_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Delta T)/dX = -(L/U) dWbar/dX."""
    return (
        -lift_n(params) * np.asarray(dwbardx_mps_per_m, dtype=float) / params.ubar_mps
    )


def thrust_gradient_per_x_over_span_n(
    dwbardx_mps_per_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Delta T)/d(X/B), the plotted streamwise thrust gradient."""
    return params.span_m * thrust_gradient_n_per_m(dwbardx_mps_per_m, params)


def lift_weighted_thrust_gradient_per_x_over_span_n(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Delta T)/d(X/B) directly from the horseshoe dWbar/dX."""
    return thrust_gradient_per_x_over_span_n(
        lift_weighted_dwbardx_mps_per_m(x_m, y_m, z_m, params),
        params,
    )


def lift_weighted_tip_pair_thrust_gradient_per_x_over_span_n(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return d(Delta T)/d(X/B) from the lift-weighted tip-vortex-pair model."""
    return thrust_gradient_per_x_over_span_n(
        lift_weighted_tip_pair_dwbardx_mps_per_m(x_m, y_m, z_m, params),
        params,
    )


def stability_margin_per_s(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    z_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
    *,
    gradient_method: str = "analytic",
) -> np.ndarray:
    """Return M = -max(real(lambda)) for the full two-way streamwise Jacobian."""
    x, y, z = _broadcast_xyz(x_m, y_m, z_m)
    terms = pair_wake_terms(x, y, z, params, gradient_method=gradient_method)
    return _stability_margin_from_terms(x, terms, params)


def _stability_margin_from_terms(
    x: np.ndarray,
    terms: dict[str, np.ndarray],
    params: FlatWakeParams,
) -> np.ndarray:
    """Return stability margin using already-computed wake terms."""
    this_lift = lift_n(params)
    this_mass = mass_kg(params)
    ubar = params.ubar_mps
    beta_plus = (
        params.speed_damping_n_per_mps + this_lift * terms["w_plus_mps"] / ubar**2
    ) / this_mass
    beta_minus = (
        params.speed_damping_n_per_mps + this_lift * terms["w_minus_mps"] / ubar**2
    ) / this_mass
    a_plus = this_lift * terms["dw_plus_dX_mps_per_m"] / (this_mass * ubar)
    a_minus = this_lift * terms["dw_minus_dX_mps_per_m"] / (this_mass * ubar)

    shape = x.shape
    jacobian = np.zeros(shape + (3, 3), dtype=float)
    jacobian[..., 0, 1] = -1.0
    jacobian[..., 0, 2] = 1.0
    jacobian[..., 1, 0] = a_plus
    jacobian[..., 1, 1] = -beta_plus
    jacobian[..., 2, 0] = a_minus
    jacobian[..., 2, 2] = -beta_minus
    eigvals = np.linalg.eigvals(jacobian.reshape(-1, 3, 3)).reshape(shape + (3,))
    return -np.max(np.real(eigvals), axis=-1)


def compute_fields(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    params: FlatWakeParams = FlatWakeParams(),
    *,
    gradient_method: str = "analytic",
) -> dict[str, np.ndarray]:
    """Return power and stability fields for one coordinate grid."""
    terms = pair_wake_terms(x_m, y_m, z_m, params, gradient_method=gradient_method)
    return {
        "delta_p_1_w": power_change_w(terms["w_plus_mps"], params),
        "delta_p_avg_w": -0.5
        * lift_n(params)
        * (terms["w_plus_mps"] + terms["w_minus_mps"]),
        "stability_margin_per_s": _stability_margin_from_terms(
            np.asarray(x_m, dtype=float),
            terms,
            params,
        ),
        "dw_plus_dX_mps_per_m": terms["dw_plus_dX_mps_per_m"],
    }


def d_min_m(y_m: np.ndarray | float, params: FlatWakeParams = FlatWakeParams()):
    """Return the minimum possible spanwise element separation."""
    return np.maximum(np.abs(y_m) - params.span_m, 0.0)


def d_max_m(y_m: np.ndarray | float, params: FlatWakeParams = FlatWakeParams()):
    """Return the maximum possible spanwise element separation."""
    return np.abs(y_m) + params.span_m


def critical_radius_tip_pair_m(
    y_m: np.ndarray | float,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return R_c(|Y|) for the point-receiver tip-vortex-pair boundary."""
    y_abs = np.abs(np.asarray(y_m, dtype=float))
    semispan = 0.5 * params.span_m
    result = np.full_like(y_abs, np.nan, dtype=float)
    valid = y_abs > semispan
    a_v = y_abs[valid] - semispan
    c_v = y_abs[valid] + semispan
    rc2 = (a_v * c_v) ** (2.0 / 3.0) * (a_v ** (2.0 / 3.0) + c_v ** (2.0 / 3.0))
    result[valid] = np.sqrt(rc2)
    return result


def tip_pair_neutral_x_over_span(
    y_over_span: np.ndarray | float,
    z_over_span: np.ndarray | float = 0.0,
    params: FlatWakeParams = FlatWakeParams(),
) -> np.ndarray:
    """Return X/B on the point-tip-pair neutral boundary."""
    y_m = np.asarray(y_over_span, dtype=float) * params.span_m
    z_m = np.asarray(z_over_span, dtype=float) * params.span_m
    rc = critical_radius_tip_pair_m(y_m, params)
    rc2_minus_z2 = rc**2 - z_m**2
    return np.where(rc2_minus_z2 > 0.0, np.sqrt(rc2_minus_z2) / params.span_m, np.nan)


def model_parameters_dict(params: FlatWakeParams = FlatWakeParams()) -> dict[str, Any]:
    """Return JSON-friendly model parameters."""
    return {
        "model": "constant-circulation full horseshoe vortex plus tip-vortex boundary",
        "span_m": params.span_m,
        "semispan_m": 0.5 * params.span_m,
        "chord_m": params.chord_m,
        "aspect_ratio": aspect_ratio(params),
        "rho_kg_m3": params.rho_kg_m3,
        "reference_speed_mps": params.ubar_mps,
        "reference_aoa_deg": params.aoa_deg,
        "finite_wing_lift_slope_per_rad": finite_wing_lift_slope_per_rad(params),
        "lift_coefficient": lift_coefficient(params),
        "lift_n": lift_n(params),
        "level_trim_mass_kg": mass_kg(params),
        "horseshoe_circulation_gamma_t_m2_s": gamma0_m2_s(params),
        "speed_damping_n_per_mps": params.speed_damping_n_per_mps,
        "baseline_thrust_n": baseline_thrust_n(params),
        "quadrature_order": params.quadrature_order,
        "core_radius_m": params.core_radius_m,
        "wake_length_over_span_for_vector_plots": params.wake_length_over_span,
        "wake_num_segments_for_vector_plots": params.wake_num_segments,
        "stability_boundary_reference": (
            "point-receiver tip-vortex-pair contour X^2+Z^2=R_c^2(|Y|)"
        ),
        "power_to_thrust_relation": "Delta T = Delta P/U = -(L/U) * Wbar",
    }


def _style() -> None:
    """Apply a consistent style for analytical model figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "#f8f8f8",
            "axes.edgecolor": "#222222",
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.55,
            "axes.grid": True,
        }
    )


def make_slices(
    params: FlatWakeParams = FlatWakeParams(),
    n: int = 141,
    xlim_over_span: tuple[float, float] = (-2.0, 8.0),
    ylim_over_span: tuple[float, float] = (-5.0, 5.0),
    zlim_over_span: tuple[float, float] = (-5.0, 5.0),
) -> list[dict[str, Any]]:
    """Return the three coordinate slices used in the 9-panel figure."""
    span = params.span_m
    xvals = np.linspace(xlim_over_span[0] * span, xlim_over_span[1] * span, n)
    yvals = np.linspace(ylim_over_span[0] * span, ylim_over_span[1] * span, n)
    zvals = np.linspace(zlim_over_span[0] * span, zlim_over_span[1] * span, n)
    x_xy, y_xy = np.meshgrid(xvals, yvals, indexing="xy")
    y_yz, z_yz = np.meshgrid(yvals, zvals, indexing="xy")
    x_xz, z_xz = np.meshgrid(xvals, zvals, indexing="xy")
    return [
        {
            "plane": "xy",
            "title": "Z/B = 0",
            "X": x_xy,
            "Y": y_xy,
            "Z": np.zeros_like(x_xy),
            "xplot": x_xy / span,
            "yplot": y_xy / span,
            "xlabel": "X/B",
            "ylabel": "Y/B",
        },
        {
            "plane": "yz",
            "title": "X/B = 0",
            "X": np.zeros_like(y_yz),
            "Y": y_yz,
            "Z": z_yz,
            "xplot": y_yz / span,
            "yplot": z_yz / span,
            "xlabel": "Y/B",
            "ylabel": "Z/B",
        },
        {
            "plane": "xz",
            "title": "Y/B = 0",
            "X": x_xz,
            "Y": np.zeros_like(x_xz),
            "Z": z_xz,
            "xplot": x_xz / span,
            "yplot": z_xz / span,
            "xlabel": "X/B",
            "ylabel": "Z/B",
        },
    ]


def _symmetric_limits(
    arrays: list[np.ndarray], q: float = 0.985
) -> tuple[float, float]:
    """Return robust symmetric color limits around zero."""
    finite = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    if finite.size == 0:
        return -1.0, 1.0
    nonzero = np.abs(finite[np.abs(finite) > 1.0e-14])
    if nonzero.size == 0:
        return -1.0, 1.0
    vmax = float(np.quantile(nonzero, q))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(nonzero))
    return -vmax, vmax


def _draw_leader_wing_marker(
    ax: plt.Axes,
    plane: str,
    params: FlatWakeParams,
) -> None:
    """Draw the leader/source wing footprint in normalized coordinates."""
    half_span = 0.5
    chord = params.chord_m / params.span_m
    thickness = 0.04
    style = {
        "color": "black",
        "linewidth": 2.2,
        "solid_capstyle": "round",
        "zorder": 8,
    }
    if plane == "xy":
        x1, x2 = 0.0, chord
        y1, y2 = -half_span, half_span
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], **style)
    elif plane == "yz":
        y1, y2 = -half_span, half_span
        z1, z2 = -0.5 * thickness, 0.5 * thickness
        ax.plot([y1, y2, y2, y1, y1], [z1, z1, z2, z2, z1], **style)
    elif plane == "xz":
        x1, x2 = 0.0, chord
        z1, z2 = -0.5 * thickness, 0.5 * thickness
        ax.plot([x1, x2, x2, x1, x1], [z1, z1, z2, z2, z1], **style)


def plot_stability_maps(
    output_path: Path = DEFAULT_OUTPUT_FIGURE,
    params: FlatWakeParams = FlatWakeParams(),
    n: int = 141,
) -> Path:
    """Save the 3-by-3 power and streamwise-stability figure."""
    _style()
    slices = make_slices(params=params, n=n)
    results = [compute_fields(sl["X"], sl["Y"], sl["Z"], params) for sl in slices]
    fields = [
        {
            "key": "delta_p_1_w",
            "label": r"$\Delta P_1 = -L \mathcal{W}_+$",
            "cbar": "power change [W]\nblue=saving, red=extra power",
            "cmap": "RdBu_r",
        },
        {
            "key": "delta_p_avg_w",
            "label": r"$\Delta P_{avg} = -L(\mathcal{W}_+ + \mathcal{W}_-)/2$",
            "cbar": "power change [W]\nblue=saving, red=extra power",
            "cmap": "RdBu_r",
        },
        {
            "key": "stability_margin_per_s",
            "label": r"$M = -\max \Re(\lambda)$",
            "cbar": "stability margin [1/s]\nblue=stable, red=unstable",
            "cmap": "RdBu",
        },
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.0), constrained_layout=True)
    for row, field in enumerate(fields):
        for col, sl in enumerate(slices):
            ax = axes[row, col]
            arr = results[col][field["key"]]
            vmin, vmax = _symmetric_limits([arr])
            mesh = ax.pcolormesh(
                sl["xplot"],
                sl["yplot"],
                arr,
                shading="auto",
                cmap=field["cmap"],
                vmin=vmin,
                vmax=vmax,
            )
            if np.nanmin(arr) <= 0.0 <= np.nanmax(arr):
                ax.contour(
                    sl["xplot"],
                    sl["yplot"],
                    arr,
                    levels=[0.0],
                    colors="black",
                    linestyles="--",
                    linewidths=1.3 if row < 2 else 2.0,
                )
            _draw_leader_wing_marker(ax, sl["plane"], params)
            if row == 0:
                ax.set_title(sl["title"])
            ax.set_xlabel(sl["xlabel"])
            ax.set_ylabel(
                f"{field['label']}\n{sl['ylabel']}" if col == 0 else sl["ylabel"]
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
            cbar.set_label(field["cbar"])

    fig.suptitle(
        "Biot-Savart Horseshoe/Tip-Vortex Power Saving and Streamwise Stability",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FIGURE)
    parser.add_argument("--grid-size", type=int, default=141)
    return parser.parse_args()


def main() -> None:
    """Build the default streamwise stability figure."""
    args = parse_args()
    output_path = plot_stability_maps(output_path=args.output, n=args.grid_size)
    print(output_path)


if __name__ == "__main__":
    main()
