"""Render an 8-panel single-wing wake comparison (Simulation vs Analytical).

Rows (same view, side-by-side):
1) XY disturbance-velocity quiver slice
2) XY vertical-velocity (u_z) colormap
3) YZ streamline slice
4) XZ disturbance-velocity quiver slice

Columns:
- Left: simulation wake (saved ring-vortex wake)
- Right: analytical wake (two trailing tip-vortex filaments)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from pterasoftware import _aerodynamics_functions

_STREAMWISE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "streamwise_stability_energy"
)
_SINGLE_WING_CASE_NAME = (
    "single_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_t9s_corrected_translating_dt12"
)
_RAW_DEFAULT_CASE_DIR = _STREAMWISE_ROOT / "raw_sources" / _SINGLE_WING_CASE_NAME
DEFAULT_CASE_DIR = (
    _RAW_DEFAULT_CASE_DIR
    if _RAW_DEFAULT_CASE_DIR.exists()
    else _STREAMWISE_ROOT / _SINGLE_WING_CASE_NAME
)
DEFAULT_STEP = 863
DEFAULT_OUTPUT = DEFAULT_CASE_DIR / "single_wing_wake_eight_panel_sim_vs_analytic.png"

# Geometry/constants used in this single-wing sweep family.
SPAN_M = 1.0
CHORD_M = 0.1
HALF_SPAN_M = SPAN_M / 2.0
AIR_DENSITY_KG_M3 = 1.225
REFERENCE_SPEED_MPS = 1.0
WEIGHT_N = 0.0235
TIP_VORTEX_GAMMA_SCALE = 0.665
TIP_VORTEX_CORE_RADIUS_M = 0.003


def _apply_style() -> None:
    """Apply a shared visual style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.facecolor": "#f7f7f7",
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.6,
            "axes.grid": True,
        }
    )


def _load_snapshot(case_dir: Path, step: int) -> dict[str, np.ndarray]:
    """Load one streamed-history snapshot."""
    snapshot = case_dir / "streamed_history" / f"step_{step:06d}.npz"
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    with np.load(snapshot) as data:
        return {name: data[name] for name in data.files}


def _bp1_to_e(
    points_bp1: np.ndarray,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
) -> np.ndarray:
    """Convert row-vector points from body frame BP1 to Earth frame E."""
    R_pas_BP1_to_E = R_pas_E_to_BP1.T
    return points_bp1 @ R_pas_BP1_to_E.T + pos_E


def _make_plane_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return meshgrid arrays."""
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    return np.meshgrid(x, y)


def _yz_vortex_seed_points(
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Return clean seed points around the two tip vortices in the YZ plane."""
    seed_points: list[tuple[float, float]] = []
    radii = (0.10, 0.18, 0.30, 0.46, 0.64)
    for y_center in (-HALF_SPAN_M, HALF_SPAN_M):
        z_center = 0.0
        for radius in radii:
            outward_angle = np.pi if y_center < 0.0 else 0.0
            for angle in (outward_angle, np.pi / 2.0):
                y = y_center + radius * np.cos(angle)
                z = z_center + radius * np.sin(angle)
                if y_min <= y <= y_max and z_min <= z <= z_max:
                    seed_points.append((y, z))
    return np.asarray(seed_points, dtype=float)


def _quiver_scale_for_axes(
    magnitudes: tuple[np.ndarray, ...],
    target_axis_length_m: float,
) -> float:
    """Return a quiver scale that makes disturbance arrows readable."""
    finite_values = np.concatenate(
        [mag[np.isfinite(mag)].ravel() for mag in magnitudes if mag.size > 0]
    )
    if finite_values.size == 0:
        return 1.0
    reference_speed = max(float(np.nanpercentile(finite_values, 95.0)), 1.0e-8)
    return reference_speed / target_axis_length_m


def _sim_wake_velocity_E(
    points_E: np.ndarray,
    wake_br_E: np.ndarray,
    wake_fr_E: np.ndarray,
    wake_fl_E: np.ndarray,
    wake_bl_E: np.ndarray,
    wake_strengths: np.ndarray,
    wake_rc0s: np.ndarray,
    wake_ages: np.ndarray,
) -> np.ndarray:
    """Compute simulation-induced velocity from saved ring-wake vortices."""
    singularity_counts = np.zeros(4, dtype=np.int64)
    return _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
        stackP_GP1_CgP1=points_E,
        stackBrrvp_GP1_CgP1=wake_br_E,
        stackFrrvp_GP1_CgP1=wake_fr_E,
        stackFlrvp_GP1_CgP1=wake_fl_E,
        stackBlrvp_GP1_CgP1=wake_bl_E,
        strengths=wake_strengths,
        r_c0s=wake_rc0s,
        singularity_counts=singularity_counts,
        ages=wake_ages,
        nu=0.0,
    )


def _analytic_wake_velocity_E(
    points_E: np.ndarray,
    start_right_E: np.ndarray,
    start_left_E: np.ndarray,
    end_x_E: float,
    tip_gamma_m2_s: float,
    sign: float,
) -> np.ndarray:
    """Compute analytical velocity from two long trailing tip-vortex filaments."""
    starts = np.vstack((start_right_E, start_left_E))
    ends = np.vstack(
        (
            np.array([end_x_E, start_right_E[1], start_right_E[2]], dtype=float),
            np.array([end_x_E, start_left_E[1], start_left_E[2]], dtype=float),
        )
    )
    strengths = np.array([sign * tip_gamma_m2_s, -sign * tip_gamma_m2_s], dtype=float)
    r_c0s = np.full(2, TIP_VORTEX_CORE_RADIUS_M, dtype=float)
    singularity_counts = np.zeros(4, dtype=np.int64)
    return _aerodynamics_functions._collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=points_E,
        stackSlvp_GP1_CgP1=starts,
        stackElvp_GP1_CgP1=ends,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=0.0,
    )


def _choose_analytic_sign(
    points_xy_E: np.ndarray,
    sim_uz_xy: np.ndarray,
    start_right_E: np.ndarray,
    start_left_E: np.ndarray,
    end_x_E: float,
    tip_gamma_m2_s: float,
) -> float:
    """Choose filament sign convention by maximizing XY u_z correlation to simulation."""
    best_sign = 1.0
    best_corr = -np.inf
    sim_flat = sim_uz_xy.reshape(-1)
    for sign in (1.0, -1.0):
        vel = _analytic_wake_velocity_E(
            points_E=points_xy_E,
            start_right_E=start_right_E,
            start_left_E=start_left_E,
            end_x_E=end_x_E,
            tip_gamma_m2_s=tip_gamma_m2_s,
            sign=sign,
        )
        uz = vel[:, 2]
        corr = float(np.corrcoef(sim_flat, uz)[0, 1])
        if not np.isfinite(corr):
            corr = -np.inf
        if corr > best_corr:
            best_corr = corr
            best_sign = sign
    return best_sign


def _draw_wing_projection(
    ax_xy: plt.Axes,
    ax_yz: plt.Axes,
    ax_xz: plt.Axes,
    ax_uz: plt.Axes,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
    x_origin_E: float,
) -> None:
    """Draw solid wing projections."""
    wing_xy_bp1 = np.array(
        [
            [0.0, -HALF_SPAN_M, 0.0],
            [CHORD_M, -HALF_SPAN_M, 0.0],
            [CHORD_M, HALF_SPAN_M, 0.0],
            [0.0, HALF_SPAN_M, 0.0],
        ],
        dtype=float,
    )
    wing_xy_E = _bp1_to_e(wing_xy_bp1, R_pas_E_to_BP1, pos_E)
    wing_xy_plot_x = wing_xy_E[:, 0] - x_origin_E
    ax_xy.fill(wing_xy_plot_x, wing_xy_E[:, 1], color="black", alpha=0.95, zorder=7)
    ax_uz.fill(wing_xy_plot_x, wing_xy_E[:, 1], color="black", alpha=0.95, zorder=7)

    span_segment_bp1 = np.array(
        [[0.5 * CHORD_M, -HALF_SPAN_M, 0.0], [0.5 * CHORD_M, HALF_SPAN_M, 0.0]],
        dtype=float,
    )
    span_segment_E = _bp1_to_e(span_segment_bp1, R_pas_E_to_BP1, pos_E)
    ax_yz.plot(
        span_segment_E[:, 1],
        span_segment_E[:, 2],
        color="black",
        linewidth=5.0,
        solid_capstyle="round",
        zorder=7,
    )

    chord_segment_bp1 = np.array([[0.0, 0.0, 0.0], [CHORD_M, 0.0, 0.0]], dtype=float)
    chord_segment_E = _bp1_to_e(chord_segment_bp1, R_pas_E_to_BP1, pos_E)
    ax_xz.plot(
        chord_segment_E[:, 0] - x_origin_E,
        chord_segment_E[:, 2],
        color="black",
        linewidth=5.0,
        solid_capstyle="round",
        zorder=7,
    )


def build_figure(case_dir: Path, step: int, output: Path) -> None:
    """Build and save the 8-panel comparison figure."""
    _apply_style()
    data = _load_snapshot(case_dir=case_dir, step=step)
    pos_E = data["position_E_E"]
    R_pas_E_to_BP1 = data["R_pas_E_to_BP1"]

    wake_br_E = _bp1_to_e(data["wake_br"], R_pas_E_to_BP1, pos_E)
    wake_fr_E = _bp1_to_e(data["wake_fr"], R_pas_E_to_BP1, pos_E)
    wake_fl_E = _bp1_to_e(data["wake_fl"], R_pas_E_to_BP1, pos_E)
    wake_bl_E = _bp1_to_e(data["wake_bl"], R_pas_E_to_BP1, pos_E)

    # Domains are queried in Earth frame, then plotted in a body-following x frame.
    x_min, x_max = float(pos_E[0] - 0.4), float(pos_E[0] + 4.5)
    y_min, y_max = float(pos_E[1] - 0.9), float(pos_E[1] + 0.9)
    z_min, z_max = float(pos_E[2] - 0.8), float(pos_E[2] + 0.8)
    x_plot_origin_E = float(pos_E[0])
    x_min_plot = x_min - x_plot_origin_E
    x_max_plot = x_max - x_plot_origin_E

    z_xy = float(pos_E[2])
    x_yz = float(pos_E[0] + 1.0)
    x_yz_plot = x_yz - x_plot_origin_E
    y_xz = float(pos_E[1])

    # Query grids.
    X_xy, Y_xy = _make_plane_grid(x_min, x_max, y_min, y_max, nx=230, ny=160)
    X_xy_plot = X_xy - x_plot_origin_E
    points_xy = np.column_stack((X_xy.ravel(), Y_xy.ravel(), np.full(X_xy.size, z_xy)))

    Y_yz, Z_yz = _make_plane_grid(y_min, y_max, z_min, z_max, nx=180, ny=180)
    points_yz = np.column_stack((np.full(Y_yz.size, x_yz), Y_yz.ravel(), Z_yz.ravel()))

    X_xz, Z_xz = _make_plane_grid(x_min, x_max, z_min, z_max, nx=230, ny=160)
    X_xz_plot = X_xz - x_plot_origin_E
    points_xz = np.column_stack((X_xz.ravel(), np.full(X_xz.size, y_xz), Z_xz.ravel()))

    # Simulation fields.
    sim_vel_xy = _sim_wake_velocity_E(
        points_E=points_xy,
        wake_br_E=wake_br_E,
        wake_fr_E=wake_fr_E,
        wake_fl_E=wake_fl_E,
        wake_bl_E=wake_bl_E,
        wake_strengths=data["wake_strengths"],
        wake_rc0s=data["wake_rc0s"],
        wake_ages=data["wake_ages"],
    )
    sim_vel_yz = _sim_wake_velocity_E(
        points_E=points_yz,
        wake_br_E=wake_br_E,
        wake_fr_E=wake_fr_E,
        wake_fl_E=wake_fl_E,
        wake_bl_E=wake_bl_E,
        wake_strengths=data["wake_strengths"],
        wake_rc0s=data["wake_rc0s"],
        wake_ages=data["wake_ages"],
    )
    sim_vel_xz = _sim_wake_velocity_E(
        points_E=points_xz,
        wake_br_E=wake_br_E,
        wake_fr_E=wake_fr_E,
        wake_fl_E=wake_fl_E,
        wake_bl_E=wake_bl_E,
        wake_strengths=data["wake_strengths"],
        wake_rc0s=data["wake_rc0s"],
        wake_ages=data["wake_ages"],
    )

    # Analytical setup: two long trailing tip filaments.
    trailing_right_bp1 = np.array([CHORD_M, +HALF_SPAN_M, 0.0], dtype=float)
    trailing_left_bp1 = np.array([CHORD_M, -HALF_SPAN_M, 0.0], dtype=float)
    start_right_E = _bp1_to_e(trailing_right_bp1[None, :], R_pas_E_to_BP1, pos_E)[0]
    start_left_E = _bp1_to_e(trailing_left_bp1[None, :], R_pas_E_to_BP1, pos_E)[0]
    end_x_E = x_max + 12.0
    gamma_base = WEIGHT_N / (AIR_DENSITY_KG_M3 * REFERENCE_SPEED_MPS * SPAN_M)
    tip_gamma = TIP_VORTEX_GAMMA_SCALE * gamma_base

    sim_uz_xy = sim_vel_xy[:, 2]
    sign = _choose_analytic_sign(
        points_xy_E=points_xy,
        sim_uz_xy=sim_uz_xy,
        start_right_E=start_right_E,
        start_left_E=start_left_E,
        end_x_E=end_x_E,
        tip_gamma_m2_s=tip_gamma,
    )

    ana_vel_xy = _analytic_wake_velocity_E(
        points_E=points_xy,
        start_right_E=start_right_E,
        start_left_E=start_left_E,
        end_x_E=end_x_E,
        tip_gamma_m2_s=tip_gamma,
        sign=sign,
    )
    ana_vel_yz = _analytic_wake_velocity_E(
        points_E=points_yz,
        start_right_E=start_right_E,
        start_left_E=start_left_E,
        end_x_E=end_x_E,
        tip_gamma_m2_s=tip_gamma,
        sign=sign,
    )
    ana_vel_xz = _analytic_wake_velocity_E(
        points_E=points_xz,
        start_right_E=start_right_E,
        start_left_E=start_left_E,
        end_x_E=end_x_E,
        tip_gamma_m2_s=tip_gamma,
        sign=sign,
    )

    # Reshape for plotting. Quiver rows use disturbance velocity only; the
    # U=1 m/s freestream is intentionally removed.
    sim_U_xy = sim_vel_xy[:, 0].reshape(X_xy.shape)
    sim_V_xy = sim_vel_xy[:, 1].reshape(X_xy.shape)
    sim_speed_xy = np.hypot(sim_U_xy, sim_V_xy)
    sim_Uz_xy = sim_vel_xy[:, 2].reshape(X_xy.shape)

    sim_U_yz = sim_vel_yz[:, 1].reshape(Y_yz.shape)
    sim_V_yz = sim_vel_yz[:, 2].reshape(Y_yz.shape)
    sim_speed_yz = np.hypot(sim_U_yz, sim_V_yz)

    sim_U_xz = sim_vel_xz[:, 0].reshape(X_xz.shape)
    sim_V_xz = sim_vel_xz[:, 2].reshape(X_xz.shape)
    sim_speed_xz = np.hypot(sim_U_xz, sim_V_xz)

    ana_U_xy = ana_vel_xy[:, 0].reshape(X_xy.shape)
    ana_V_xy = ana_vel_xy[:, 1].reshape(X_xy.shape)
    ana_speed_xy = np.hypot(ana_U_xy, ana_V_xy)
    ana_Uz_xy = ana_vel_xy[:, 2].reshape(X_xy.shape)

    ana_U_yz = ana_vel_yz[:, 1].reshape(Y_yz.shape)
    ana_V_yz = ana_vel_yz[:, 2].reshape(Y_yz.shape)
    ana_speed_yz = np.hypot(ana_U_yz, ana_V_yz)

    ana_U_xz = ana_vel_xz[:, 0].reshape(X_xz.shape)
    ana_V_xz = ana_vel_xz[:, 2].reshape(X_xz.shape)
    ana_speed_xz = np.hypot(ana_U_xz, ana_V_xz)

    speed_max_xy = max(
        float(np.nanpercentile(sim_speed_xy, 99.0)),
        float(np.nanpercentile(ana_speed_xy, 99.0)),
        1.0e-6,
    )
    speed_max_yz = max(
        float(np.nanpercentile(sim_speed_yz, 99.0)),
        float(np.nanpercentile(ana_speed_yz, 99.0)),
        1.0e-6,
    )
    speed_max_xz = max(
        float(np.nanpercentile(sim_speed_xz, 99.0)),
        float(np.nanpercentile(ana_speed_xz, 99.0)),
        1.0e-6,
    )
    quiver_norm_xy = Normalize(vmin=0.0, vmax=speed_max_xy)
    stream_norm_yz = Normalize(vmin=0.0, vmax=speed_max_yz)
    quiver_norm_xz = Normalize(vmin=0.0, vmax=speed_max_xz)

    uz_lim = float(
        np.nanmax(
            [
                np.nanpercentile(np.abs(sim_Uz_xy), 99.0),
                np.nanpercentile(np.abs(ana_Uz_xy), 99.0),
            ]
        )
    )
    uz_lim = max(uz_lim, 1.0e-6)
    uz_norm = TwoSlopeNorm(vmin=-uz_lim, vcenter=0.0, vmax=uz_lim)

    fig, axes = plt.subplots(4, 2, figsize=(13.8, 17.5), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.015,
        h_pad=0.035,
        wspace=0.025,
        hspace=0.04,
    )

    yz_stream_kw = {
        "linewidth": 0.9,
        "cmap": "viridis",
        "norm": stream_norm_yz,
        "arrowsize": 0.65,
        "start_points": _yz_vortex_seed_points(y_min, y_max, z_min, z_max),
        "integration_direction": "both",
        "maxlength": 6.0,
        "broken_streamlines": False,
    }
    quiver_kw = {
        "cmap": "viridis",
        "angles": "xy",
        "scale_units": "xy",
        "width": 0.0042,
        "headwidth": 5.0,
        "headlength": 7.0,
        "headaxislength": 5.8,
        "minshaft": 1.5,
        "minlength": 0.0,
        "pivot": "middle",
        "edgecolor": "#171717",
        "linewidth": 0.18,
        "alpha": 0.98,
    }

    # Row 1: XY disturbance quiver.
    xy_stride = (slice(None, None, 10), slice(None, None, 14))
    xy_sim_quiver_scale = _quiver_scale_for_axes(
        (sim_speed_xy,),
        target_axis_length_m=0.24,
    )
    xy_ana_quiver_scale = _quiver_scale_for_axes(
        (ana_speed_xy,),
        target_axis_length_m=0.24,
    )
    q_sim_xy = axes[0, 0].quiver(
        X_xy_plot[xy_stride],
        Y_xy[xy_stride],
        sim_U_xy[xy_stride],
        sim_V_xy[xy_stride],
        sim_speed_xy[xy_stride],
        norm=quiver_norm_xy,
        scale=xy_sim_quiver_scale,
        **quiver_kw,
    )
    axes[0, 1].quiver(
        X_xy_plot[xy_stride],
        Y_xy[xy_stride],
        ana_U_xy[xy_stride],
        ana_V_xy[xy_stride],
        ana_speed_xy[xy_stride],
        norm=quiver_norm_xy,
        scale=xy_ana_quiver_scale,
        **quiver_kw,
    )
    for ax in axes[0, :]:
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min, y_max)

    # Row 2: XY uz.
    uz_sim_mesh = axes[1, 0].pcolormesh(
        X_xy_plot, Y_xy, sim_Uz_xy, shading="auto", cmap="RdBu_r", norm=uz_norm
    )
    axes[1, 1].pcolormesh(
        X_xy_plot, Y_xy, ana_Uz_xy, shading="auto", cmap="RdBu_r", norm=uz_norm
    )
    for ax in axes[1, :]:
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min, y_max)

    # Row 3: YZ streamlines.
    axes[2, 0].streamplot(
        Y_yz, Z_yz, sim_U_yz, sim_V_yz, color=sim_speed_yz, **yz_stream_kw
    )
    st_ana_yz = axes[2, 1].streamplot(
        Y_yz, Z_yz, ana_U_yz, ana_V_yz, color=ana_speed_yz, **yz_stream_kw
    )

    # Row 4: XZ disturbance quiver.
    xz_stride = (slice(None, None, 10), slice(None, None, 14))
    xz_sim_quiver_scale = _quiver_scale_for_axes(
        (sim_speed_xz,),
        target_axis_length_m=0.26,
    )
    xz_ana_quiver_scale = _quiver_scale_for_axes(
        (ana_speed_xz,),
        target_axis_length_m=0.26,
    )
    q_sim_xz = axes[3, 0].quiver(
        X_xz_plot[xz_stride],
        Z_xz[xz_stride],
        sim_U_xz[xz_stride],
        sim_V_xz[xz_stride],
        sim_speed_xz[xz_stride],
        norm=quiver_norm_xz,
        scale=xz_sim_quiver_scale,
        **quiver_kw,
    )
    axes[3, 1].quiver(
        X_xz_plot[xz_stride],
        Z_xz[xz_stride],
        ana_U_xz[xz_stride],
        ana_V_xz[xz_stride],
        ana_speed_xz[xz_stride],
        norm=quiver_norm_xz,
        scale=xz_ana_quiver_scale,
        **quiver_kw,
    )
    for ax in axes[3, :]:
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(z_min, z_max)

    # Wing overlays.
    for col in range(2):
        _draw_wing_projection(
            ax_xy=axes[0, col],
            ax_yz=axes[2, col],
            ax_xz=axes[3, col],
            ax_uz=axes[1, col],
            R_pas_E_to_BP1=R_pas_E_to_BP1,
            pos_E=pos_E,
            x_origin_E=x_plot_origin_E,
        )

    # Titles.
    axes[0, 0].set_title(f"Simulation: XY Disturbance Quiver (z={z_xy:.3f} m)")
    axes[0, 1].set_title("Analytical: XY Disturbance Quiver")
    axes[1, 0].set_title("Simulation: XY Vertical Velocity $u_z$")
    axes[1, 1].set_title("Analytical: XY Vertical Velocity $u_z$")
    axes[2, 0].set_title(f"Simulation: YZ Streamlines (x={x_yz_plot:.3f} m)")
    axes[2, 1].set_title("Analytical: YZ Streamlines")
    axes[3, 0].set_title(f"Simulation: XZ Disturbance Quiver (y={y_xz:.3f} m)")
    axes[3, 1].set_title("Analytical: XZ Disturbance Quiver")

    # Labels.
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")
    axes[2, 0].set_ylabel("z (m)")
    axes[3, 0].set_ylabel("z (m)")
    for col in range(2):
        axes[0, col].tick_params(labelbottom=False)
        axes[1, col].set_xlabel("$x - x_\\mathrm{wing}$ (m)")
        axes[2, col].set_xlabel("y (m)")
        axes[3, col].set_xlabel("$x - x_\\mathrm{wing}$ (m)")
    for row in range(4):
        axes[row, 1].tick_params(labelleft=False)

    for row in range(4):
        for col in range(2):
            axes[row, col].set_aspect("equal", adjustable="box")

    c_xy = fig.colorbar(q_sim_xy, ax=axes[0, :], fraction=0.03, pad=0.01)
    c_xy.set_label("XY disturbance speed (m/s)")

    c_uz = fig.colorbar(uz_sim_mesh, ax=axes[1, :], fraction=0.04, pad=0.02)
    c_uz.set_label("$u_z$ (m/s)")

    c_stream_yz = fig.colorbar(
        st_ana_yz.lines,
        ax=axes[2, :],
        fraction=0.03,
        pad=0.01,
    )
    c_stream_yz.set_label("YZ in-plane speed (m/s)")

    c_xz = fig.colorbar(q_sim_xz, ax=axes[3, :], fraction=0.03, pad=0.01)
    c_xz.set_label("XZ disturbance speed (m/s)")

    fig.suptitle(
        f"Single-Wing Wake: Simulation vs Analytical (8 Panels, step {int(data['step'][0])})\n"
        "Wing starts at x=0; freestream removed; quiver lengths display-scaled per panel",
        y=1.01,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an 8-panel single-wing wake simulation-vs-analytic figure."
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    build_figure(case_dir=args.case_dir, step=args.step, output=args.output)
    print(args.output)


if __name__ == "__main__":
    main()
