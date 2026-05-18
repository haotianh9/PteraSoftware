"""Plot single-wing free-wake slices against a horseshoe wake model.

This figure compares the simulated free-vortex wake with the analytical velocity
field induced by a constant-circulation bound segment and two semi-infinite
trailing tip vortices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from pterasoftware import _aerodynamics_functions

try:
    from examples import horseshoe_tip_vortex_stability as horseshoe_model
except ImportError:  # pragma: no cover - supports direct script execution
    import horseshoe_tip_vortex_stability as horseshoe_model

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

SPAN_M = 1.0
CHORD_M = 0.1
HALF_SPAN_M = SPAN_M / 2.0
ANALYTICAL_ALPHA_DEG = 5.0
ANALYTICAL_SPEED_MPS = 1.0
ANALYTICAL_NUM_SPANWISE_QUAD = 384
ANALYTICAL_CORE_RADIUS_M = 0.01 * CHORD_M


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


def _e_to_bp1(
    points_E: np.ndarray,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
) -> np.ndarray:
    """Convert row-vector points from Earth frame E to body frame BP1."""
    return (points_E - pos_E) @ R_pas_E_to_BP1.T


def _bp1_vectors_to_e(
    vectors_bp1: np.ndarray,
    R_pas_E_to_BP1: np.ndarray,
) -> np.ndarray:
    """Rotate row-vector components from BP1 to Earth-frame components."""
    R_pas_BP1_to_E = R_pas_E_to_BP1.T
    return vectors_bp1 @ R_pas_BP1_to_E.T


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


def _analytical_horseshoe_velocity_bp1(
    points_bp1: np.ndarray,
    *,
    alpha_deg: float = ANALYTICAL_ALPHA_DEG,
    speed_mps: float = ANALYTICAL_SPEED_MPS,
    num_quad: int = ANALYTICAL_NUM_SPANWISE_QUAD,
) -> np.ndarray:
    """Return the shared manuscript horseshoe model velocity in the BP1 frame."""
    points = np.asarray(points_bp1, dtype=float)
    params = horseshoe_model.HorseshoeWakeParams(
        span_m=SPAN_M,
        chord_m=CHORD_M,
        ubar_mps=speed_mps,
        aoa_deg=alpha_deg,
        quadrature_order=num_quad,
        core_radius_m=ANALYTICAL_CORE_RADIUS_M,
    )
    return horseshoe_model.point_velocity_mps(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        params=params,
    )


def _analytical_horseshoe_velocity_E(
    points_E: np.ndarray,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
) -> np.ndarray:
    """Evaluate the analytical horseshoe wake and rotate it into Earth axes."""
    points_bp1 = _e_to_bp1(points_E, R_pas_E_to_BP1, pos_E)
    velocity_bp1 = _analytical_horseshoe_velocity_bp1(points_bp1)
    return _bp1_vectors_to_e(velocity_bp1, R_pas_E_to_BP1)


def _panel_norm(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local zero-centered color norm."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = max(1.0e-8, float(np.nanpercentile(np.abs(finite_values), 99.0)))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _draw_wing_xy(
    ax: plt.Axes,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
    x_origin_E: float,
) -> None:
    """Draw the solid wing footprint in an XY view."""
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
    ax.fill(wing_xy_plot_x, wing_xy_E[:, 1], color="black", alpha=0.95, zorder=7)


def _draw_wing_yz(
    ax: plt.Axes,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
) -> None:
    """Draw the solid wing footprint in a YZ view."""
    span_segment_bp1 = np.array(
        [[0.5 * CHORD_M, -HALF_SPAN_M, 0.0], [0.5 * CHORD_M, HALF_SPAN_M, 0.0]],
        dtype=float,
    )
    span_segment_E = _bp1_to_e(span_segment_bp1, R_pas_E_to_BP1, pos_E)
    ax.plot(
        span_segment_E[:, 1],
        span_segment_E[:, 2],
        color="black",
        linewidth=5.0,
        solid_capstyle="round",
        zorder=7,
    )


def _draw_wing_xz(
    ax: plt.Axes,
    R_pas_E_to_BP1: np.ndarray,
    pos_E: np.ndarray,
    x_origin_E: float,
) -> None:
    """Draw the solid wing footprint in an XZ view."""
    chord_segment_bp1 = np.array([[0.0, 0.0, 0.0], [CHORD_M, 0.0, 0.0]], dtype=float)
    chord_segment_E = _bp1_to_e(chord_segment_bp1, R_pas_E_to_BP1, pos_E)
    ax.plot(
        chord_segment_E[:, 0] - x_origin_E,
        chord_segment_E[:, 2],
        color="black",
        linewidth=5.0,
        solid_capstyle="round",
        zorder=7,
    )


def _format_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply common axis formatting."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _add_quiver_panel(
    ax: plt.Axes,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    scalar_grid: np.ndarray,
    u_plot: np.ndarray,
    v_plot: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
) -> None:
    """Plot a signed vertical-velocity panel with disturbance-velocity quivers."""
    speed = np.hypot(u_plot, v_plot)
    mesh = ax.pcolormesh(
        x_plot,
        y_plot,
        scalar_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=_panel_norm(scalar_grid),
    )
    stride_y = max(1, x_plot.shape[0] // 22)
    stride_x = max(1, x_plot.shape[1] // 28)
    sl = (slice(None, None, stride_y), slice(None, None, stride_x))

    sampled_speed = speed[sl]
    reference = max(1.0e-12, float(np.nanpercentile(sampled_speed, 98.0)))
    display_length = 0.055 * max(
        float(np.nanmax(x_plot) - np.nanmin(x_plot)),
        float(np.nanmax(y_plot) - np.nanmin(y_plot)),
    )
    u_display = u_plot[sl] / reference * display_length
    v_display = v_plot[sl] / reference * display_length

    ax.quiver(
        x_plot[sl],
        y_plot[sl],
        u_display,
        v_display,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        headwidth=4.5,
        headlength=5.5,
        headaxislength=4.8,
        alpha=0.88,
        zorder=6,
    )
    ax.set_title(title)
    _format_axes(ax, xlabel=xlabel, ylabel=ylabel)
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.012)
    cbar.set_label(cbar_label)


def _add_streamline_panel(
    ax: plt.Axes,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    scalar_grid: np.ndarray,
    u_plot: np.ndarray,
    v_plot: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
) -> None:
    """Plot a signed vertical-velocity panel with disturbance-velocity streamlines."""
    speed = np.hypot(u_plot, v_plot)
    mesh = ax.pcolormesh(
        x_plot,
        y_plot,
        scalar_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=_panel_norm(scalar_grid),
    )
    linewidth = 0.55 + 1.6 * speed / max(1.0e-12, float(np.nanpercentile(speed, 98.0)))
    ax.streamplot(
        x_plot[0, :],
        y_plot[:, 0],
        u_plot,
        v_plot,
        color="black",
        density=1.35,
        linewidth=np.clip(linewidth, 0.45, 2.0),
        arrowsize=0.9,
        minlength=0.08,
        zorder=6,
    )
    ax.set_title(title)
    _format_axes(ax, xlabel=xlabel, ylabel=ylabel)
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.012)
    cbar.set_label(cbar_label)


def _plot_pair(
    axes: np.ndarray,
    row: int,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    sim_grid: np.ndarray,
    model_grid: np.ndarray,
    title_left: str,
    title_right: str,
    xlabel: str,
    ylabel: str,
    right_cbar_label: str,
) -> None:
    """Plot one simulation/model row with separate panel colorbars."""
    sim_norm = _panel_norm(sim_grid)
    model_norm = _panel_norm(model_grid)
    sim_mesh = axes[row, 0].pcolormesh(
        x_plot,
        y_plot,
        sim_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=sim_norm,
    )
    model_mesh = axes[row, 1].pcolormesh(
        x_plot,
        y_plot,
        model_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=model_norm,
    )
    axes[row, 0].set_title(title_left)
    axes[row, 1].set_title(title_right)
    for col in range(2):
        axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].set_aspect("equal", adjustable="box")
        axes[row, col].grid(False)
    sim_cbar = axes[row, 0].figure.colorbar(
        sim_mesh,
        ax=axes[row, 0],
        fraction=0.046,
        pad=0.012,
    )
    sim_cbar.set_label("simulation $u_z$ (m/s)")
    model_cbar = axes[row, 0].figure.colorbar(
        model_mesh,
        ax=axes[row, 1],
        fraction=0.046,
        pad=0.012,
    )
    model_cbar.set_label(right_cbar_label)


def build_figure(case_dir: Path, step: int, output: Path) -> None:
    """Build and save the simulation-vs-analytical-wake slice comparison figure."""
    _apply_style()
    data = _load_snapshot(case_dir=case_dir, step=step)
    pos_E = data["position_E_E"]
    R_pas_E_to_BP1 = data["R_pas_E_to_BP1"]

    wake_br_E = _bp1_to_e(data["wake_br"], R_pas_E_to_BP1, pos_E)
    wake_fr_E = _bp1_to_e(data["wake_fr"], R_pas_E_to_BP1, pos_E)
    wake_fl_E = _bp1_to_e(data["wake_fl"], R_pas_E_to_BP1, pos_E)
    wake_bl_E = _bp1_to_e(data["wake_bl"], R_pas_E_to_BP1, pos_E)

    x_min, x_max = float(pos_E[0] - 0.4), float(pos_E[0] + 4.5)
    y_min, y_max = float(pos_E[1] - 0.9), float(pos_E[1] + 0.9)
    z_min, z_max = float(pos_E[2] - 0.8), float(pos_E[2] + 0.8)
    x_origin_E = float(pos_E[0])

    z_xy = float(pos_E[2])
    x_yz = float(pos_E[0] + 1.0)
    y_xz = float(pos_E[1])

    X_xy, Y_xy = _make_plane_grid(x_min, x_max, y_min, y_max, nx=230, ny=160)
    points_xy = np.column_stack((X_xy.ravel(), Y_xy.ravel(), np.full(X_xy.size, z_xy)))

    Y_yz, Z_yz = _make_plane_grid(y_min, y_max, z_min, z_max, nx=180, ny=180)
    points_yz = np.column_stack((np.full(Y_yz.size, x_yz), Y_yz.ravel(), Z_yz.ravel()))

    X_xz, Z_xz = _make_plane_grid(x_min, x_max, z_min, z_max, nx=230, ny=160)
    points_xz = np.column_stack((X_xz.ravel(), np.full(X_xz.size, y_xz), Z_xz.ravel()))

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
    analytical_vel_xy = _analytical_horseshoe_velocity_E(
        points_E=points_xy,
        R_pas_E_to_BP1=R_pas_E_to_BP1,
        pos_E=pos_E,
    )
    analytical_vel_yz = _analytical_horseshoe_velocity_E(
        points_E=points_yz,
        R_pas_E_to_BP1=R_pas_E_to_BP1,
        pos_E=pos_E,
    )
    analytical_vel_xz = _analytical_horseshoe_velocity_E(
        points_E=points_xz,
        R_pas_E_to_BP1=R_pas_E_to_BP1,
        pos_E=pos_E,
    )

    x_xy_plot = X_xy - x_origin_E
    x_xz_plot = X_xz - x_origin_E
    x_yz_plot = x_yz - x_origin_E

    sim_uz_xy = sim_vel_xy[:, 2].reshape(X_xy.shape)
    sim_uz_yz = sim_vel_yz[:, 2].reshape(Y_yz.shape)
    sim_uz_xz = sim_vel_xz[:, 2].reshape(X_xz.shape)
    sim_ux_xy = sim_vel_xy[:, 0].reshape(X_xy.shape)
    sim_uy_xy = sim_vel_xy[:, 1].reshape(X_xy.shape)
    sim_uy_yz = sim_vel_yz[:, 1].reshape(Y_yz.shape)
    sim_uz_yz_inplane = sim_vel_yz[:, 2].reshape(Y_yz.shape)
    sim_ux_xz = sim_vel_xz[:, 0].reshape(X_xz.shape)
    sim_uz_xz_inplane = sim_vel_xz[:, 2].reshape(X_xz.shape)
    analytical_uz_xy = analytical_vel_xy[:, 2].reshape(X_xy.shape)
    analytical_ux_xy = analytical_vel_xy[:, 0].reshape(X_xy.shape)
    analytical_uy_xy = analytical_vel_xy[:, 1].reshape(X_xy.shape)
    analytical_uz_yz = analytical_vel_yz[:, 2].reshape(Y_yz.shape)
    analytical_uy_yz = analytical_vel_yz[:, 1].reshape(Y_yz.shape)
    analytical_uz_yz_inplane = analytical_vel_yz[:, 2].reshape(Y_yz.shape)
    analytical_uz_xz = analytical_vel_xz[:, 2].reshape(X_xz.shape)
    analytical_ux_xz = analytical_vel_xz[:, 0].reshape(X_xz.shape)
    analytical_uz_xz_inplane = analytical_vel_xz[:, 2].reshape(X_xz.shape)

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.6), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.015,
        h_pad=0.035,
        wspace=0.025,
        hspace=0.04,
    )

    _add_quiver_panel(
        ax=axes[0, 0],
        x_plot=x_xy_plot,
        y_plot=Y_xy,
        scalar_grid=sim_uz_xy,
        u_plot=sim_ux_xy,
        v_plot=sim_uy_xy,
        title=f"Simulation: XY $u_z$ with disturbance quiver (Z/B=0, step {int(data['step'][0])})",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="y (m)",
        cbar_label="simulation $u_z$ (m/s)",
    )
    _add_quiver_panel(
        ax=axes[0, 1],
        x_plot=x_xy_plot,
        y_plot=Y_xy,
        scalar_grid=analytical_uz_xy,
        u_plot=analytical_ux_xy,
        v_plot=analytical_uy_xy,
        title="Analytical horseshoe: XY $u_z$ with velocity quiver",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="y (m)",
        cbar_label="analytical $u_z$ (m/s)",
    )

    _add_streamline_panel(
        ax=axes[1, 0],
        x_plot=Y_yz,
        y_plot=Z_yz,
        scalar_grid=sim_uz_yz,
        u_plot=sim_uy_yz,
        v_plot=sim_uz_yz_inplane,
        title=f"Simulation: YZ $u_z$ with disturbance streamlines (X/B={x_yz_plot / SPAN_M:.1f})",
        xlabel="y (m)",
        ylabel="z (m)",
        cbar_label="simulation $u_z$ (m/s)",
    )
    _add_streamline_panel(
        ax=axes[1, 1],
        x_plot=Y_yz,
        y_plot=Z_yz,
        scalar_grid=analytical_uz_yz,
        u_plot=analytical_uy_yz,
        v_plot=analytical_uz_yz_inplane,
        title=f"Analytical horseshoe: YZ $u_z$ with streamlines (X/B={x_yz_plot / SPAN_M:.1f})",
        xlabel="y (m)",
        ylabel="z (m)",
        cbar_label="analytical $u_z$ (m/s)",
    )

    _add_quiver_panel(
        ax=axes[2, 0],
        x_plot=x_xz_plot,
        y_plot=Z_xz,
        scalar_grid=sim_uz_xz,
        u_plot=sim_ux_xz,
        v_plot=sim_uz_xz_inplane,
        title="Simulation: XZ $u_z$ with disturbance quiver (Y/B=0)",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="z (m)",
        cbar_label="simulation $u_z$ (m/s)",
    )
    _add_quiver_panel(
        ax=axes[2, 1],
        x_plot=x_xz_plot,
        y_plot=Z_xz,
        scalar_grid=analytical_uz_xz,
        u_plot=analytical_ux_xz,
        v_plot=analytical_uz_xz_inplane,
        title="Analytical horseshoe: XZ $u_z$ with velocity quiver (Y/B=0)",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="z (m)",
        cbar_label="analytical $u_z$ (m/s)",
    )

    for col in range(2):
        _draw_wing_xy(axes[0, col], R_pas_E_to_BP1, pos_E, x_origin_E)
        _draw_wing_yz(axes[1, col], R_pas_E_to_BP1, pos_E)
        _draw_wing_xz(axes[2, col], R_pas_E_to_BP1, pos_E, x_origin_E)

    fig.suptitle(
        "Single-Wing Wake: Free-Wake Simulation vs Analytical Horseshoe/Tip-Vortex Model\n"
        "Analytical panels use the Biot-Savart velocity of a bound segment "
        "and two semi-infinite trailing tip vortices.",
        y=1.01,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a single-wing wake slice figure comparing simulated free-wake "
            "velocity to the analytical horseshoe/tip-vortex velocity field."
        )
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
