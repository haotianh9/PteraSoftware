"""Plot single-wing free-wake slices against the reduced Wbar model.

This figure compares the simulated vertical induced velocity ``u_z`` with the
analytical ``Wbar_model`` supplied in the external math-model script.  The model
column is not a reconstructed velocity vector field; it is the scalar lift-weighted
upwash closure used by the formation power/stability figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plot_streamwise_sim_by_aoa_single_analytic as streamwise_model
from matplotlib.colors import TwoSlopeNorm

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
DEFAULT_OUTPUT = DEFAULT_CASE_DIR / "single_wing_wake_slices_sim_vs_wbar_model.png"

SPAN_M = 1.0
CHORD_M = 0.1
HALF_SPAN_M = SPAN_M / 2.0


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


def _model_wbar_grid(
    x_rel_m: np.ndarray,
    y_rel_m: np.ndarray,
    z_rel_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the reduced analytical Wbar model on a grid."""
    vectorized_model = np.vectorize(streamwise_model.analytical_wbar_model_mps)
    return vectorized_model(
        x_rel_m / SPAN_M,
        y_rel_m / SPAN_M,
        z_rel_m / SPAN_M,
    )


def _row_norm(sim_grid: np.ndarray, model_grid: np.ndarray) -> TwoSlopeNorm:
    """Return a row-local zero-centered color norm for a sim/model pair."""
    finite_values = np.concatenate(
        (
            sim_grid[np.isfinite(sim_grid)].ravel(),
            model_grid[np.isfinite(model_grid)].ravel(),
        )
    )
    if finite_values.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = max(1.0e-8, float(np.nanpercentile(np.abs(finite_values), 99.0)))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _draw_wing_projection(
    ax_xy: plt.Axes,
    ax_yz: plt.Axes,
    ax_xz: plt.Axes,
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
) -> None:
    """Plot one simulation/model row with one shared colorbar."""
    norm = _row_norm(sim_grid, model_grid)
    sim_mesh = axes[row, 0].pcolormesh(
        x_plot,
        y_plot,
        sim_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    axes[row, 1].pcolormesh(
        x_plot,
        y_plot,
        model_grid,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    axes[row, 0].set_title(title_left)
    axes[row, 1].set_title(title_right)
    for col in range(2):
        axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].set_aspect("equal", adjustable="box")
        axes[row, col].grid(False)
    cbar = axes[row, 0].figure.colorbar(
        sim_mesh,
        ax=axes[row, :],
        fraction=0.035,
        pad=0.012,
    )
    cbar.set_label("vertical induced velocity / reduced upwash (m/s)")


def build_figure(case_dir: Path, step: int, output: Path) -> None:
    """Build and save the simulation-vs-Wbar-model slice comparison figure."""
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

    x_xy_plot = X_xy - x_origin_E
    x_xz_plot = X_xz - x_origin_E
    x_yz_plot = x_yz - x_origin_E

    sim_uz_xy = sim_vel_xy[:, 2].reshape(X_xy.shape)
    sim_uz_yz = sim_vel_yz[:, 2].reshape(Y_yz.shape)
    sim_uz_xz = sim_vel_xz[:, 2].reshape(X_xz.shape)

    model_xy = _model_wbar_grid(
        x_rel_m=X_xy - pos_E[0],
        y_rel_m=Y_xy - pos_E[1],
        z_rel_m=np.full_like(X_xy, z_xy - pos_E[2]),
    )
    model_yz = _model_wbar_grid(
        x_rel_m=np.full_like(Y_yz, x_yz - pos_E[0]),
        y_rel_m=Y_yz - pos_E[1],
        z_rel_m=Z_yz - pos_E[2],
    )
    model_xz = _model_wbar_grid(
        x_rel_m=X_xz - pos_E[0],
        y_rel_m=np.full_like(X_xz, y_xz - pos_E[1]),
        z_rel_m=Z_xz - pos_E[2],
    )

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 12.0), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.015,
        h_pad=0.035,
        wspace=0.025,
        hspace=0.04,
    )

    _plot_pair(
        axes=axes,
        row=0,
        x_plot=x_xy_plot,
        y_plot=Y_xy,
        sim_grid=sim_uz_xy,
        model_grid=model_xy,
        title_left=f"Simulation: $u_z$ in XY plane (Z/B=0, step {int(data['step'][0])})",
        title_right="Reduced model: $\\bar W(X,Y,0)$",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="y (m)",
    )
    _plot_pair(
        axes=axes,
        row=1,
        x_plot=Y_yz,
        y_plot=Z_yz,
        sim_grid=sim_uz_yz,
        model_grid=model_yz,
        title_left=f"Simulation: $u_z$ in YZ plane (X/B={x_yz_plot / SPAN_M:.1f})",
        title_right=f"Reduced model: $\\bar W({x_yz_plot / SPAN_M:.1f},Y,Z)$",
        xlabel="y (m)",
        ylabel="z (m)",
    )
    _plot_pair(
        axes=axes,
        row=2,
        x_plot=x_xz_plot,
        y_plot=Z_xz,
        sim_grid=sim_uz_xz,
        model_grid=model_xz,
        title_left="Simulation: $u_z$ in XZ plane (Y/B=0)",
        title_right="Reduced model: $\\bar W(X,0,Z)$",
        xlabel="$x - x_\\mathrm{wing}$ (m)",
        ylabel="z (m)",
    )

    for col in range(2):
        _draw_wing_projection(
            ax_xy=axes[0, col],
            ax_yz=axes[1, col],
            ax_xz=axes[2, col],
            R_pas_E_to_BP1=R_pas_E_to_BP1,
            pos_E=pos_E,
            x_origin_E=x_origin_E,
        )

    fig.suptitle(
        "Single-Wing Wake: Free-Wake Simulation vs Reduced Wbar Model\n"
        "The model column is the scalar lift-weighted upwash closure from "
        "math_models, not a reconstructed velocity-vector field.",
        y=1.01,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a single-wing wake slice figure comparing simulated u_z to the "
            "reduced Wbar model from math_models."
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
