"""Plot simulation AOA slices with one AOA-independent analytical reference.

The analytical panel is intentionally shown only once.  The lightweight model used here
does not include an angle-of-attack-dependent circulation law; it uses one calibrated
tip-vortex strength and one single-wing normalization.  Repeating it for AOA 5, 10,
and 15 deg would imply physics that the model does not currently contain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "streamwise_stability_energy"
    / "curated_fixed_wing_dataset"
)
DEFAULT_INPUT_CSV = DEFAULT_OUTPUT_DIR / "curated_streamwise_summary.csv"
DEFAULT_OUTPUT_FIGURE = (
    DEFAULT_OUTPUT_DIR / "figures" / "sim_by_aoa_single_analytic_reference.png"
)

# Geometry and operating constants for this sweep family.
SPAN_M = 1.0
SEMI_SPAN_M = SPAN_M / 2.0
ROOT_CHORD_M = 0.1
AIR_DENSITY_KG_M3 = 1.225
REFERENCE_SPEED_MPS = 1.0
WEIGHT_N = 0.0235

# Calibrated once against AOA=5, Z/B=0, X/B<=5 to avoid per-panel overfitting.
TIP_VORTEX_GAMMA_SCALE = 0.665

# Keep X/B=7 now that the long convergence reruns are in the curated dataset.
X_MAX_OVER_SPAN = 7.0
SIMULATION_CONDITIONS: tuple[tuple[float, float, str], ...] = (
    (5.0, 0.0, "Simulation: AOA 5 deg, Z/B = 0"),
    (5.0, 0.5, "Simulation: AOA 5 deg, Z/B = 0.5"),
    (10.0, 0.0, "Simulation: AOA 10 deg, Z/B = 0"),
    (15.0, 0.0, "Simulation: AOA 15 deg, Z/B = 0"),
)
ANALYTIC_REFERENCE_AOA_DEG = 5.0
ANALYTIC_REFERENCE_Z_OVER_SPAN = 0.0


def _apply_constant_style() -> None:
    """Apply one global figure style for consistent output."""
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.facecolor": "#f7f7f7",
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.6,
            "axes.grid": True,
        }
    )


def _grid_edges(values: np.ndarray) -> np.ndarray:
    """Return pcolormesh edges from sorted nonuniform center coordinates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a 1D array with at least two elements.")
    mids = 0.5 * (values[1:] + values[:-1])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], mids, [last]))


def _tip_vortex_vertical_velocity(
    x_m: float,
    y_m: float,
    z_m: float,
    gamma_tip_m2_s: float,
) -> float:
    """Return induced vertical velocity from two semi-infinite tip vortices."""
    velocity_z_mps = 0.0
    # Circulation signs chosen to produce downwash near center and upwash outboard.
    for y0_m, gamma_m2_s in (
        (+SEMI_SPAN_M, -gamma_tip_m2_s),
        (-SEMI_SPAN_M, +gamma_tip_m2_s),
    ):
        dy_m = y_m - y0_m
        dz_m = z_m
        radial_sq_m2 = dy_m * dy_m + dz_m * dz_m
        if radial_sq_m2 < 1.0e-12:
            continue
        finite_segment_factor = (1.0 + x_m / np.sqrt(x_m * x_m + radial_sq_m2)) / (
            4.0 * np.pi
        )
        velocity_z_mps += -gamma_m2_s * dy_m / radial_sq_m2 * finite_segment_factor
    return float(velocity_z_mps)


def _lift_weighted_wbar_mps(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    gamma_tip_m2_s: float,
) -> float:
    """Compute lift-weighted Wbar over the receiver span."""
    xi_m = np.linspace(-SEMI_SPAN_M, +SEMI_SPAN_M, 241)
    weights = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * xi_m / SPAN_M) ** 2))
    x_m = x_over_span * SPAN_M
    y_m = y_over_span * SPAN_M
    z_m = z_over_span * SPAN_M
    w_samples = np.array(
        [
            _tip_vortex_vertical_velocity(
                x_m=x_m,
                y_m=y_m + this_xi_m,
                z_m=z_m,
                gamma_tip_m2_s=gamma_tip_m2_s,
            )
            for this_xi_m in xi_m
        ],
        dtype=float,
    )
    return float(np.average(w_samples, weights=weights))


def _predict_rear_ratio_from_analytic_wbar(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    single_body_thrust_n: float,
    gamma_tip_m2_s: float,
) -> float:
    """Predict rear required-thrust ratio from the analytical Wbar model."""
    if single_body_thrust_n <= 0.0:
        return float("nan")
    wbar_mps = _lift_weighted_wbar_mps(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        gamma_tip_m2_s=gamma_tip_m2_s,
    )
    delta_thrust_n = -(WEIGHT_N / REFERENCE_SPEED_MPS) * wbar_mps
    predicted_rear_thrust_n = single_body_thrust_n + delta_thrust_n
    return float(predicted_rear_thrust_n / single_body_thrust_n)


def _condition_mask(data: np.ndarray, aoa_deg: float, z_over_span: float) -> np.ndarray:
    """Return the mask for one simulation condition."""
    return (
        np.isclose(data["aoa_deg"], aoa_deg)
        & np.isclose(data["zB"], z_over_span)
        & (data["xB"] <= X_MAX_OVER_SPAN)
    )


def _simulation_grid(
    data: np.ndarray,
    aoa_deg: float,
    z_over_span: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, and rear-thrust-ratio simulation grid for one condition."""
    subset_indices = np.where(_condition_mask(data, aoa_deg, z_over_span))[0]
    if subset_indices.size == 0:
        raise RuntimeError(f"No data found for AOA={aoa_deg}, Z/B={z_over_span}.")

    x_values = np.array(sorted(np.unique(data["xB"][subset_indices])), dtype=float)
    y_values = np.array(sorted(np.unique(data["yB"][subset_indices])), dtype=float)
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    y_index = {value: idx for idx, value in enumerate(y_values)}
    x_index = {value: idx for idx, value in enumerate(x_values)}

    for row_index in subset_indices:
        this_y = y_index[float(data["yB"][row_index])]
        this_x = x_index[float(data["xB"][row_index])]
        grid[this_y, this_x] = float(data["rear_ratio"][row_index])

    return x_values, y_values, grid


def _analytic_reference_grid(
    data: np.ndarray,
    gamma_tip_m2_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a single AOA-independent analytical reference grid."""
    subset_indices = np.where(
        _condition_mask(
            data,
            ANALYTIC_REFERENCE_AOA_DEG,
            ANALYTIC_REFERENCE_Z_OVER_SPAN,
        )
    )[0]
    if subset_indices.size == 0:
        raise RuntimeError("No data found for the analytical reference grid.")

    x_values = np.array(sorted(np.unique(data["xB"][subset_indices])), dtype=float)
    y_values = np.array(sorted(np.unique(data["yB"][subset_indices])), dtype=float)
    single_body_thrust_n = float(np.mean(data["single_body_thrust_N"][subset_indices]))
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)

    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            grid[y_index, x_index] = _predict_rear_ratio_from_analytic_wbar(
                x_over_span=float(x_over_span),
                y_over_span=float(y_over_span),
                z_over_span=ANALYTIC_REFERENCE_Z_OVER_SPAN,
                single_body_thrust_n=single_body_thrust_n,
                gamma_tip_m2_s=gamma_tip_m2_s,
            )

    return x_values, y_values, grid


def _draw_source_wing_outline(ax: plt.Axes) -> None:
    """Draw source-wing footprint in normalized coordinates."""
    wing_x = np.array([0.0, ROOT_CHORD_M / SPAN_M, ROOT_CHORD_M / SPAN_M, 0.0, 0.0])
    wing_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
    ax.plot(wing_x, wing_y, color="black", linewidth=1.0, alpha=0.75)


def _plot_grid(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    color_norm: TwoSlopeNorm,
    title: str,
) -> mpl.collections.QuadMesh:
    """Plot one grid and source-wing outline."""
    x_edges = _grid_edges(x_values)
    y_edges = _grid_edges(y_values)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        shading="auto",
        cmap="RdBu_r",
        norm=color_norm,
    )
    _draw_source_wing_outline(ax)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_title(title)
    ax.set_xlabel("X/B")
    ax.set_ylabel("Y/B")
    ax.grid(False)
    return mesh


def build_figure(input_csv: Path, output_figure: Path) -> None:
    """Build and save the simulation-plus-one-analytic-reference figure."""
    _apply_constant_style()
    data = np.genfromtxt(
        input_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )

    gamma_base_m2_s = WEIGHT_N / (AIR_DENSITY_KG_M3 * REFERENCE_SPEED_MPS * SPAN_M)
    gamma_tip_m2_s = TIP_VORTEX_GAMMA_SCALE * gamma_base_m2_s

    sim_payload = []
    all_values: list[np.ndarray] = []
    for aoa_deg, z_over_span, label in SIMULATION_CONDITIONS:
        x_values, y_values, grid = _simulation_grid(data, aoa_deg, z_over_span)
        sim_payload.append((label, x_values, y_values, grid))
        all_values.append(grid[np.isfinite(grid)])

    analytic_x, analytic_y, analytic_grid = _analytic_reference_grid(
        data=data,
        gamma_tip_m2_s=gamma_tip_m2_s,
    )
    all_values.append(analytic_grid[np.isfinite(analytic_grid)])
    finite_values = np.concatenate(all_values)
    ratio_dev = np.max(np.abs(finite_values - 1.0))
    ratio_half_range = max(0.05, float(ratio_dev))
    color_norm = TwoSlopeNorm(
        vmin=1.0 - ratio_half_range,
        vcenter=1.0,
        vmax=1.0 + ratio_half_range,
    )

    fig = plt.figure(figsize=(13.5, 15.0), constrained_layout=True)
    gridspec = fig.add_gridspec(4, 2, width_ratios=(1.0, 1.05))
    mesh = None
    for row_index, (label, x_values, y_values, grid) in enumerate(sim_payload):
        ax = fig.add_subplot(gridspec[row_index, 0])
        mesh = _plot_grid(ax, x_values, y_values, grid, color_norm, label)

    analytic_ax = fig.add_subplot(gridspec[:, 1])
    mesh = _plot_grid(
        analytic_ax,
        analytic_x,
        analytic_y,
        analytic_grid,
        color_norm,
        ("Single analytical reference\n" "AOA-independent tip-vortex model, Z/B = 0"),
    )
    analytic_ax.text(
        0.02,
        0.02,
        "Shown once because Gamma is not AOA-dependent in this model.",
        transform=analytic_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )

    cbar = fig.colorbar(mesh, ax=fig.axes, fraction=0.025, pad=0.015)
    cbar.set_label("Normalized Required Rear Thrust  $T_{rear}/T_{single}$")
    fig.suptitle(
        "Simulation AOA Slices with One Analytical Reference",
        y=0.998,
    )
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot simulation AOA slices with one analytical reference."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to curated streamwise summary CSV.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_OUTPUT_FIGURE,
        help="Path to output figure file.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    build_figure(input_csv=args.input_csv, output_figure=args.output_figure)
    print(args.output_figure)


if __name__ == "__main__":
    main()
