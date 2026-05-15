"""Plot simulation AOA slices with full and wake-only analytical references.

The analytical panels use the in-repo constant-circulation horseshoe/tip-vortex
model implemented in ``horseshoe_tip_vortex_stability.py`` and convert power
change to thrust change with ``Delta T = Delta P / U``.

The comparison figure plots the same quantity in all panels:
``T_rear / T_single``.  The analytical panel uses the matching single-wing
baseline thrust from the curated simulation table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

try:
    from examples import horseshoe_tip_vortex_stability as horseshoe_model
except ImportError:  # pragma: no cover - supports direct script execution
    import horseshoe_tip_vortex_stability as horseshoe_model

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
ANALYTIC_PARAMS = horseshoe_model.HorseshoeWakeParams(
    span_m=SPAN_M,
    chord_m=ROOT_CHORD_M,
    aoa_deg=ANALYTIC_REFERENCE_AOA_DEG,
)


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


def analytical_wbar_model_mps(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    wake_only: bool = False,
) -> float:
    """Evaluate analytical lift-weighted upwash."""
    wbar_function = (
        horseshoe_model.lift_weighted_tip_pair_wbar_mps
        if wake_only
        else horseshoe_model.lift_weighted_wbar_mps
    )
    return float(
        wbar_function(
            x_over_span * SPAN_M,
            y_over_span * SPAN_M,
            z_over_span * SPAN_M,
            ANALYTIC_PARAMS,
        )
    )


def analytical_delta_thrust_n(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    wake_only: bool = False,
) -> float:
    """Return analytical interaction thrust correction from Wbar."""
    wbar_mps = analytical_wbar_model_mps(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        wake_only=wake_only,
    )
    return float(horseshoe_model.thrust_change_n(wbar_mps, ANALYTIC_PARAMS))


def analytical_rear_thrust_ratio(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    baseline_thrust_n: float | None = None,
    wake_only: bool = False,
) -> float:
    """Return analytical required-thrust ratio using the selected baseline."""
    if baseline_thrust_n is None:
        baseline_thrust_n = horseshoe_model.baseline_thrust_n(ANALYTIC_PARAMS)
    delta_thrust_n = analytical_delta_thrust_n(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
        wake_only=wake_only,
    )
    return float((baseline_thrust_n + delta_thrust_n) / baseline_thrust_n)


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
    *,
    wake_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an AOA-independent analytical reference grid."""
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
    baseline_thrust_n = float(np.mean(data["single_body_thrust_N"][subset_indices]))
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)

    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            grid[y_index, x_index] = analytical_rear_thrust_ratio(
                x_over_span=float(x_over_span),
                y_over_span=float(y_over_span),
                z_over_span=ANALYTIC_REFERENCE_Z_OVER_SPAN,
                baseline_thrust_n=baseline_thrust_n,
                wake_only=wake_only,
            )

    return x_values, y_values, grid


def analytical_rear_thrust_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    z_over_span: float,
    baseline_thrust_n: float | None = None,
    wake_only: bool = False,
) -> np.ndarray:
    """Return analytical rear required thrust on an X/B-Y/B grid."""
    if baseline_thrust_n is None:
        baseline_thrust_n = horseshoe_model.baseline_thrust_n(ANALYTIC_PARAMS)
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            grid[y_index, x_index] = baseline_thrust_n + analytical_delta_thrust_n(
                x_over_span=float(x_over_span),
                y_over_span=float(y_over_span),
                z_over_span=float(z_over_span),
                wake_only=wake_only,
            )
    return grid


def analytical_rear_dthrust_dxb_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    z_over_span: float,
    wake_only: bool = False,
) -> np.ndarray:
    """Return analytical rear dT/d(X/B)."""
    gradient_function = (
        horseshoe_model.lift_weighted_tip_pair_thrust_gradient_per_x_over_span_n
        if wake_only
        else horseshoe_model.lift_weighted_thrust_gradient_per_x_over_span_n
    )
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            grid[y_index, x_index] = gradient_function(
                float(x_over_span) * SPAN_M,
                float(y_over_span) * SPAN_M,
                float(z_over_span) * SPAN_M,
                ANALYTIC_PARAMS,
            )
    return grid


def _ratio_cbar_ticks(norm: TwoSlopeNorm) -> list[float]:
    """Return ratio colorbar ticks with explicit sub-unity marks."""
    dense_under_one = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    high_step = 0.5 if norm.vmax > 2.5 else 0.25
    above_one = np.arange(1.0 + high_step, norm.vmax + 0.5 * high_step, high_step)
    ticks = np.unique(np.concatenate((dense_under_one, above_one)))
    return [float(tick) for tick in ticks if norm.vmin <= tick <= norm.vmax]


def analytical_model_parameters() -> dict:
    """Return the parameter choices used by the analytical reference model."""
    parameters = horseshoe_model.model_parameters_dict(ANALYTIC_PARAMS)
    parameters.update(
        {
            "source_file": "examples/horseshoe_tip_vortex_stability.py",
            "baseline_thrust_choice": (
                "Analytical comparison ratios use the matching simulation "
                "single-wing baseline from the curated CSV."
            ),
            "wake_model": (
                "constant-circulation full horseshoe vortex plus tip-vortex "
                "point-receiver stability boundary"
            ),
        }
    )
    parameters.update(
        {
            "power_to_thrust_relation": "Delta T = Delta P/U = -(L/U) * Wbar",
            "plotted_analytical_quantity": (
                "T_rear/T_single using the same single-wing normalization as "
                "the simulation panels"
            ),
            "analytical_reference_panels": (
                "full horseshoe (bound segment plus trailing wake) and wake-only "
                "trailing tip-vortex pair"
            ),
            "analytic_reference_aoa_deg": ANALYTIC_REFERENCE_AOA_DEG,
            "analytic_reference_z_over_span": ANALYTIC_REFERENCE_Z_OVER_SPAN,
        }
    )
    return parameters


def _draw_source_wing_outline(ax: plt.Axes) -> None:
    """Draw source-wing footprint in normalized coordinates."""
    wing_x = np.array([0.0, ROOT_CHORD_M / SPAN_M, ROOT_CHORD_M / SPAN_M, 0.0, 0.0])
    wing_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
    ax.fill(wing_x, wing_y, color="black", alpha=0.94, zorder=8)


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
    ax.set_xlim(min(-0.02, x_edges[0]), x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_title(title)
    ax.set_xlabel("X/B")
    ax.set_ylabel("Y/B")
    ax.grid(False)
    return mesh


def _ratio_norm_from_grid(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local diverging norm centered at the single-wing value."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        return TwoSlopeNorm(vmin=0.95, vcenter=1.0, vmax=1.05)
    half_range = max(0.05, float(np.nanmax(np.abs(finite_values - 1.0))))
    vmin = max(0.0, 1.0 - half_range)
    return TwoSlopeNorm(
        vmin=vmin,
        vcenter=1.0,
        vmax=1.0 + half_range,
    )


def build_figure(input_csv: Path, output_figure: Path) -> None:
    """Build and save the simulation plus analytical reference figure."""
    _apply_constant_style()
    data = np.genfromtxt(
        input_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )

    sim_payload = []
    for aoa_deg, z_over_span, label in SIMULATION_CONDITIONS:
        x_values, y_values, grid = _simulation_grid(data, aoa_deg, z_over_span)
        sim_payload.append((label, x_values, y_values, grid))

    analytic_x, analytic_y, analytic_full_grid = _analytic_reference_grid(
        data=data,
        wake_only=False,
    )
    _, _, analytic_wake_grid = _analytic_reference_grid(
        data=data,
        wake_only=True,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.5, 13.6),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.07, w_pad=0.025, hspace=0.035)
    flat_axes = axes.ravel()
    for row_index, (label, x_values, y_values, grid) in enumerate(sim_payload):
        ax = flat_axes[row_index]
        mesh = _plot_grid(
            ax,
            x_values,
            y_values,
            grid,
            _ratio_norm_from_grid(grid),
            label,
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_ticks(_ratio_cbar_ticks(mesh.norm))
        cbar.set_label("$T_{rear}/T_{single}$")

    analytic_ax = flat_axes[len(sim_payload)]
    mesh = _plot_grid(
        analytic_ax,
        analytic_x,
        analytic_y,
        analytic_full_grid,
        _ratio_norm_from_grid(analytic_full_grid),
        ("Analytical full horseshoe\n" "bound segment + trailing wake, Z/B = 0"),
    )
    cbar = fig.colorbar(mesh, ax=analytic_ax, fraction=0.046, pad=0.025)
    cbar.set_ticks(_ratio_cbar_ticks(mesh.norm))
    cbar.set_label("$T_{rear}/T_{single}$")
    analytic_ax.text(
        0.02,
        0.02,
        "Reference AOA 5 deg.",
        transform=analytic_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )

    wake_ax = flat_axes[len(sim_payload) + 1]
    mesh = _plot_grid(
        wake_ax,
        analytic_x,
        analytic_y,
        analytic_wake_grid,
        _ratio_norm_from_grid(analytic_wake_grid),
        ("Analytical wake-only\n" "trailing tip-vortex pair, Z/B = 0"),
    )
    cbar = fig.colorbar(mesh, ax=wake_ax, fraction=0.046, pad=0.025)
    cbar.set_ticks(_ratio_cbar_ticks(mesh.norm))
    cbar.set_label("$T_{rear}/T_{single}$")

    fig.suptitle(
        "Rear-Wing Required Thrust Ratio: Simulation, Full Horseshoe, and Wake-Only",
        y=1.012,
    )
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot simulation AOA slices with full and wake-only references."
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
