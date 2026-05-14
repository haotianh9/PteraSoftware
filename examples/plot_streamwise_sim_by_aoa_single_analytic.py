"""Plot simulation AOA slices with one AOA-independent analytical reference.

The analytical panel is intentionally shown only once. The lightweight model used here
does not include an angle-of-attack-dependent circulation law. It mirrors the
``Wbar_model`` parameters in
``/home/hht/Dropbox/Research/PostDoc_IRPHE/Code/Bird_flock/math_models/``
and converts power change to thrust change with ``Delta T = Delta P / U``.
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

# Analytical reference parameters mirrored from lifting_line_streamwise_stability.py.
ANALYTIC_MODEL_SOURCE = (
    "/home/hht/Dropbox/Research/PostDoc_IRPHE/Code/Bird_flock/math_models/"
    "lifting_line_streamwise_stability.py"
)
ANALYTIC_MASS_KG = 1.0
ANALYTIC_GRAVITY_MPS2 = 9.81
ANALYTIC_LIFT_N = ANALYTIC_MASS_KG * ANALYTIC_GRAVITY_MPS2
ANALYTIC_REFERENCE_SPEED_MPS = 1.0
ANALYTIC_DB_PRIME_N_PER_MPS = 1.0
WAKE_UPWASH_AMPLITUDE_OVER_U = 0.15
WAKE_X_GATE_OVER_SPAN = 0.25
WAKE_X_RISE_OVER_SPAN = 1.0
WAKE_X_DECAY_OVER_SPAN = 8.0
WAKE_Y_LOBE_OVER_SPAN = 0.65
WAKE_Y_WIDTH_OVER_SPAN = 0.25
WAKE_Z_WIDTH_OVER_SPAN = 0.35

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


def analytical_wbar_model_mps(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
) -> float:
    """Evaluate the analytical lift-weighted upwash model from math_models."""
    x_m = x_over_span * SPAN_M
    y_m = y_over_span * SPAN_M
    z_m = z_over_span * SPAN_M
    upwash_amplitude_mps = WAKE_UPWASH_AMPLITUDE_OVER_U * ANALYTIC_REFERENCE_SPEED_MPS
    x_gate_m = WAKE_X_GATE_OVER_SPAN * SPAN_M
    x_rise_m = WAKE_X_RISE_OVER_SPAN * SPAN_M
    x_decay_m = WAKE_X_DECAY_OVER_SPAN * SPAN_M
    y_lobe_m = WAKE_Y_LOBE_OVER_SPAN * SPAN_M
    y_width_m = WAKE_Y_WIDTH_OVER_SPAN * SPAN_M
    z_width_m = WAKE_Z_WIDTH_OVER_SPAN * SPAN_M

    gate = 1.0 / (1.0 + np.exp(-x_m / x_gate_m))
    downstream_x_m = max(x_m, 0.0)
    streamwise_shape = (1.0 - np.exp(-downstream_x_m / x_rise_m)) * np.exp(
        -downstream_x_m / x_decay_m
    )
    lateral_shape = np.exp(-(((abs(y_m) - y_lobe_m) / y_width_m) ** 2))
    vertical_shape = np.exp(-((z_m / z_width_m) ** 2))
    return float(
        upwash_amplitude_mps * gate * streamwise_shape * lateral_shape * vertical_shape
    )


def _predict_rear_ratio_from_analytic_wbar(
    x_over_span: float,
    y_over_span: float,
    z_over_span: float,
    single_body_thrust_n: float,
) -> float:
    """Predict rear required-thrust ratio from the analytical Wbar model."""
    if single_body_thrust_n <= 0.0:
        return float("nan")
    wbar_mps = analytical_wbar_model_mps(
        x_over_span=x_over_span,
        y_over_span=y_over_span,
        z_over_span=z_over_span,
    )
    delta_thrust_n = -(ANALYTIC_LIFT_N / ANALYTIC_REFERENCE_SPEED_MPS) * wbar_mps
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
            )

    return x_values, y_values, grid


def analytical_rear_thrust_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    z_over_span: float,
    single_body_thrust_n: float,
) -> np.ndarray:
    """Return analytical rear required-thrust values on an X/B-Y/B grid."""
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            wbar_mps = analytical_wbar_model_mps(
                x_over_span=float(x_over_span),
                y_over_span=float(y_over_span),
                z_over_span=float(z_over_span),
            )
            delta_thrust_n = (
                -(ANALYTIC_LIFT_N / ANALYTIC_REFERENCE_SPEED_MPS) * wbar_mps
            )
            grid[y_index, x_index] = single_body_thrust_n + delta_thrust_n
    return grid


def analytical_model_parameters(single_body_thrust_n: float | None = None) -> dict:
    """Return the parameter choices used by the analytical reference model."""
    parameters = {
        "span_m": SPAN_M,
        "semispan_m": SEMI_SPAN_M,
        "root_chord_m": ROOT_CHORD_M,
        "source_file": ANALYTIC_MODEL_SOURCE,
        "mass_kg": ANALYTIC_MASS_KG,
        "gravity_mps2": ANALYTIC_GRAVITY_MPS2,
        "lift_n": ANALYTIC_LIFT_N,
        "reference_speed_mps": ANALYTIC_REFERENCE_SPEED_MPS,
        "db_prime_n_per_mps": ANALYTIC_DB_PRIME_N_PER_MPS,
        "w0_over_u": WAKE_UPWASH_AMPLITUDE_OVER_U,
        "x_gate_over_span": WAKE_X_GATE_OVER_SPAN,
        "x_rise_over_span": WAKE_X_RISE_OVER_SPAN,
        "x_decay_over_span": WAKE_X_DECAY_OVER_SPAN,
        "y_lobe_over_span": WAKE_Y_LOBE_OVER_SPAN,
        "y_width_over_span": WAKE_Y_WIDTH_OVER_SPAN,
        "z_width_over_span": WAKE_Z_WIDTH_OVER_SPAN,
        "wake_model": (
            "direct lift-weighted Wbar_model: downstream logistic gate, "
            "streamwise rise/decay, symmetric lateral upwash lobes, vertical decay"
        ),
        "power_to_thrust_relation": "Delta T = Delta P/U = -(L/U) * Wbar",
        "analytic_reference_aoa_deg": ANALYTIC_REFERENCE_AOA_DEG,
        "analytic_reference_z_over_span": ANALYTIC_REFERENCE_Z_OVER_SPAN,
    }
    if single_body_thrust_n is not None:
        parameters["single_body_thrust_n"] = float(single_body_thrust_n)
    return parameters


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


def _ratio_norm_from_grid(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local diverging norm centered at the single-wing value."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        return TwoSlopeNorm(vmin=0.95, vcenter=1.0, vmax=1.05)
    ratio_half_range = max(0.05, float(np.nanmax(np.abs(finite_values - 1.0))))
    return TwoSlopeNorm(
        vmin=1.0 - ratio_half_range,
        vcenter=1.0,
        vmax=1.0 + ratio_half_range,
    )


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

    sim_payload = []
    for aoa_deg, z_over_span, label in SIMULATION_CONDITIONS:
        x_values, y_values, grid = _simulation_grid(data, aoa_deg, z_over_span)
        sim_payload.append((label, x_values, y_values, grid))

    analytic_x, analytic_y, analytic_grid = _analytic_reference_grid(data=data)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.5, 13.0),
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for row_index, (label, x_values, y_values, grid) in enumerate(sim_payload):
        ax = flat_axes[row_index]
        mesh = _plot_grid(
            ax, x_values, y_values, grid, _ratio_norm_from_grid(grid), label
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label("$T_{rear}/T_{single}$")

    analytic_ax = flat_axes[len(sim_payload)]
    mesh = _plot_grid(
        analytic_ax,
        analytic_x,
        analytic_y,
        analytic_grid,
        _ratio_norm_from_grid(analytic_grid),
        ("Single analytical reference\n" "AOA-independent Wbar model, Z/B = 0"),
    )
    cbar = fig.colorbar(mesh, ax=analytic_ax, fraction=0.046, pad=0.025)
    cbar.set_label("$T_{rear}/T_{single}$")
    analytic_ax.text(
        0.02,
        0.02,
        "Shown once because Wbar is not AOA-dependent in this model.",
        transform=analytic_ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )
    flat_axes[-1].axis("off")

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
