"""Plot two-flyer fixed-wing load deltas against a single-flyer baseline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    ROOT / "output" / "free_flight_cases" / "streamwise_stability_energy"
)
DEFAULT_INPUT_CSV = (
    DEFAULT_RESULTS_ROOT
    / "curated_fixed_wing_dataset"
    / "curated_streamwise_summary.csv"
)
DEFAULT_OUTPUT_FIGURE_DIR = (
    DEFAULT_RESULTS_ROOT / "curated_fixed_wing_dataset" / "figures"
)
DEFAULT_OUTPUT_DATA_CSV = (
    DEFAULT_RESULTS_ROOT
    / "curated_fixed_wing_dataset"
    / "two_flyer_load_deltas_vs_single.csv"
)

SPAN_M = 1.0
CHORD_M = 0.1
DEFAULT_FINAL_WINDOW_S = 2.0

CONDITIONS: tuple[tuple[float, float, str], ...] = (
    (5.0, 0.0, "AOA 5 deg, Z/B = 0"),
    (5.0, 0.5, "AOA 5 deg, Z/B = 0.5"),
    (5.0, 1.0, "AOA 5 deg, Z/B = 1"),
    (10.0, 0.0, "AOA 10 deg, Z/B = 0"),
    (15.0, 0.0, "AOA 15 deg, Z/B = 0"),
)

PREFERRED_FAR_LATERAL_BASELINE_ROOT_BY_AOA = {
    5.0: (
        "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_Z0-0p5-1_"
        "X0p25-5_Y0-1p5_t9s_corrected_translating_dt12_parallel10"
    ),
    10.0: "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA10_Z0_X0p25-9_Y0-1p5_t9s_dt12",
    15.0: "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA15_Z0_X0p25-9_Y0-1p5_t9s_dt12",
}


@dataclass(frozen=True)
class ComponentSpec:
    """One load component to map."""

    name: str
    array_name: str
    axis: int
    units: str
    output_suffix: str


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("Fy", "force", 1, "N", "Fy"),
    ComponentSpec("Fz", "force", 2, "N", "Fz"),
    ComponentSpec("roll", "moment", 0, "N m", "roll"),
    ComponentSpec("yaw", "moment", 1, "N m", "yaw"),
    ComponentSpec("pitch", "moment", 2, "N m", "pitch"),
)


@dataclass(frozen=True)
class LoadMeans:
    """Final-window mean clamp forces and moments."""

    forces_E_N: np.ndarray
    moments_E_Cg_Nm: np.ndarray
    positions_E_m: np.ndarray | None = None


@dataclass(frozen=True)
class Baseline:
    """Single-flyer baseline load for one AOA."""

    aoa_deg: float
    loads: LoadMeans
    source: str


def _style() -> None:
    """Apply the fixed-wing dataset plotting style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.facecolor": "#f8f8f8",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d2d2d2",
            "grid.linewidth": 0.55,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""
    return json.loads(path.read_text())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read curated CSV rows."""
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _history_array(
    history: np.lib.npyio.NpzFile,
    candidates: tuple[str, ...],
) -> np.ndarray:
    """Return the first matching array from a history file."""
    for key in candidates:
        if key in history.files:
            return np.asarray(history[key], dtype=float)
    raise KeyError(f"None of {candidates} were found in {history.files}.")


def _final_window_num_steps(
    summary: dict[str, Any],
    values_len: int,
    fallback_window_s: float = DEFAULT_FINAL_WINDOW_S,
) -> int:
    """Return the number of samples to use for the final-window average."""
    time_step_s = summary.get("time_step_s", summary.get("delta_time_s"))
    window_s = summary.get("final_average_window_s", fallback_window_s)
    if time_step_s is None or float(time_step_s) <= 0.0:
        return values_len
    return min(values_len, max(1, int(round(float(window_s) / float(time_step_s)))))


def _load_final_window_means(history_path: Path) -> LoadMeans:
    """Load final-window mean forces and moments from a history file."""
    summary_path = history_path.with_name("summary.json")
    summary = _read_json(summary_path) if summary_path.exists() else {}
    with np.load(history_path) as history:
        forces = _history_array(history, ("clamp_forces_E_N", "clamp_force_E_N"))
        moments = _history_array(
            history,
            ("clamp_moments_E_Cg_Nm", "clamp_moment_E_Cg_Nm"),
        )
        positions = (
            np.asarray(history["positions_E_E_m"][0], dtype=float)
            if "positions_E_E_m" in history.files
            else None
        )

    if forces.ndim == 2:
        forces = forces[:, None, :]
    if moments.ndim == 2:
        moments = moments[:, None, :]
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(
            f"Expected force history shape (n, bodies, 3), got {forces.shape}."
        )
    if moments.ndim != 3 or moments.shape[2] != 3:
        raise ValueError(
            f"Expected moment history shape (n, bodies, 3), got {moments.shape}."
        )
    if forces.shape[:2] != moments.shape[:2]:
        raise ValueError(
            "Force and moment histories must have matching sample/body dimensions."
        )

    steps = _final_window_num_steps(summary, values_len=forces.shape[0])
    return LoadMeans(
        forces_E_N=np.mean(forces[-steps:], axis=0),
        moments_E_Cg_Nm=np.mean(moments[-steps:], axis=0),
        positions_E_m=positions,
    )


def _single_wing_baseline_path(results_root: Path, aoa_deg: float) -> Path | None:
    """Return the true single-wing history path when available."""
    if not np.isclose(aoa_deg, 5.0):
        return None
    path = (
        results_root
        / "raw_sources"
        / "single_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_t9s_corrected_translating_dt12"
        / "single_wing_history.npz"
    )
    return path if path.exists() else None


def _far_lateral_baseline_path(results_root: Path, aoa_deg: float) -> Path:
    """Return the preferred far-lateral baseline history for one AOA."""
    root_name = PREFERRED_FAR_LATERAL_BASELINE_ROOT_BY_AOA.get(round(float(aoa_deg), 6))
    if root_name is not None:
        path = (
            results_root
            / "raw_sources"
            / root_name
            / "baseline_far_lateral"
            / "history.npz"
        )
        if path.exists():
            return path

    candidates = []
    baseline_summaries = (results_root / "raw_sources").glob(
        "*/baseline_far_lateral/summary.json"
    )
    for summary_path in sorted(baseline_summaries):
        summary = _read_json(summary_path)
        if np.isclose(float(summary.get("angle_of_attack_deg", np.nan)), aoa_deg):
            candidates.append(summary_path.with_name("history.npz"))
    if not candidates:
        raise FileNotFoundError(f"No far-lateral baseline found for AOA={aoa_deg}.")
    return candidates[0]


def _single_body_load_from_history(loads: LoadMeans) -> LoadMeans:
    """Collapse a single or far-lateral baseline history to one body load."""
    if loads.forces_E_N.shape[0] == 1:
        return loads
    return LoadMeans(
        forces_E_N=np.mean(loads.forces_E_N, axis=0, keepdims=True),
        moments_E_Cg_Nm=np.mean(loads.moments_E_Cg_Nm, axis=0, keepdims=True),
        positions_E_m=None,
    )


def _load_baselines(
    results_root: Path,
    aoa_values: set[float],
    prefer_true_single: bool,
) -> dict[float, Baseline]:
    """Load one single-body baseline for each AOA."""
    baselines: dict[float, Baseline] = {}
    for aoa_deg in sorted(aoa_values):
        true_single_path = (
            _single_wing_baseline_path(results_root, aoa_deg)
            if prefer_true_single
            else None
        )
        if true_single_path is not None:
            path = true_single_path
            source = str(path.relative_to(results_root))
        else:
            path = _far_lateral_baseline_path(results_root, aoa_deg)
            source = str(path.relative_to(results_root)) + " (body-mean far lateral)"
        loads = _single_body_load_from_history(_load_final_window_means(path))
        baselines[aoa_deg] = Baseline(aoa_deg=aoa_deg, loads=loads, source=source)
    return baselines


def _body_indices_from_positions(positions_E_m: np.ndarray | None) -> tuple[int, int]:
    """Return fore and rear body indices from initial positions."""
    if positions_E_m is None or positions_E_m.shape[0] < 2:
        raise ValueError("Two-body history does not contain usable body positions.")
    x_positions = np.asarray(positions_E_m[:, 0], dtype=float)
    fore_index = int(np.argmax(x_positions))
    rear_index = int(np.argmin(x_positions))
    if fore_index == rear_index:
        raise ValueError("Could not distinguish fore and rear bodies by X position.")
    return fore_index, rear_index


def _component_values(loads: LoadMeans, spec: ComponentSpec) -> np.ndarray:
    """Return one component vector across bodies."""
    if spec.array_name == "force":
        return loads.forces_E_N[:, spec.axis]
    return loads.moments_E_Cg_Nm[:, spec.axis]


def _display_name(spec: ComponentSpec) -> str:
    """Return a human-readable component name."""
    return spec.name if spec.array_name == "force" else spec.name.title()


def _delta_column_name(body: str, spec: ComponentSpec) -> str:
    """Return the CSV column name for one body/component delta."""
    units_label = "N" if spec.units == "N" else "Nm"
    return f"{body}_delta_{spec.name}_{units_label}"


def build_delta_rows(
    input_rows: list[dict[str, str]],
    results_root: Path,
    prefer_true_single: bool,
) -> tuple[list[dict[str, Any]], dict[float, Baseline]]:
    """Build per-case fore/rear load deltas against the matching baseline."""
    aoa_values = {float(row["aoa_deg"]) for row in input_rows}
    baselines = _load_baselines(results_root, aoa_values, prefer_true_single)
    delta_rows: list[dict[str, Any]] = []

    for row in input_rows:
        aoa_deg = float(row["aoa_deg"])
        summary_path = results_root / row["source_summary"]
        history_path = summary_path.with_name("history.npz")
        case_loads = _load_final_window_means(history_path)
        fore_index, rear_index = _body_indices_from_positions(case_loads.positions_E_m)
        baseline = baselines[aoa_deg].loads

        delta_row: dict[str, Any] = {
            "aoa_deg": aoa_deg,
            "xB": float(row["xB"]),
            "yB": float(row["yB"]),
            "zB": float(row["zB"]),
            "fore_body_index": fore_index,
            "rear_body_index": rear_index,
            "source_summary": row["source_summary"],
            "baseline_source": baselines[aoa_deg].source,
        }
        for spec in COMPONENTS:
            case_values = _component_values(case_loads, spec)
            single_value = float(_component_values(baseline, spec)[0])
            single_key = f"single_{spec.name}_{spec.units.replace(' ', '')}"
            delta_row[single_key] = single_value
            delta_row[_delta_column_name("fore", spec)] = float(
                case_values[fore_index] - single_value
            )
            delta_row[_delta_column_name("rear", spec)] = float(
                case_values[rear_index] - single_value
            )
        delta_rows.append(delta_row)

    return delta_rows, baselines


def _write_delta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the computed delta rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _edges(values: np.ndarray) -> np.ndarray:
    """Return pcolormesh cell edges from center values."""
    if values.size < 2:
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])
        return values
    mids = 0.5 * (values[1:] + values[:-1])
    return np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            mids,
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )


def _draw_fore_wing_outline(ax: plt.Axes) -> None:
    """Draw the fore/source wing footprint in normalized coordinates."""
    wing_x = np.array([0.0, CHORD_M / SPAN_M, CHORD_M / SPAN_M, 0.0, 0.0])
    wing_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
    ax.fill(wing_x, wing_y, color="black", alpha=0.94, zorder=8)


def _grid_from_rows(
    rows: list[dict[str, Any]],
    aoa_deg: float,
    z_over_span: float,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X values, Y values, and one delta grid."""
    subset = [
        row
        for row in rows
        if np.isclose(float(row["aoa_deg"]), aoa_deg)
        and np.isclose(float(row["zB"]), z_over_span)
    ]
    if not subset:
        return np.array([]), np.array([]), np.empty((0, 0), dtype=float)
    x_values = np.array(sorted({float(row["xB"]) for row in subset}), dtype=float)
    y_values = np.array(sorted({float(row["yB"]) for row in subset}), dtype=float)
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    for row in subset:
        grid[y_index[float(row["yB"])], x_index[float(row["xB"])]] = float(
            row[value_key]
        )
    return x_values, y_values, grid


def _norm_for_grid(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local diverging norm centered on zero."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        half_range = 1.0
    else:
        half_range = max(1.0e-14, float(np.nanmax(np.abs(finite_values))))
    return TwoSlopeNorm(vmin=-half_range, vcenter=0.0, vmax=half_range)


def plot_component_map(
    rows: list[dict[str, Any]],
    spec: ComponentSpec,
    output_path: Path,
) -> None:
    """Plot fore/rear delta maps for one component."""
    fig, axes = plt.subplots(
        len(CONDITIONS),
        2,
        figsize=(13.2, 3.55 * len(CONDITIONS)),
        constrained_layout=True,
    )
    mesh = None
    for row_index, (aoa_deg, z_over_span, condition_label) in enumerate(CONDITIONS):
        for col_index, (body, body_label) in enumerate(
            (("fore", "Fore flyer - single"), ("rear", "Rear flyer - single"))
        ):
            ax = axes[row_index, col_index]
            key = _delta_column_name(body, spec)
            x_values, y_values, grid = _grid_from_rows(
                rows,
                aoa_deg=aoa_deg,
                z_over_span=z_over_span,
                value_key=key,
            )
            if grid.size == 0:
                ax.set_axis_off()
                continue
            x_edges = _edges(x_values)
            y_edges = _edges(y_values)
            norm = _norm_for_grid(grid)
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )
            _draw_fore_wing_outline(ax)
            ax.set_title(f"{condition_label} | {body_label}")
            ax.set_xlabel("X/B")
            ax.set_ylabel("Y/B")
            ax.set_xlim(min(-0.02, x_edges[0]), x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.grid(False)
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.025)
            cbar.set_label(f"case - single {_display_name(spec)} ({spec.units})")
            cbar.ax.tick_params(labelsize=7)
    if mesh is None:
        raise RuntimeError(f"No map data available for {spec.name}.")

    fig.suptitle(
        "Two-Flyer Fixed-Wing Delta "
        f"{_display_name(spec)} Against Single-Flyer Baseline"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_figures(
    input_csv: Path,
    results_root: Path,
    output_figure_dir: Path,
    output_data_csv: Path,
    prefer_true_single: bool,
) -> list[Path]:
    """Build all component maps and return output paths."""
    _style()
    rows = _read_csv_rows(input_csv)
    delta_rows, _ = build_delta_rows(
        input_rows=rows,
        results_root=results_root,
        prefer_true_single=prefer_true_single,
    )
    _write_delta_csv(output_data_csv, delta_rows)

    output_paths = []
    for spec in COMPONENTS:
        output_path = (
            output_figure_dir
            / f"two_flyer_delta_{spec.output_suffix}_vs_single_maps.png"
        )
        plot_component_map(delta_rows, spec, output_path)
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot maps of fore/rear fixed-wing two-flyer load deltas relative "
            "to the matching single-flyer baseline."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=DEFAULT_OUTPUT_FIGURE_DIR,
    )
    parser.add_argument("--output-data-csv", type=Path, default=DEFAULT_OUTPUT_DATA_CSV)
    parser.add_argument(
        "--prefer-true-single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the true single-wing AOA 5 history when available; otherwise "
            "use the body-mean far-lateral baseline for that AOA."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    output_paths = build_figures(
        input_csv=args.input_csv,
        results_root=args.results_root,
        output_figure_dir=args.output_figure_dir,
        output_data_csv=args.output_data_csv,
        prefer_true_single=bool(args.prefer_true_single),
    )
    print(f"Wrote delta data to: {args.output_data_csv}")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
