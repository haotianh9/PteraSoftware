"""Organize fixed-wing streamwise formation results into one coherent dataset.

The raw output directory contains several exploratory and rerun roots.  This script
builds a canonical dataset by selecting the best available summary for each
``(AOA, X/B, Y/B, Z/B)`` point, normalizing thrust by the matching single-wing
baseline, and regenerating publication-style maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    ROOT / "output" / "free_flight_cases" / "streamwise_stability_energy"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "curated_fixed_wing_dataset"

SPAN_M = 1.0
CHORD_M = 0.1
AOA_VALUES = (5.0, 10.0, 15.0)


def _raw_results_root(results_root: Path) -> Path:
    """Return the directory that holds raw simulation result roots."""
    raw_sources = results_root / "raw_sources"
    return raw_sources if raw_sources.exists() else results_root


@dataclass(frozen=True)
class SourceRoot:
    """A raw result root and its priority for duplicate case selection."""

    path: Path
    priority: int
    note: str


def _style() -> None:
    """Apply a consistent plotting style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "#f8f8f8",
            "axes.edgecolor": "#222222",
            "grid.color": "#d2d2d2",
            "grid.linewidth": 0.55,
            "axes.grid": True,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""
    return json.loads(path.read_text())


def _case_key(summary: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return the canonical case key, excluding baselines."""
    try:
        aoa = float(summary["angle_of_attack_deg"])
        x_over_span = float(summary["x_over_span_initial"])
        y_over_span = float(summary["y_over_span_prescribed"])
        z_over_span = float(summary["z_over_span_prescribed"])
    except (KeyError, TypeError, ValueError):
        return None
    if np.isclose(y_over_span, 10.0):
        return None
    return (
        round(aoa, 6),
        round(x_over_span, 6),
        round(y_over_span, 6),
        round(z_over_span, 6),
    )


def _run_label(x_over_span: float, y_over_span: float, z_over_span: float) -> str:
    """Return the case-directory label used by the sweep scripts."""

    def fmt(value: float) -> str:
        prefix = "m" if value < 0.0 else "p"
        return prefix + f"{abs(value):.3f}".replace(".", "p")

    return f"xB_{fmt(x_over_span)}_yB_{fmt(y_over_span)}_zB_{fmt(z_over_span)}"


def _source_roots(results_root: Path) -> list[SourceRoot]:
    """Return raw roots used for curation, in increasing priority."""
    raw_root = _raw_results_root(results_root)
    names = (
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_Z0-0p5-1_X0p25-5_Y0-1p5_t9s_corrected_translating_dt12_parallel10",
            10,
            "AOA5 base grid, X/B<=5",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_Z0-0p5-1_X4-7-9_Y0-1p5_t9s_dt12",
            15,
            "AOA5 X/B=4 and provisional X/B=7 grid; X/B=9 excluded",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_Z0-0p5-1_X7-9_Y0-1p5_t12s_dt12",
            20,
            "AOA5 12 s X/B=7 partial grid; X/B=9 excluded",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA10_Z0_X0p25-9_Y0-1p5_t9s_dt12",
            10,
            "AOA10 base grid; X/B=9 absent in this root",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA15_Z0_X0p25-9_Y0-1p5_t9s_dt12",
            10,
            "AOA15 base grid; X/B=9 absent in this root",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA10_Z0_X7-9_Y0-1p5_t12s_dt12",
            20,
            "AOA10 12 s X/B=7 partial grid; X/B=9 excluded",
        ),
        (
            "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA15_Z0_X7-9_Y0-1p5_t12s_dt12",
            20,
            "AOA15 12 s X/B=7 partial grid; X/B=9 excluded",
        ),
        (
            "rerun_slopeonly_after9s_AOA05_X7_Z0_Y0p75-1p5_t20_threads4",
            50,
            "AOA5 converged X/B=7, Z/B=0 reruns",
        ),
        (
            "rerun_slopeonly_after9s_AOA05_X7_Z0p5-1_Y0-1p5_t20_threads4",
            50,
            "AOA5 converged X/B=7, Z/B=0.5 selected reruns",
        ),
        (
            "rerun_slopeonly_after9s_AOA10_X5-7_Z0_selectedY_t20_threads4",
            50,
            "AOA10 converged selected reruns",
        ),
        (
            "rerun_slopeonly_after9s_AOA15_X5_Z0_Y0p25-1p25_t20_threads4",
            50,
            "AOA15 converged X/B=5 selected reruns",
        ),
        (
            "rerun_slopeonly_after9s_AOA15_X7_Z0_Y0p5-1p5_t20_threads4",
            50,
            "AOA15 converged X/B=7 selected reruns",
        ),
    )
    return [
        SourceRoot(raw_root / name, priority, note)
        for name, priority, note in names
        if (raw_root / name).exists()
    ]


def _baseline_by_aoa(results_root: Path) -> dict[float, float]:
    """Return single-wing baseline thrust for each AOA."""
    raw_root = _raw_results_root(results_root)
    baseline_roots = {
        5.0: "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_Z0-0p5-1_X0p25-5_Y0-1p5_t9s_corrected_translating_dt12_parallel10",
        10.0: "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA10_Z0_X0p25-9_Y0-1p5_t9s_dt12",
        15.0: "fixed_wing_rect_B1_c0p1_fixed_trim_U1_AOA15_Z0_X0p25-9_Y0-1p5_t9s_dt12",
    }
    baselines: dict[float, float] = {}
    for aoa, root_name in baseline_roots.items():
        summary_path = raw_root / root_name / "baseline_far_lateral" / "summary.json"
        summary = _read_json(summary_path)
        thrust = np.asarray(
            summary["final_window_mean_required_thrust_E_N"], dtype=float
        )
        baselines[aoa] = float(np.mean(thrust))
    return baselines


def _select_rows(results_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the best available row for every case key."""
    baselines = _baseline_by_aoa(results_root)
    selected: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    source_notes = []

    for source in _source_roots(results_root):
        source_notes.append(
            {
                "root": str(source.path.relative_to(results_root)),
                "priority": source.priority,
                "note": source.note,
            }
        )
        for summary_path in sorted(source.path.glob("*/summary.json")):
            summary = _read_json(summary_path)
            if summary.get("run_status") != "ok":
                rejected.append(
                    {
                        "summary": str(summary_path.relative_to(results_root)),
                        "reason": f"run_status={summary.get('run_status')}",
                    }
                )
                continue
            key = _case_key(summary)
            if key is None:
                continue
            aoa, x_over_span, y_over_span, z_over_span = key
            if x_over_span > 7.0:
                rejected.append(
                    {
                        "summary": str(summary_path.relative_to(results_root)),
                        "reason": "X/B > 7 excluded from coherent dataset",
                    }
                )
                continue

            thrust = np.asarray(
                summary["final_window_mean_required_thrust_E_N"], dtype=float
            )
            single_thrust = baselines[aoa]
            row = {
                "aoa_deg": aoa,
                "xB": x_over_span,
                "yB": y_over_span,
                "zB": z_over_span,
                "front_thrust_N": float(thrust[0]),
                "rear_thrust_N": float(thrust[1]),
                "pair_mean_thrust_N": float(np.mean(thrust)),
                "single_body_thrust_N": single_thrust,
                "front_ratio": float(thrust[0] / single_thrust),
                "rear_ratio": float(thrust[1] / single_thrust),
                "pair_mean_ratio": float(np.mean(thrust) / single_thrust),
                "rear_minus_front_thrust_N": float(thrust[1] - thrust[0]),
                "time_total_completed_s": float(
                    summary.get("time_total_completed_s", np.nan)
                ),
                "early_converged": bool(summary.get("early_converged", False)),
                "early_stop_time_s": summary.get("early_stop_time_s"),
                "source_priority": source.priority,
                "source_root": str(source.path.relative_to(results_root)),
                "source_summary": str(summary_path.relative_to(results_root)),
            }
            for body_index, prefix in ((0, "front"), (1, "rear")):
                stats_prefix = f"body_{body_index}_"
                force_stats = summary.get("clamp_force_yz_stats_N", {})
                torque_stats = summary.get("clamp_torque_stats_Nm", {})
                row[f"{prefix}_clamp_Fy_rms_N"] = force_stats.get(
                    f"{stats_prefix}Fy_rms", np.nan
                )
                row[f"{prefix}_clamp_Fz_rms_N"] = force_stats.get(
                    f"{stats_prefix}Fz_rms", np.nan
                )
                row[f"{prefix}_clamp_Mx_rms_Nm"] = torque_stats.get(
                    f"{stats_prefix}Mx_rms", np.nan
                )
                row[f"{prefix}_clamp_My_rms_Nm"] = torque_stats.get(
                    f"{stats_prefix}My_rms", np.nan
                )
                row[f"{prefix}_clamp_Mz_rms_Nm"] = torque_stats.get(
                    f"{stats_prefix}Mz_rms", np.nan
                )

            existing = selected.get(key)
            if existing is None:
                selected[key] = row
                continue
            existing_time = float(existing.get("time_total_completed_s", 0.0))
            row_time = float(row.get("time_total_completed_s", 0.0))
            existing_choice = (int(existing["source_priority"]), existing_time)
            row_choice = (int(row["source_priority"]), row_time)
            if row_choice >= existing_choice:
                rejected.append(
                    {
                        "summary": existing["source_summary"],
                        "reason": f"superseded by {row['source_summary']}",
                    }
                )
                selected[key] = row
            else:
                rejected.append(
                    {
                        "summary": row["source_summary"],
                        "reason": f"lower priority than {existing['source_summary']}",
                    }
                )

    rows = [selected[key] for key in sorted(selected)]
    metadata = {
        "baselines_by_aoa_N": baselines,
        "source_roots": source_notes,
        "rejected_or_superseded": rejected,
        "num_rows": len(rows),
        "excluded_x_over_span_values": [9.0],
        "normalization": (
            "front_ratio and rear_ratio are required streamwise thrust divided by "
            "the matching far-lateral single-body baseline for the same AOA"
        ),
        "color_convention": "blue < 1 means lower thrust than single-body baseline; red > 1 means higher thrust",
    }
    return rows, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid_from_rows(
    rows: list[dict[str, Any]],
    aoa: float,
    z_over_span: float,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X values, Y values, and grid for one AOA/Z slice."""
    subset = [
        row
        for row in rows
        if np.isclose(row["aoa_deg"], aoa) and np.isclose(row["zB"], z_over_span)
    ]
    if not subset:
        return np.array([]), np.array([]), np.empty((0, 0))
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


def _draw_front_wing(ax: plt.Axes) -> None:
    """Draw the body-1/front wing footprint in normalized coordinates."""
    wing_x = np.array([0.0, CHORD_M / SPAN_M, CHORD_M / SPAN_M, 0.0, 0.0])
    wing_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
    ax.plot(wing_x, wing_y, color="black", linewidth=1.0)


def _ratio_norm(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> TwoSlopeNorm:
    """Return a centered norm for all requested ratio values."""
    values = []
    for key in keys:
        values.extend(float(row[key]) for row in rows if np.isfinite(float(row[key])))
    values_array = np.asarray(values, dtype=float)
    half_range = max(0.05, float(np.nanmax(np.abs(values_array - 1.0))))
    return TwoSlopeNorm(vmin=1.0 - half_range, vcenter=1.0, vmax=1.0 + half_range)


def _plot_ratio_maps(
    rows: list[dict[str, Any]],
    output_path: Path,
    conditions: tuple[tuple[float, float, str], ...],
    title: str,
) -> None:
    """Plot front and rear thrust ratios for selected slices."""
    norm = _ratio_norm(rows, ("front_ratio", "rear_ratio"))
    fig, axes = plt.subplots(
        len(conditions),
        2,
        figsize=(13.0, 4.0 * len(conditions)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    if len(conditions) == 1:
        axes = np.asarray([axes])
    mesh = None
    for row_id, (aoa, z_over_span, label) in enumerate(conditions):
        for col_id, (value_key, body_label) in enumerate(
            (("front_ratio", "Body 1 front"), ("rear_ratio", "Body 2 rear"))
        ):
            ax = axes[row_id, col_id]
            x_values, y_values, grid = _grid_from_rows(
                rows, aoa, z_over_span, value_key
            )
            if grid.size == 0:
                ax.set_axis_off()
                continue
            mesh = ax.pcolormesh(
                _edges(x_values),
                _edges(y_values),
                grid,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )
            _draw_front_wing(ax)
            ax.set_title(f"{label} | {body_label}")
            ax.set_xlabel("X/B")
            ax.set_ylabel("Y/B")
            ax.set_xlim(_edges(x_values)[0], _edges(x_values)[-1])
            ax.set_ylim(_edges(y_values)[0], _edges(y_values)[-1])
            ax.grid(False)
    if mesh is None:
        raise RuntimeError("No data available for ratio maps.")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("Required thrust / single-wing thrust (white = 1)")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rear_derivative_maps(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot unnormalized rear dT/d(X/B) for Z/B=0."""
    conditions = (
        (5.0, 0.0, "AOA 5 deg, Z/B = 0"),
        (10.0, 0.0, "AOA 10 deg, Z/B = 0"),
        (15.0, 0.0, "AOA 15 deg, Z/B = 0"),
    )
    derivative_payload = []
    all_values = []
    for aoa, z_over_span, label in conditions:
        x_values, y_values, thrust_grid = _grid_from_rows(
            rows, aoa, z_over_span, "rear_thrust_N"
        )
        derivative_grid = np.full_like(thrust_grid, np.nan)
        for row_index in range(thrust_grid.shape[0]):
            valid = np.isfinite(thrust_grid[row_index])
            if np.count_nonzero(valid) >= 2:
                derivative_grid[row_index, valid] = np.gradient(
                    thrust_grid[row_index, valid], x_values[valid]
                )
        derivative_payload.append((x_values, y_values, derivative_grid, label))
        all_values.extend(derivative_grid[np.isfinite(derivative_grid)])

    all_values_array = np.asarray(all_values, dtype=float)
    vmax = max(1.0e-12, float(np.nanmax(np.abs(all_values_array))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 11.0), constrained_layout=True)
    mesh = None
    for ax, (x_values, y_values, grid, label) in zip(axes, derivative_payload):
        mesh = ax.pcolormesh(
            _edges(x_values),
            _edges(y_values),
            grid,
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )
        _draw_front_wing(ax)
        ax.set_title(label)
        ax.set_xlabel("X/B")
        ax.set_ylabel("Y/B")
        ax.grid(False)
    if mesh is None:
        raise RuntimeError("No data available for derivative maps.")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("Rear dT / d(X/B) (N)")
    fig.suptitle("Rear-Wing Streamwise Thrust Gradient")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _clean_stale_outputs(results_root: Path) -> list[str]:
    """Remove stale generated outputs that are superseded by the curated dataset."""
    raw_root = _raw_results_root(results_root)
    stale_paths = [
        results_root / "combined_extended_x_replots",
        results_root / "pair_force_torque_stream_check",
        raw_root / "combined_extended_x_replots",
        raw_root / "pair_force_torque_stream_check",
    ]
    removed = []
    for path in stale_paths:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(results_root)))
    for map_name in (
        "clamp_load_map.png",
        "energy_map.png",
        "theory_validation_9panel.png",
        "thrust_difference_map.png",
    ):
        for path in results_root.glob(f"rerun_slopeonly_after9s_*/{map_name}"):
            path.unlink()
            removed.append(str(path.relative_to(results_root)))
        for path in raw_root.glob(f"rerun_slopeonly_after9s_*/{map_name}"):
            path.unlink()
            removed.append(str(path.relative_to(results_root)))
    return removed


def build_dataset(
    results_root: Path, output_dir: Path, clean_stale: bool
) -> dict[str, Any]:
    """Build the curated dataset and figures."""
    _style()
    removed = _clean_stale_outputs(results_root) if clean_stale else []
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = _select_rows(results_root)
    csv_path = output_dir / "curated_streamwise_summary.csv"
    json_path = output_dir / "curated_streamwise_summary.json"
    _write_csv(csv_path, rows)

    figures_dir = output_dir / "figures"
    _plot_ratio_maps(
        rows=rows,
        output_path=figures_dir / "z0_thrust_ratio_front_rear_by_aoa.png",
        conditions=(
            (5.0, 0.0, "AOA 5 deg, Z/B = 0"),
            (10.0, 0.0, "AOA 10 deg, Z/B = 0"),
            (15.0, 0.0, "AOA 15 deg, Z/B = 0"),
        ),
        title="Fixed-Wing Formation Required Thrust Ratios, Z/B = 0",
    )
    _plot_ratio_maps(
        rows=rows,
        output_path=figures_dir / "aoa05_thrust_ratio_front_rear_by_z.png",
        conditions=(
            (5.0, 0.0, "AOA 5 deg, Z/B = 0"),
            (5.0, 0.5, "AOA 5 deg, Z/B = 0.5"),
            (5.0, 1.0, "AOA 5 deg, Z/B = 1"),
        ),
        title="AOA 5 deg Required Thrust Ratios Across Vertical Offset",
    )
    _plot_rear_derivative_maps(
        rows=rows,
        output_path=figures_dir / "z0_rear_dthrust_dxb_by_aoa.png",
    )
    # Keep comparison figures in the same curated figure directory.
    import plot_single_wing_wake_four_panel as single_wing_wake
    import plot_streamwise_sim_by_aoa_single_analytic as sim_analytic

    sim_analytic_path = figures_dir / "sim_by_aoa_single_analytic_reference.png"
    sim_analytic.build_figure(input_csv=csv_path, output_figure=sim_analytic_path)

    single_wing_case_dir = single_wing_wake.DEFAULT_CASE_DIR
    raw_single_wing_case_dir = (
        _raw_results_root(results_root)
        / "single_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_t9s_corrected_translating_dt12"
    )
    if raw_single_wing_case_dir.exists():
        single_wing_case_dir = raw_single_wing_case_dir
    single_wing_wake_path = (
        figures_dir / "single_wing_wake_eight_panel_sim_vs_analytic.png"
    )
    single_wing_wake.build_figure(
        case_dir=single_wing_case_dir,
        step=single_wing_wake.DEFAULT_STEP,
        output=single_wing_wake_path,
    )

    metadata.update(
        {
            "curated_csv": str(csv_path),
            "figures": {
                "z0_thrust_ratio_front_rear_by_aoa": str(
                    figures_dir / "z0_thrust_ratio_front_rear_by_aoa.png"
                ),
                "aoa05_thrust_ratio_front_rear_by_z": str(
                    figures_dir / "aoa05_thrust_ratio_front_rear_by_z.png"
                ),
                "z0_rear_dthrust_dxb_by_aoa": str(
                    figures_dir / "z0_rear_dthrust_dxb_by_aoa.png"
                ),
                "sim_by_aoa_single_analytic_reference": str(sim_analytic_path),
                "single_wing_wake_eight_panel_sim_vs_analytic": str(
                    single_wing_wake_path
                ),
            },
            "removed_stale_outputs": removed,
            "rows": rows,
        }
    )
    json_path.write_text(json.dumps(metadata, indent=2))
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Curated Fixed-Wing Streamwise Formation Dataset\n\n"
        "This directory is the canonical organized dataset for the fixed-wing "
        "streamwise formation sweep.  Ratios are normalized by the matching "
        "far-lateral single-wing baseline for each AOA.  Values below 1 mean "
        "lower required thrust than the single-wing baseline; values above 1 "
        "mean higher required thrust.\n\n"
        "Generated files:\n"
        f"- `{csv_path.name}`: flat curated table.\n"
        f"- `{json_path.name}`: table plus source-selection metadata.\n"
        "- `figures/z0_thrust_ratio_front_rear_by_aoa.png`\n"
        "- `figures/aoa05_thrust_ratio_front_rear_by_z.png`\n"
        "- `figures/z0_rear_dthrust_dxb_by_aoa.png`\n"
        "- `figures/sim_by_aoa_single_analytic_reference.png`\n"
        "- `figures/single_wing_wake_eight_panel_sim_vs_analytic.png`\n"
        "\nRebuild command:\n\n"
        "```bash\n"
        ".venv/bin/python examples/organize_streamwise_fixed_wing_dataset.py\n"
        "```\n"
    )
    root_readme_path = results_root / "README.md"
    root_readme_path.write_text(
        "# Streamwise Stability Energy Results\n\n"
        "Clean layout:\n\n"
        "- `curated_fixed_wing_dataset/`: canonical dataset tables and regenerated figures.\n"
        "- `raw_sources/`: archived raw simulation case directories used to rebuild the curated dataset.\n\n"
        "Use the curated folder for analysis and manuscript figures. The raw-source "
        "folder is kept only for traceability and regeneration.\n\n"
        "Rebuild command:\n\n"
        "```bash\n"
        ".venv/bin/python examples/organize_streamwise_fixed_wing_dataset.py\n"
        "```\n"
    )
    metadata["readme"] = str(readme_path)
    metadata["results_root_readme"] = str(root_readme_path)
    json_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--clean-stale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove stale generated plots/debug result dirs superseded by curation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the organizer."""
    args = parse_args()
    metadata = build_dataset(
        results_root=args.results_root,
        output_dir=args.output_dir,
        clean_stale=args.clean_stale,
    )
    print(metadata["curated_csv"])
    for figure_path in metadata["figures"].values():
        print(figure_path)


if __name__ == "__main__":
    main()
