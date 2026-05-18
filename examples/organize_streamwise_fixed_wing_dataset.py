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
THEORY_REFERENCE_AOA_DEG = 5.0
THEORY_REFERENCE_Z_OVER_SPAN = 0.0
Z0_AOA_CONDITIONS: tuple[tuple[float, float, str], ...] = (
    (5.0, 0.0, "AOA 5 deg, Z/B = 0"),
    (10.0, 0.0, "AOA 10 deg, Z/B = 0"),
    (15.0, 0.0, "AOA 15 deg, Z/B = 0"),
)
MAP_FIGSIZE = (16.0, 18.0)
REAR_MAP_FIGSIZE = (MAP_FIGSIZE[0] / 2.0, MAP_FIGSIZE[1])


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
            "savefig.bbox": "standard",
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
        "color_convention": (
            "Thrust-ratio maps: blue < 1 means lower thrust than single-body "
            "baseline and red > 1 means higher thrust. Power maps: blue is "
            "power saving and red is extra power consumption. Stability maps: "
            "blue is stable and red is unstable."
        ),
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
    ax.fill(wing_x, wing_y, color="black", alpha=0.94, zorder=8)


def _draw_tip_pair_neutral_boundary(
    ax: plt.Axes,
    y_values: np.ndarray,
    z_over_span: float,
) -> None:
    """Overlay the derived point-tip-vortex neutral stability boundary."""
    import plot_streamwise_sim_by_aoa_single_analytic as sim_analytic

    if y_values.size == 0:
        return
    y_min = max(float(np.nanmin(y_values)), 0.5 + 1.0e-6)
    y_max = float(np.nanmax(y_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
        return
    y_dense = np.linspace(y_min, y_max, 300)
    x_boundary = sim_analytic.horseshoe_model.tip_pair_neutral_x_over_span(
        y_dense,
        z_over_span,
        sim_analytic.ANALYTIC_PARAMS,
    )
    valid = np.isfinite(x_boundary)
    if np.any(valid):
        ax.plot(
            x_boundary[valid],
            y_dense[valid],
            color="#111111",
            linestyle="-",
            linewidth=1.45,
            zorder=9,
            label="tip-vortex $\\partial_X\\mathcal{W}=0$",
        )


def _horseshoe_module():
    """Import the analytical horseshoe/tip-vortex model."""
    try:
        from examples import horseshoe_tip_vortex_stability as horseshoe_model
    except ImportError:  # pragma: no cover - supports direct script execution
        import horseshoe_tip_vortex_stability as horseshoe_model
    return horseshoe_model


def _analytical_params_for_aoa(aoa_deg: float):
    """Return analytical horseshoe parameters for the plotted AOA."""
    horseshoe_model = _horseshoe_module()
    return horseshoe_model.HorseshoeWakeParams(
        span_m=SPAN_M,
        chord_m=CHORD_M,
        aoa_deg=float(aoa_deg),
    )


def _condition_baseline_thrust(
    rows: list[dict[str, Any]],
    aoa_deg: float,
    z_over_span: float,
) -> float:
    """Return the matching simulation single-wing thrust normalization."""
    values = [
        float(row["single_body_thrust_N"])
        for row in rows
        if np.isclose(float(row["aoa_deg"]), aoa_deg)
        and np.isclose(float(row["zB"]), z_over_span)
    ]
    if not values:
        raise RuntimeError(
            f"No baseline thrust found for AOA={aoa_deg}, Z/B={z_over_span}."
        )
    return float(np.mean(values))


def _analytical_ratio_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    aoa_deg: float,
    z_over_span: float,
    baseline_thrust_n: float,
    *,
    body: str,
    wake_only: bool,
) -> np.ndarray:
    """Return front/rear analytical thrust ratio on the simulation grid."""
    horseshoe_model = _horseshoe_module()
    params = _analytical_params_for_aoa(aoa_deg)
    wbar_function = (
        horseshoe_model.lift_weighted_tip_pair_wbar_mps
        if wake_only
        else horseshoe_model.lift_weighted_wbar_mps
    )
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    for y_index, y_over_span in enumerate(y_values):
        for x_index, x_over_span in enumerate(x_values):
            if body == "front":
                source_x = -float(x_over_span) * SPAN_M
                source_y = -float(y_over_span) * SPAN_M
                source_z = -float(z_over_span) * SPAN_M
            elif body == "rear":
                source_x = float(x_over_span) * SPAN_M
                source_y = float(y_over_span) * SPAN_M
                source_z = float(z_over_span) * SPAN_M
            else:
                raise ValueError("body must be 'front' or 'rear'.")
            wbar_mps = wbar_function(
                source_x,
                source_y,
                source_z,
                params,
            )
            delta_thrust_n = horseshoe_model.thrust_change_n(wbar_mps, params)
            grid[y_index, x_index] = (
                baseline_thrust_n + float(delta_thrust_n)
            ) / baseline_thrust_n
    return grid


def _analytical_rear_dthrust_dxb_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    aoa_deg: float,
    z_over_span: float,
    *,
    wake_only: bool,
) -> np.ndarray:
    """Return analytical rear dT/d(X/B) on the simulation grid."""
    horseshoe_model = _horseshoe_module()
    params = _analytical_params_for_aoa(aoa_deg)
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
                params,
            )
    return grid


def _ratio_cbar_ticks(norm: TwoSlopeNorm) -> list[float]:
    """Return ratio colorbar ticks with explicit ticks between zero and one."""
    dense_under_one = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    if norm.vmax <= 2.0:
        above_one = np.arange(1.25, norm.vmax + 0.125, 0.25)
    elif norm.vmax <= 4.0:
        above_one = np.arange(1.5, norm.vmax + 0.25, 0.5)
    else:
        above_one = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    ticks = np.unique(np.concatenate((dense_under_one, above_one)))
    return [float(tick) for tick in ticks if norm.vmin <= tick <= norm.vmax]


def _ratio_norm(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> TwoSlopeNorm:
    """Return a centered norm for all requested ratio values."""
    values = []
    for key in keys:
        values.extend(float(row[key]) for row in rows if np.isfinite(float(row[key])))
    values_array = np.asarray(values, dtype=float)
    half_range = max(0.05, float(np.nanmax(np.abs(values_array - 1.0))))
    return TwoSlopeNorm(
        vmin=max(0.0, 1.0 - half_range),
        vcenter=1.0,
        vmax=1.0 + half_range,
    )


def _ratio_norm_for_grid(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local ratio norm centered at the single-wing value."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        half_range = 0.05
    else:
        half_range = max(0.05, float(np.nanmax(np.abs(finite_values - 1.0))))
    return TwoSlopeNorm(
        vmin=max(0.0, 1.0 - half_range),
        vcenter=1.0,
        vmax=1.0 + half_range,
    )


def _derivative_norm_for_grid(grid: np.ndarray) -> TwoSlopeNorm:
    """Return a panel-local dT/dX norm centered on neutral stability."""
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        half_range = 1.0
    else:
        half_range = max(1.0e-12, float(np.nanmax(np.abs(finite_values))))
    return TwoSlopeNorm(vmin=-half_range, vcenter=0.0, vmax=half_range)


def _plot_zero_contour(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
) -> None:
    """Draw the zero contour when the gridded field crosses zero."""
    if (
        x_values.size < 2
        or y_values.size < 2
        or not np.any(np.isfinite(grid))
        or not (np.nanmin(grid) <= 0.0 <= np.nanmax(grid))
    ):
        return
    ax.contour(
        x_values,
        y_values,
        grid,
        levels=[0.0],
        colors="#222222",
        linestyles="--",
        linewidths=1.15,
        zorder=9,
    )


def _plot_ratio_maps(
    rows: list[dict[str, Any]],
    output_path: Path,
    conditions: tuple[tuple[float, float, str], ...],
    title: str,
    *,
    include_analytical: bool = False,
) -> None:
    """Plot front and rear thrust ratios for selected slices."""
    column_specs = [("front", "Front"), ("rear", "Rear")]
    num_rows = len(conditions) + (2 if include_analytical else 0)
    fig, axes = plt.subplots(
        num_rows,
        len(column_specs),
        figsize=MAP_FIGSIZE if include_analytical else (13.0, 4.0 * len(conditions)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    if num_rows == 1:
        axes = np.asarray([axes])
    mesh = None
    for row_id, (aoa, z_over_span, label) in enumerate(conditions):
        for col_id, (body, body_label) in enumerate(column_specs):
            ax = axes[row_id, col_id]
            value_key = "front_ratio" if body == "front" else "rear_ratio"
            x_values, y_values, grid = _grid_from_rows(
                rows, aoa, z_over_span, value_key
            )
            if grid.size == 0:
                ax.set_axis_off()
                continue
            norm = (
                _ratio_norm_for_grid(grid)
                if include_analytical
                else _ratio_norm(
                    rows,
                    ("front_ratio", "rear_ratio"),
                )
            )
            mesh = ax.pcolormesh(
                _edges(x_values),
                _edges(y_values),
                grid,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )
            _draw_front_wing(ax)
            ax.set_title(f"{label} | Simulation {body_label.lower()}")
            ax.set_xlabel("X/B")
            ax.set_ylabel("Y/B")
            ax.set_xlim(min(-0.02, _edges(x_values)[0]), _edges(x_values)[-1])
            ax.set_ylim(_edges(y_values)[0], _edges(y_values)[-1])
            ax.grid(False)
            if include_analytical:
                cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_ticks(_ratio_cbar_ticks(norm))
                cbar.set_label("$T/T_{single}$")
                cbar.ax.tick_params(labelsize=7)
    if include_analytical:
        theory_x_values, theory_y_values, _ = _grid_from_rows(
            rows,
            THEORY_REFERENCE_AOA_DEG,
            THEORY_REFERENCE_Z_OVER_SPAN,
            "rear_ratio",
        )
        theory_baseline_thrust_n = _condition_baseline_thrust(
            rows,
            THEORY_REFERENCE_AOA_DEG,
            THEORY_REFERENCE_Z_OVER_SPAN,
        )
        theory_specs = (
            ("full horseshoe", False),
            ("wake-only tip pair", True),
        )
        for theory_offset, (theory_label, wake_only) in enumerate(theory_specs):
            row_id = len(conditions) + theory_offset
            for col_id, (body, body_label) in enumerate(column_specs):
                ax = axes[row_id, col_id]
                grid = _analytical_ratio_grid(
                    theory_x_values,
                    theory_y_values,
                    THEORY_REFERENCE_AOA_DEG,
                    THEORY_REFERENCE_Z_OVER_SPAN,
                    theory_baseline_thrust_n,
                    body=body,
                    wake_only=wake_only,
                )
                norm = _ratio_norm_for_grid(grid)
                mesh = ax.pcolormesh(
                    _edges(theory_x_values),
                    _edges(theory_y_values),
                    grid,
                    shading="auto",
                    cmap="RdBu_r",
                    norm=norm,
                )
                _draw_front_wing(ax)
                ax.set_title(
                    f"Theory {theory_label} | {body_label}\n"
                    f"single reference AOA {THEORY_REFERENCE_AOA_DEG:g} deg"
                )
                ax.set_xlabel("X/B")
                ax.set_ylabel("Y/B")
                ax.set_xlim(
                    min(-0.02, _edges(theory_x_values)[0]),
                    _edges(theory_x_values)[-1],
                )
                ax.set_ylim(_edges(theory_y_values)[0], _edges(theory_y_values)[-1])
                ax.grid(False)
                cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_ticks(_ratio_cbar_ticks(norm))
                cbar.set_label("$T/T_{single}$")
                cbar.ax.tick_params(labelsize=7)
    if mesh is None:
        raise RuntimeError("No data available for ratio maps.")
    if not include_analytical:
        norm = _ratio_norm(rows, ("front_ratio", "rear_ratio"))
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.015)
        cbar.set_ticks(_ratio_cbar_ticks(norm))
        cbar.set_label("Required thrust / single-wing thrust (white = 1)")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rear_derivative_maps(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot unnormalized rear dT/d(X/B) for Z/B=0."""
    fig, axes = plt.subplots(
        len(Z0_AOA_CONDITIONS) + 2,
        1,
        figsize=REAR_MAP_FIGSIZE,
        constrained_layout=True,
    )
    axes = np.ravel(axes)
    mesh = None
    for row_id, (aoa, z_over_span, label) in enumerate(Z0_AOA_CONDITIONS):
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
        ax = axes[row_id]
        panel_norm = _derivative_norm_for_grid(derivative_grid)
        mesh = ax.pcolormesh(
            _edges(x_values),
            _edges(y_values),
            derivative_grid,
            shading="auto",
            cmap="RdBu_r",
            norm=panel_norm,
        )
        _plot_zero_contour(ax, x_values, y_values, derivative_grid)
        _draw_front_wing(ax)
        ax.set_title(f"{label} | Simulation rear")
        ax.set_xlabel("X/B")
        ax.set_ylabel("Y/B")
        ax.set_xlim(min(-0.02, _edges(x_values)[0]), _edges(x_values)[-1])
        ax.set_ylim(_edges(y_values)[0], _edges(y_values)[-1])
        ax.grid(False)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("dT / d(X/B) (N)")
        cbar.ax.tick_params(labelsize=7)
    theory_x_values, theory_y_values, _ = _grid_from_rows(
        rows,
        THEORY_REFERENCE_AOA_DEG,
        THEORY_REFERENCE_Z_OVER_SPAN,
        "rear_thrust_N",
    )
    theory_payload = (
        (
            _analytical_rear_dthrust_dxb_grid(
                theory_x_values,
                theory_y_values,
                THEORY_REFERENCE_AOA_DEG,
                THEORY_REFERENCE_Z_OVER_SPAN,
                wake_only=False,
            ),
            "Theory full horseshoe rear\n"
            f"single reference AOA {THEORY_REFERENCE_AOA_DEG:g} deg",
            "full_horseshoe",
        ),
        (
            _analytical_rear_dthrust_dxb_grid(
                theory_x_values,
                theory_y_values,
                THEORY_REFERENCE_AOA_DEG,
                THEORY_REFERENCE_Z_OVER_SPAN,
                wake_only=True,
            ),
            "Theory wake-only rear\n"
            f"single reference AOA {THEORY_REFERENCE_AOA_DEG:g} deg",
            "wake_only",
        ),
    )
    for theory_offset, (grid, panel_title, analytic_kind) in enumerate(theory_payload):
        ax = axes[len(Z0_AOA_CONDITIONS) + theory_offset]
        panel_norm = _derivative_norm_for_grid(grid)
        mesh = ax.pcolormesh(
            _edges(theory_x_values),
            _edges(theory_y_values),
            grid,
            shading="auto",
            cmap="RdBu_r",
            norm=panel_norm,
        )
        _plot_zero_contour(ax, theory_x_values, theory_y_values, grid)
        if analytic_kind == "wake_only":
            _draw_tip_pair_neutral_boundary(
                ax,
                y_values=theory_y_values,
                z_over_span=THEORY_REFERENCE_Z_OVER_SPAN,
            )
        _draw_front_wing(ax)
        ax.set_title(panel_title)
        ax.set_xlabel("X/B")
        ax.set_ylabel("Y/B")
        ax.set_xlim(min(-0.02, _edges(theory_x_values)[0]), _edges(theory_x_values)[-1])
        ax.set_ylim(_edges(theory_y_values)[0], _edges(theory_y_values)[-1])
        ax.grid(False)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("dT / d(X/B) (N)")
        cbar.ax.tick_params(labelsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", framealpha=0.88)
    if mesh is None:
        raise RuntimeError("No data available for derivative maps.")
    fig.suptitle(
        "Rear-Wing Streamwise Thrust Gradient: Simulation, Full Horseshoe, and Wake-Only"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
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
        conditions=Z0_AOA_CONDITIONS,
        title=(
            "Fixed-Wing Required Thrust Ratios, Z/B = 0: "
            "Simulation and Horseshoe Theory"
        ),
        include_analytical=True,
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
    import horseshoe_tip_vortex_stability as horseshoe_model
    import plot_single_wing_wake_four_panel as single_wing_wake
    import plot_streamwise_sim_by_aoa_single_analytic as sim_analytic

    sim_analytic_path = figures_dir / "sim_by_aoa_single_analytic_reference.png"
    sim_analytic.build_figure(input_csv=csv_path, output_figure=sim_analytic_path)

    stability_9panel_path = figures_dir / "streamwise_stability_9panel.png"
    horseshoe_model.plot_stability_maps(output_path=stability_9panel_path)

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
                "streamwise_stability_9panel": str(stability_9panel_path),
                "single_wing_wake_eight_panel_sim_vs_analytic": str(
                    single_wing_wake_path
                ),
            },
            "analytical_model_parameters": sim_analytic.analytical_model_parameters(),
            "coordinate_convention": (
                "Body 1 is front, Body 2 is rear, and X/B is front-minus-rear "
                "streamwise spacing. Map manuscript bird 2 to plotted front body "
                "if the manuscript uses X=x_2-x_1 with bird 2 ahead."
            ),
            "analytical_comparison_note": (
                "Streamwise-map analytical comparisons and the 9-panel stability "
                "figure use the in-repo horseshoe/tip-vortex Biot-Savart model "
                "from examples/horseshoe_tip_vortex_stability.py. The Z/B=0 "
                "thrust-ratio figure shows simulation front/rear rows by AOA, "
                "then one full-horseshoe and one wake-only analytical reference "
                "row at the end. The dT/d(X/B) figure likewise places one full "
                "and one wake-only analytical reference after the simulation rows."
            ),
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
        "Coordinate convention in these simulation outputs: Body 1 is the front "
        "wing, Body 2 is the rear wing, and `X/B` is the streamwise front-minus-rear "
        "spacing. If the manuscript writes `X=x_2-x_1` with bird 2 ahead, then the "
        "manuscript's bird 2 corresponds to the plotted front body and the "
        "manuscript's bird 1 corresponds to the plotted rear body.\n\n"
        "Streamwise-map analytical comparisons and the 9-panel stability figure use "
        "the in-repo analytical horseshoe/tip-vortex Biot-Savart model in "
        "`examples/horseshoe_tip_vortex_stability.py`. The Z/B=0 thrust-ratio and "
        "dT/d(X/B) figures show simulation rows by AOA first, then one full-horseshoe "
        "and one wake-only analytical reference at the end. The wake-only dT/d(X/B) "
        "reference also overlays the point-receiver tip-vortex-pair neutral contour. "
        "`sim_by_aoa_single_analytic_reference.png` likewise includes separate "
        "full-horseshoe and wake-only analytical thrust-ratio panels.\n\n"
        "Generated files:\n"
        f"- `{csv_path.name}`: flat curated table.\n"
        f"- `{json_path.name}`: table plus source-selection metadata.\n"
        "- `figures/z0_thrust_ratio_front_rear_by_aoa.png`\n"
        "- `figures/aoa05_thrust_ratio_front_rear_by_z.png`\n"
        "- `figures/z0_rear_dthrust_dxb_by_aoa.png`\n"
        "- `figures/sim_by_aoa_single_analytic_reference.png`\n"
        "- `figures/streamwise_stability_9panel.png`\n"
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
