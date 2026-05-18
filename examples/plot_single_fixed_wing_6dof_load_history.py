"""Plot single fixed-wing 6DOF clamp load history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    ROOT / "output" / "free_flight_cases" / "streamwise_stability_energy"
)
DEFAULT_INPUT_HISTORY = (
    DEFAULT_RESULTS_ROOT
    / "raw_sources"
    / "single_wing_rect_B1_c0p1_fixed_trim_U1_AOA05_t9s_corrected_translating_dt12"
    / "single_wing_history.npz"
)
DEFAULT_OUTPUT_FIGURE = (
    DEFAULT_RESULTS_ROOT
    / "curated_fixed_wing_dataset"
    / "figures"
    / "single_wing_6dof_load_history.png"
)

FORCE_KEY_CANDIDATES = ("clamp_force_E_N", "clamp_forces_E_N")
MOMENT_KEY_CANDIDATES = ("clamp_moment_E_Cg_Nm", "clamp_moments_E_Cg_Nm")


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
            "legend.fontsize": 8,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON metadata if present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _array_from_candidates(
    history: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
) -> np.ndarray:
    """Return the first matching array from an NPZ history."""
    for key in keys:
        if key in history.files:
            return np.asarray(history[key], dtype=float)
    raise KeyError(f"None of {keys} were found in {history.files}.")


def _validate_history(
    times_s: np.ndarray,
    forces_E_N: np.ndarray,
    moments_E_Cg_Nm: np.ndarray,
) -> None:
    """Validate the shape of one 6DOF history."""
    if times_s.ndim != 1:
        raise ValueError("time_s must be a one-dimensional array.")
    if forces_E_N.shape != (times_s.size, 3):
        raise ValueError(
            "Force history must have shape (len(time_s), 3); "
            f"got {forces_E_N.shape}."
        )
    if moments_E_Cg_Nm.shape != (times_s.size, 3):
        raise ValueError(
            "Moment history must have shape (len(time_s), 3); "
            f"got {moments_E_Cg_Nm.shape}."
        )


def _case_title(metadata: dict[str, Any], input_history: Path) -> str:
    """Return a compact figure title."""
    case = metadata.get("case", input_history.parent.name)
    aoa = metadata.get("angle_of_attack_deg")
    speed = metadata.get("prescribed_streamwise_speed_mps")
    details = []
    if aoa is not None:
        details.append(f"AOA={float(aoa):g} deg")
    if speed is not None:
        details.append(f"U={float(speed):g} m/s")
    suffix = "" if not details else " (" + ", ".join(details) + ")"
    return f"Single fixed-wing 6DOF clamp loads: {case}{suffix}"


def plot_load_history(
    input_history: Path,
    output_figure: Path,
    show_tail_mean: bool,
) -> None:
    """Plot force and moment components over time."""
    _style()
    with np.load(input_history) as history:
        times_s = np.asarray(history["time_s"], dtype=float)
        forces_E_N = _array_from_candidates(history, FORCE_KEY_CANDIDATES)
        moments_E_Cg_Nm = _array_from_candidates(history, MOMENT_KEY_CANDIDATES)

    _validate_history(times_s, forces_E_N, moments_E_Cg_Nm)
    metadata = _read_json(input_history.with_name("summary.json"))

    output_figure.parent.mkdir(parents=True, exist_ok=True)

    component_data = (
        ("Fx", "Force X (N)", forces_E_N[:, 0], "tab:blue"),
        ("Fy", "Force Y (N)", forces_E_N[:, 1], "tab:orange"),
        ("Fz", "Force Z (N)", forces_E_N[:, 2], "tab:green"),
        ("Roll", "Roll moment (N m)", moments_E_Cg_Nm[:, 0], "tab:red"),
        ("Yaw", "Yaw moment (N m)", moments_E_Cg_Nm[:, 1], "tab:purple"),
        ("Pitch", "Pitch moment (N m)", moments_E_Cg_Nm[:, 2], "tab:brown"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True)
    for ax, (label, ylabel, values, color) in zip(axes.ravel(), component_data):
        ax.plot(times_s, values, color=color, linewidth=1.7, label=label)
        if show_tail_mean and times_s.size > 1:
            tail_start_s = max(float(times_s[0]), float(times_s[-1]) - 2.0)
            tail_mask = times_s >= tail_start_s
            if np.any(tail_mask):
                tail_mean = float(np.mean(values[tail_mask]))
                ax.axhline(
                    tail_mean,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label="last 2 s mean",
                )
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(_case_title(metadata, input_history), y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output_figure)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot single fixed-wing 6DOF force and torque histories."
    )
    parser.add_argument("--input-history", type=Path, default=DEFAULT_INPUT_HISTORY)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_OUTPUT_FIGURE)
    parser.add_argument(
        "--tail-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the last-2-second mean on each component.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the plotter."""
    args = parse_args()
    plot_load_history(
        input_history=args.input_history,
        output_figure=args.output_figure,
        show_tail_mean=bool(args.tail_mean),
    )
    print(f"Saved 6DOF load history plot to: {args.output_figure}")


if __name__ == "__main__":
    main()
