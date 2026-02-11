"""Plot convergence of a coefficient or load from the convergence cache.

Creates a 4 x 4 grid of subplots showing how the selected quantity varies with
num_chordwise_panels refinement.

    Rows: panel aspect ratio (coarsest to finest, top to bottom)
    Columns 1 through 3: wake length (1, 2, 3)
    Column 4: all wake lengths overlaid for direct comparison

Each subplot shows lines for prescribed wake (solid with circle markers) and free wake
(dashed with triangle markers). The y axis is shared within each row for easy comparison
across wake lengths.

Usage:
    python plot_convergence_cache.py <cache_file> [options]

Examples:
    python plot_convergence_cache.py convergence_cache/L170V_R180V_170Hz.json
    python plot_convergence_cache.py convergence_cache/L170V_R180V_170Hz.json --quantity cFY
    python plot_convergence_cache.py convergence_cache/L170V_R180V_170Hz.json --quantity FX
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COEFFICIENT_NAMES = ("cFX", "cFY", "cFZ", "cMX", "cMY", "cMZ")
LOAD_NAMES = ("FX", "FY", "FZ", "MX", "MY", "MZ")
_N_TO_MGF = 1.0 / 9.80665e-6
LOAD_UNITS = ("mgf", "mgf", "mgf", "N*m", "N*m", "N*m")
LOAD_SCALES = (_N_TO_MGF, _N_TO_MGF, _N_TO_MGF, 1.0, 1.0, 1.0)

# Colors for wake length lines in the overlay column.
WAKE_LENGTH_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
}


@contextlib.contextmanager
def _lock_cache_file(cache_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for safe concurrent cache access.

    Uses a .lock file adjacent to the cache file. On Windows uses msvcrt locking, on
    Unix uses fcntl file locking.

    :param cache_path: Path to the cache file to lock.
    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt

            # Write a byte so there is content to lock.
            lock_fh.write(" ")
            lock_fh.flush()
            lock_fh.seek(0)
            # LK_LOCK retries for approximately 10 seconds before raising OSError.
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


def _load_cache(cache_path: Path) -> dict:
    """Load the convergence cache JSON file with file locking.

    :param cache_path: Path to the cache JSON file.
    :return: Parsed JSON data as a dictionary.
    """
    with _lock_cache_file(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)


def _parse_cache(data: dict) -> list[dict]:
    """Parse cache entries into a list of structured records.

    :param data: Raw cache dictionary (will be modified to remove _optimized_dt).
    :return: List of dictionaries with parsed fields.
    """
    data.pop("_optimized_dt", None)

    records = []
    for key, value in data.items():
        parts = key.split(",")
        records.append(
            {
                "prescribed_wake": parts[0].strip() == "True",
                "wake_length": int(parts[1].strip()),
                "panel_ar": int(parts[2].strip()),
                "num_chordwise": int(parts[3].strip()),
                "coefficients": value["coefficients"],
                "loads": value["loads"],
                "time": value["time"],
            }
        )
    return records


def _resolve_quantity(quantity_arg: str) -> tuple[str, int, str, str, float]:
    """Resolve a quantity argument to a data key, index, display name, unit, and scale.

    Accepts a coefficient name (e.g. "cFX"), a load name (e.g. "FX"), or an integer
    index (0 through 5, interpreted as a coefficient index).

    :param quantity_arg: Quantity name or index string.
    :return: Tuple of (data_key, index, display_name, unit, scale) where data_key is
        "coefficients" or "loads", unit is an empty string for coefficients, and scale
        is a multiplicative factor applied to the raw values before plotting.
    """
    arg = quantity_arg.strip()

    # Try as coefficient name.
    for i, name in enumerate(COEFFICIENT_NAMES):
        if arg == name:
            return "coefficients", i, name, "", 1.0

    # Try as load name.
    for i, name in enumerate(LOAD_NAMES):
        if arg == name:
            return "loads", i, name, LOAD_UNITS[i], LOAD_SCALES[i]

    # Try as integer index (default to coefficient).
    try:
        index = int(arg)
        if 0 <= index < len(COEFFICIENT_NAMES):
            return "coefficients", index, COEFFICIENT_NAMES[index], "", 1.0
    except ValueError:
        pass

    all_names = ", ".join(COEFFICIENT_NAMES) + ", " + ", ".join(LOAD_NAMES)
    raise ValueError(
        f"Unknown quantity '{quantity_arg}'. Valid names: {all_names} or coefficient "
        f"indices 0 through 5."
    )


def _filter_records(
    records: list[dict],
    panel_ar: int,
    wake_length: int,
    prescribed_wake: bool,
) -> tuple[list[int], list[float]]:
    """Filter and sort records for a specific parameter combination.

    :param records: All parsed cache records.
    :param panel_ar: Panel aspect ratio to filter by.
    :param wake_length: Wake length to filter by.
    :param prescribed_wake: Whether to filter for prescribed (True) or free (False) wake.
    :return: Tuple of (num_chordwise values, coefficient values), both sorted by
        num_chordwise.
    """
    filtered = [
        r
        for r in records
        if r["panel_ar"] == panel_ar
        and r["wake_length"] == wake_length
        and r["prescribed_wake"] == prescribed_wake
    ]
    filtered.sort(key=lambda r: r["num_chordwise"])
    x = [r["num_chordwise"] for r in filtered]
    return x, filtered


def _plot_wake_lines(
    ax: plt.Axes,
    records: list[dict],
    panel_ar: int,
    wake_length: int,
    data_key: str,
    value_index: int,
    scale: float = 1.0,
    color: str = "tab:blue",
    label_prefix: str = "",
) -> None:
    """Plot prescribed and free wake lines on a single axes.

    :param ax: Matplotlib axes to plot on.
    :param records: All parsed cache records.
    :param panel_ar: Panel aspect ratio to filter by.
    :param wake_length: Wake length to filter by.
    :param data_key: Record key to read values from ("coefficients" or "loads").
    :param value_index: Index within the data array to plot.
    :param scale: Multiplicative factor applied to raw values before plotting.
    :param color: Color for both lines. Prescribed uses solid, free uses dashed.
    :param label_prefix: Prefix for legend labels (e.g. "WL=2, ").
    """
    # Prescribed wake.
    x_vals, filtered = _filter_records(records, panel_ar, wake_length, True)
    if filtered:
        y_vals = [r[data_key][value_index] * scale for r in filtered]
        ax.plot(
            x_vals,
            y_vals,
            "o-",
            color=color,
            markersize=5,
            label=f"{label_prefix}Prescribed",
        )

    # Free wake.
    x_vals, filtered = _filter_records(records, panel_ar, wake_length, False)
    if filtered:
        y_vals = [r[data_key][value_index] * scale for r in filtered]
        ax.plot(
            x_vals,
            y_vals,
            "^--",
            color=color,
            markersize=5,
            label=f"{label_prefix}Free",
        )


def plot_convergence(
    cache_path: Path,
    data_key: str = "coefficients",
    value_index: int = 0,
    display_name: str = "cFX",
    unit: str = "",
    scale: float = 1.0,
) -> None:
    """Create the 4 x 4 convergence plot grid.

    :param cache_path: Path to the convergence cache JSON file.
    :param data_key: Record key to read values from ("coefficients" or "loads").
    :param value_index: Index within the data array to plot (0 through 5).
    :param display_name: Human readable name for the plotted quantity.
    :param unit: Unit string to append to y axis labels (e.g. "mgf", "N*m").
    :param scale: Multiplicative factor applied to raw values before plotting.
    """
    data = _load_cache(cache_path)
    records = _parse_cache(data)

    if not records:
        print("No simulation records found in cache.")
        return

    # Get unique sorted parameter values.
    panel_ars = sorted(set(r["panel_ar"] for r in records), reverse=True)
    wake_lengths = sorted(set(r["wake_length"] for r in records))
    all_chordwise = sorted(set(r["num_chordwise"] for r in records))

    num_rows = min(len(panel_ars), 4)
    # Use up to 3 individual wake length columns + 1 overlay column.
    num_wl_cols = min(len(wake_lengths), 3)
    num_cols = num_wl_cols + 1

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 4 * num_rows),
        squeeze=False,
        sharey="row",
    )
    fig.suptitle(
        f"Convergence of {display_name}  \u2014  {cache_path.stem}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    y_label = f"{display_name} ({unit})" if unit else display_name

    for row, panel_ar in enumerate(panel_ars[:num_rows]):
        # Individual wake length columns.
        for col, wake_length in enumerate(wake_lengths[:num_wl_cols]):
            ax = axes[row, col]
            _plot_wake_lines(
                ax,
                records,
                panel_ar,
                wake_length,
                data_key,
                value_index,
                scale=scale,
                color="tab:blue",
            )

            # Formatting.
            ax.set_title(f"AR = {panel_ar}, Wake Length = {wake_length}", fontsize=10)
            ax.set_xlabel("Num Chordwise Panels")
            if col == 0:
                ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_chordwise)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

        # Overlay column: all wake lengths on one plot.
        ax_overlay = axes[row, num_wl_cols]
        for wake_length in wake_lengths[:num_wl_cols]:
            color = WAKE_LENGTH_COLORS.get(wake_length, "tab:gray")
            _plot_wake_lines(
                ax_overlay,
                records,
                panel_ar,
                wake_length,
                data_key,
                value_index,
                scale=scale,
                color=color,
                label_prefix=f"WL={wake_length}, ",
            )

        ax_overlay.set_title(f"AR = {panel_ar}, All Wake Lengths", fontsize=10)
        ax_overlay.set_xlabel("Num Chordwise Panels")
        ax_overlay.grid(True, alpha=0.3)
        ax_overlay.set_xticks(all_chordwise)
        if row == 0:
            ax_overlay.legend(fontsize=7, ncol=2)

    # Use scientific E notation on y axes when plotting loads.
    if data_key == "loads":
        for ax in axes.flat:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence of a coefficient or load from the convergence cache."
    )
    parser.add_argument(
        "cache_file",
        type=Path,
        help="Path to the convergence cache JSON file.",
    )
    parser.add_argument(
        "--quantity",
        default="cFX",
        help=(
            "Quantity to plot. Accepts a coefficient name "
            "(cFX, cFY, cFZ, cMX, cMY, cMZ), a load name "
            "(FX, FY, FZ, MX, MY, MZ), or a coefficient index "
            "(0 through 5). Default: cFX."
        ),
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        sys.exit(1)

    data_key, value_index, display_name, unit, scale = _resolve_quantity(args.quantity)
    print(f"Plotting {display_name} from {cache_path.name}...")

    plot_convergence(cache_path, data_key, value_index, display_name, unit, scale)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _default_cache = (
            Path(__file__).parent / "convergence_cache" / "L170V_R180V_170Hz.json"
        )
        print(f"Plotting cFX from {_default_cache.name}...")
        plot_convergence(_default_cache)
