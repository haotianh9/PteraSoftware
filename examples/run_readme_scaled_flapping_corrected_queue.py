"""Run the corrected in-phase README-scaled flapping single and pair cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

DEFAULT_BASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "readme_scaled_flapping_streamwise"
)
DEFAULT_X_OVER_SPAN = (0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_Y_OVER_SPAN = (0.0, 0.25, 0.5, 0.75, 1.0)


def _log(message: str, log_path: Path) -> None:
    """Print a timestamped message and append it to a persistent log."""
    stamp = time.strftime("%a %b %d %H:%M:%S %Z %Y")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a") as log_file:
        log_file.write(line + "\n")


def _thread_env(num_threads: int) -> dict[str, str]:
    """Return an environment that allows numerical kernels to use all requested CPUs."""
    env = os.environ.copy()
    thread_value = str(num_threads)
    env.update(
        {
            "OMP_NUM_THREADS": thread_value,
            "OPENBLAS_NUM_THREADS": thread_value,
            "MKL_NUM_THREADS": thread_value,
            "NUMBA_NUM_THREADS": thread_value,
            "NUMEXPR_NUM_THREADS": thread_value,
        }
    )
    return env


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    cpu_set: str,
    num_threads: int,
) -> None:
    """Run one simulation command with CPU affinity and a per-case log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_command = ["taskset", "-c", cpu_set] + command
    with log_path.open("w") as log_file:
        subprocess.run(
            full_command,
            cwd=Path(__file__).resolve().parents[1],
            env=_thread_env(num_threads),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _case_label(x_over_span: float, y_over_span: float) -> str:
    """Create a stable directory/log label for one pair case."""
    return f"xB_{x_over_span:0.3f}_yB_{y_over_span:0.3f}_zB_0".replace(".", "p")


def run_queue(
    *,
    base_dir: Path,
    cpu_set: str,
    num_threads: int,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
) -> None:
    """Run the corrected single case first, then all pair cases sequentially."""
    base_dir.mkdir(parents=True, exist_ok=True)
    queue_log = base_dir / "corrected_inphase_sequential_runner.log"
    single_dir = base_dir / "single_free_x_20p_resolved_wake"
    pair_root = (
        base_dir / "two_flyer_free_x_grid_y0_y025_y05_y075_y1_z0_20p_resolved_wake"
    )

    _log(
        f"Corrected in-phase queue started; cpu_set={cpu_set}, threads={num_threads}",
        queue_log,
    )
    _log("Running corrected single flyer first.", queue_log)
    _run_logged(
        [
            ".venv/bin/python",
            "examples/free_flight_readme_scaled_flapping_streamwise.py",
            "--output-dir",
            str(single_dir),
            "--prescribed-steps",
            str(prescribed_steps),
            "--free-steps",
            str(free_steps),
            "--steps-per-flap",
            str(steps_per_flap),
            "--history-stride",
            str(steps_per_flap),
            "--save-every-n-steps",
            "2",
            "--dense-diagnostics-every-n-steps",
            "1",
            "--no-show-progress",
            "--no-animate",
        ],
        base_dir / "single_free_x_20p_resolved_wake.log",
        cpu_set=cpu_set,
        num_threads=num_threads,
    )

    summary = json.loads((single_dir / "summary.json").read_text())
    initial_speed = float(
        summary.get("converged_speed_mps") or summary["final_5_cycle_mean_ux_mps"]
    )
    _log(
        f"Corrected single complete; pair initial speed = {initial_speed:.9f} m/s.",
        queue_log,
    )

    for y_over_span in y_values:
        for x_over_span in x_values:
            label = _case_label(x_over_span, y_over_span)
            _log(
                f"Starting corrected pair case X/B={x_over_span}, "
                f"Y/B={y_over_span}, Z/B=0.",
                queue_log,
            )
            _run_logged(
                [
                    ".venv/bin/python",
                    "examples/free_flight_readme_scaled_flapping_two_flyer_streamwise.py",
                    "--output-root",
                    str(pair_root),
                    "--x-over-span",
                    str(x_over_span),
                    "--y-over-span",
                    str(y_over_span),
                    "--z-over-span",
                    "0.0",
                    "--prescribed-steps",
                    str(prescribed_steps),
                    "--free-steps",
                    str(free_steps),
                    "--steps-per-flap",
                    str(steps_per_flap),
                    "--initial-speed",
                    f"{initial_speed:.12f}",
                    "--save-every-n-steps",
                    "2",
                    "--dense-diagnostics-every-n-steps",
                    "1",
                ],
                pair_root / "logs" / f"{label}.log",
                cpu_set=cpu_set,
                num_threads=num_threads,
            )
            _log(
                f"Finished corrected pair case X/B={x_over_span}, "
                f"Y/B={y_over_span}, Z/B=0.",
                queue_log,
            )

    _log("All corrected in-phase single and pair cases complete.", queue_log)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--prescribed-steps", type=int, default=144)
    parser.add_argument("--free-steps", type=int, default=960)
    parser.add_argument("--steps-per-flap", type=int, default=48)
    parser.add_argument(
        "--x-over-span",
        type=float,
        nargs="+",
        default=DEFAULT_X_OVER_SPAN,
    )
    parser.add_argument(
        "--y-over-span",
        type=float,
        nargs="+",
        default=DEFAULT_Y_OVER_SPAN,
    )
    return parser.parse_args()


def main() -> None:
    """Run the queue."""
    args = parse_args()
    run_queue(
        base_dir=args.base_dir,
        cpu_set=args.cpu_set,
        num_threads=args.num_threads,
        prescribed_steps=args.prescribed_steps,
        free_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        x_values=tuple(args.x_over_span),
        y_values=tuple(args.y_over_span),
    )


if __name__ == "__main__":
    main()
