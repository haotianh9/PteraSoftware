"""Run corrected README-scaled pair cases two at a time after the single case."""

from __future__ import annotations

import argparse
import concurrent.futures
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
DEFAULT_CPU_SETS = ("0-9", "10-19")


def _log(message: str, log_path: Path) -> None:
    """Write a timestamped queue message."""
    stamp = time.strftime("%a %b %d %H:%M:%S %Z %Y")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a") as log_file:
        log_file.write(line + "\n")


def _thread_env(num_threads: int) -> dict[str, str]:
    """Return an environment configured for one worker."""
    env = os.environ.copy()
    value = str(num_threads)
    env.update(
        {
            "OMP_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "NUMBA_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
        }
    )
    return env


def _case_label(x_over_span: float, y_over_span: float) -> str:
    """Create a stable label for one pair case."""
    return f"xB_{x_over_span:0.3f}_yB_{y_over_span:0.3f}_zB_0".replace(".", "p")


def _case_dir_name(x_over_span: float, y_over_span: float) -> str:
    """Create the two-flyer script's output directory label."""

    def label(prefix: str, value: float) -> str:
        sign = "m" if value < 0.0 else "p"
        return f"{prefix}_{sign}{abs(value):.3f}".replace(".", "p")

    return "_".join(
        (label("xB", x_over_span), label("yB", y_over_span), label("zB", 0.0))
    )


def _case_is_converged(pair_root: Path, x_over_span: float, y_over_span: float) -> bool:
    """Return whether a case summary says distance convergence has been reached."""
    summary_path = pair_root / _case_dir_name(x_over_span, y_over_span) / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(summary.get("distance_convergence", {}).get("converged", False))


def _case_status(pair_root: Path, x_over_span: float, y_over_span: float) -> str | None:
    """Return the recorded case status, if a summary exists."""
    summary_path = pair_root / _case_dir_name(x_over_span, y_over_span) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    return summary.get("status")


def _mark_case_failed(
    *,
    pair_root: Path,
    x_over_span: float,
    y_over_span: float,
    error: str,
    case_log_path: Path,
) -> None:
    """Write a minimal failed summary if the child process could not do it."""
    case_dir = pair_root / _case_dir_name(x_over_span, y_over_span)
    summary_path = case_dir / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            if existing.get("status") == "failed":
                return
        except json.JSONDecodeError:
            pass
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "case": "readme_scaled_flapping_two_flyer_streamwise_free_x_restartable",
        "status": "failed",
        "body_convention": "body_1_front_body_2_rear",
        "x_over_span_initial": x_over_span,
        "y_over_span_initial": y_over_span,
        "z_over_span_initial": 0.0,
        "error": error,
        "case_log": str(case_log_path),
        "distance_convergence": {
            "converged": False,
            "reason": "case_failed",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _wait_for_single(single_dir: Path, log_path: Path) -> float:
    """Wait until the corrected single case writes its summary and return speed."""
    summary_path = single_dir / "summary.json"
    while not summary_path.exists():
        _log(f"Waiting for single summary at {summary_path}.", log_path)
        time.sleep(60)
    summary = json.loads(summary_path.read_text())
    speed = float(
        summary.get("converged_speed_mps") or summary["final_5_cycle_mean_ux_mps"]
    )
    _log(f"Single summary found; pair initial speed = {speed:.9f} m/s.", log_path)
    return speed


def _run_pair_case(
    *,
    pair_root: Path,
    x_over_span: float,
    y_over_span: float,
    initial_speed: float,
    cpu_set: str,
    num_threads: int,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    extend_until_converged: bool,
    extension_periods: int,
    max_free_periods: int,
    distance_slope_tol_xb_per_period: float,
) -> None:
    """Run one corrected pair case."""
    repo_root = Path(__file__).resolve().parents[1]
    label = _case_label(x_over_span, y_over_span)
    log_path = pair_root / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        cpu_set,
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
    ]
    if extend_until_converged:
        command.extend(
            [
                "--extend-until-distance-converged",
                "--extension-periods",
                str(extension_periods),
                "--max-free-periods",
                str(max_free_periods),
                "--distance-slope-tol-xb-per-period",
                str(distance_slope_tol_xb_per_period),
            ]
        )
    with log_path.open("w") as log_file:
        subprocess.run(
            command,
            cwd=repo_root,
            env=_thread_env(num_threads),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_parallel_queue(
    *,
    base_dir: Path,
    cpu_sets: tuple[str, ...],
    num_threads_per_worker: int,
    prescribed_steps: int,
    free_steps: int,
    steps_per_flap: int,
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    resume_existing: bool,
    extend_unconverged: bool,
    skip_converged: bool,
    extension_periods: int,
    max_free_periods: int,
    distance_slope_tol_xb_per_period: float,
) -> None:
    """Run all pair cases with bounded two-worker concurrency."""
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "corrected_inphase_pair_parallel_runner.log"
    single_dir = base_dir / "single_free_x_20p_resolved_wake"
    pair_root = (
        base_dir / "two_flyer_free_x_grid_y0_y025_y05_y075_y1_z0_20p_resolved_wake"
    )
    pair_root.mkdir(parents=True, exist_ok=True)
    (pair_root / "logs").mkdir(exist_ok=True)

    initial_speed = _wait_for_single(single_dir, log_path)
    cases = [(x, y) for y in y_values for x in x_values]
    _log(
        f"Starting {len(cases)} pair cases with {len(cpu_sets)} workers, "
        f"cpu_sets={cpu_sets}, threads_per_worker={num_threads_per_worker}.",
        log_path,
    )

    next_case_index = 0
    available_cpu_sets = list(cpu_sets)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpu_sets)) as executor:
        futures: dict[concurrent.futures.Future[None], tuple[float, float, str]] = {}
        while next_case_index < len(cases) or futures:
            while next_case_index < len(cases) and available_cpu_sets:
                cpu_set = available_cpu_sets.pop(0)
                x_over_span, y_over_span = cases[next_case_index]
                next_case_index += 1
                if _case_status(pair_root, x_over_span, y_over_span) == "failed":
                    _log(
                        f"Skipping failed pair X/B={x_over_span}, "
                        f"Y/B={y_over_span}, Z/B=0. Delete its summary to rerun.",
                        log_path,
                    )
                    available_cpu_sets.append(cpu_set)
                    continue
                if skip_converged and _case_is_converged(
                    pair_root, x_over_span, y_over_span
                ):
                    _log(
                        f"Skipping converged pair X/B={x_over_span}, "
                        f"Y/B={y_over_span}, Z/B=0.",
                        log_path,
                    )
                    available_cpu_sets.append(cpu_set)
                    continue
                _log(
                    f"Starting pair X/B={x_over_span}, Y/B={y_over_span}, "
                    f"Z/B=0 on CPUs {cpu_set}.",
                    log_path,
                )
                future = executor.submit(
                    _run_pair_case,
                    pair_root=pair_root,
                    x_over_span=x_over_span,
                    y_over_span=y_over_span,
                    initial_speed=initial_speed,
                    cpu_set=cpu_set,
                    num_threads=num_threads_per_worker,
                    prescribed_steps=prescribed_steps,
                    free_steps=free_steps,
                    steps_per_flap=steps_per_flap,
                    extend_until_converged=extend_unconverged or resume_existing,
                    extension_periods=extension_periods,
                    max_free_periods=max_free_periods,
                    distance_slope_tol_xb_per_period=distance_slope_tol_xb_per_period,
                )
                futures[future] = (x_over_span, y_over_span, cpu_set)

            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                x_over_span, y_over_span, cpu_set = futures.pop(future)
                available_cpu_sets.append(cpu_set)
                label = _case_label(x_over_span, y_over_span)
                case_log_path = pair_root / "logs" / f"{label}.log"
                try:
                    future.result()
                except subprocess.CalledProcessError as error:
                    _mark_case_failed(
                        pair_root=pair_root,
                        x_over_span=x_over_span,
                        y_over_span=y_over_span,
                        error=f"child process exited with {error.returncode}",
                        case_log_path=case_log_path,
                    )
                    _log(
                        f"FAILED pair X/B={x_over_span}, Y/B={y_over_span}, "
                        f"Z/B=0 from CPUs {cpu_set}; see {case_log_path}.",
                        log_path,
                    )
                    continue
                except Exception as error:
                    _mark_case_failed(
                        pair_root=pair_root,
                        x_over_span=x_over_span,
                        y_over_span=y_over_span,
                        error=str(error),
                        case_log_path=case_log_path,
                    )
                    _log(
                        f"FAILED pair X/B={x_over_span}, Y/B={y_over_span}, "
                        f"Z/B=0 from CPUs {cpu_set}: {error}; see {case_log_path}.",
                        log_path,
                    )
                    continue
                _log(
                    f"Finished pair X/B={x_over_span}, Y/B={y_over_span}, "
                    f"Z/B=0 from CPUs {cpu_set}.",
                    log_path,
                )

    _log("All corrected in-phase pair cases complete.", log_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--cpu-sets", nargs="+", default=DEFAULT_CPU_SETS)
    parser.add_argument("--num-threads-per-worker", type=int, default=10)
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
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--extend-unconverged", action="store_true")
    parser.add_argument("--skip-converged", action="store_true")
    parser.add_argument("--extension-periods", type=int, default=10)
    parser.add_argument("--max-free-periods", type=int, default=80)
    parser.add_argument("--distance-slope-tol-xb-per-period", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    """Run the pair queue."""
    args = parse_args()
    run_parallel_queue(
        base_dir=args.base_dir,
        cpu_sets=tuple(args.cpu_sets),
        num_threads_per_worker=args.num_threads_per_worker,
        prescribed_steps=args.prescribed_steps,
        free_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        x_values=tuple(args.x_over_span),
        y_values=tuple(args.y_over_span),
        resume_existing=args.resume_existing,
        extend_unconverged=args.extend_unconverged,
        skip_converged=args.skip_converged,
        extension_periods=args.extension_periods,
        max_free_periods=args.max_free_periods,
        distance_slope_tol_xb_per_period=args.distance_slope_tol_xb_per_period,
    )


if __name__ == "__main__":
    main()
