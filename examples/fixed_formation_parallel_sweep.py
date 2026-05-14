"""Run fixed-formation energy sweeps with multiple worker processes."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from examples import free_flight_case_utils as ff_utils
    from examples import multibody_streamwise_stability_energy_sweep as sweep
except ImportError:
    import free_flight_case_utils as ff_utils
    import multibody_streamwise_stability_energy_sweep as sweep


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "streamwise_stability_energy"
    / "fixed_formation_parallel"
)


def _run_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one fixed-formation case inside a worker process."""
    numba_threads_per_worker = payload.get("numba_threads_per_worker")
    if numba_threads_per_worker is not None:
        numba.set_num_threads(int(numba_threads_per_worker))

    baseline_power_W = payload.get("baseline_power_W")
    if baseline_power_W is not None:
        baseline_power_W = np.asarray(baseline_power_W, dtype=float)
    output_dir = Path(payload["output_dir"])
    if bool(payload.get("clean_output_dir", False)) and output_dir.exists():
        shutil.rmtree(output_dir)
    save_every_n_steps = payload["save_every_n_steps"]
    history_save_dir = (
        output_dir / "streamed_history" if save_every_n_steps is not None else None
    )

    summary = sweep.run_streamwise_case(
        output_dir=output_dir,
        x_over_span=float(payload["x_over_span"]),
        y_over_span=float(payload["y_over_span"]),
        z_over_span=float(payload["z_over_span"]),
        prescribed_num_steps=int(payload["prescribed_num_steps"]),
        free_num_steps=int(payload["free_num_steps"]),
        time_step_s=float(payload["time_step_s"]),
        final_average_num_steps=int(payload["final_average_num_steps"]),
        show_progress=False,
        history_stride=int(payload["history_stride"]),
        save_every_n_steps=save_every_n_steps,
        history_save_dir=history_save_dir,
        render_wake_movie=False,
        baseline_power_W=baseline_power_W,
        compute_wbar=bool(payload["compute_wbar"]),
        aircraft_model=payload["aircraft_model"],
        angle_of_attack_deg=float(payload["angle_of_attack_deg"]),
        max_abs_x_over_span=float(payload["max_abs_x_over_span"]),
        max_abs_speed_mps=float(payload["max_abs_speed_mps"]),
        early_stop_converged=bool(payload["early_stop_converged"]),
        early_stop_min_time_s=float(payload["early_stop_min_time_s"]),
        early_stop_window_s=float(payload["early_stop_window_s"]),
        early_stop_rel_change_tol=float(payload["early_stop_rel_change_tol"]),
        early_stop_rel_slope_tol=float(payload["early_stop_rel_slope_tol"]),
        early_stop_check_every_n_steps=int(payload["early_stop_check_every_n_steps"]),
    )
    summary["_parallel_output_dir"] = payload["output_dir"]
    summary["_parallel_is_baseline"] = bool(payload.get("is_baseline", False))
    return summary


def _load_existing_summary(
    output_dir: Path,
    *,
    is_baseline: bool,
) -> dict[str, Any] | None:
    """Load a completed case summary for resumable sweeps."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    summary["_parallel_output_dir"] = str(output_dir)
    summary["_parallel_is_baseline"] = bool(is_baseline)
    return summary


def _add_baseline_deltas(
    summary: dict[str, Any],
    baseline_power_W: np.ndarray,
) -> dict[str, Any]:
    """Add baseline-relative fixed-formation power metrics to a summary."""
    if summary["run_status"] != "ok":
        return summary

    required_power_W = np.asarray(
        summary["final_window_mean_required_streamwise_power_W"],
        dtype=float,
    )
    delta_power_W = required_power_W - baseline_power_W
    summary["baseline_final_window_mean_required_power_W"] = baseline_power_W.tolist()
    summary["final_window_delta_power_vs_baseline_W"] = delta_power_W.tolist()
    summary["final_window_pair_mean_delta_power_vs_baseline_W"] = float(
        np.mean(delta_power_W)
    )

    output_dir = Path(summary["_parallel_output_dir"])
    json_summary = dict(summary)
    json_summary.pop("_parallel_output_dir", None)
    json_summary.pop("_parallel_is_baseline", None)
    ff_utils.write_json(output_dir / "summary.json", json_summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fixed-formation trim-load cases in parallel."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--x-over-span", type=str, required=True)
    parser.add_argument("--y-over-span", type=str, required=True)
    parser.add_argument("--z-over-span", type=str, required=True)
    parser.add_argument(
        "--angle-of-attack-deg",
        type=float,
        default=sweep.DEFAULT_ANGLE_OF_ATTACK_DEG,
    )
    parser.add_argument("--total-time-s", type=float, required=True)
    parser.add_argument("--prescribed-time-s", type=float, default=1.0)
    parser.add_argument("--time-step-s", type=float, default=sweep.DEFAULT_TIME_STEP_S)
    parser.add_argument("--final-average-window-s", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--numba-threads-per-worker",
        type=int,
        default=None,
        help=(
            "Numba threads per worker process. Defaults to "
            "max(1, os.cpu_count() // workers)."
        ),
    )
    parser.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help=("Multiprocessing start method. 'spawn' is safest with threaded kernels."),
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help=(
            "Maximum cases before a worker process is recycled. Use 0 to reuse "
            "workers for the full sweep after Numba kernels are warmed."
        ),
    )
    parser.add_argument("--history-stride", type=int, default=24)
    parser.add_argument("--save-every-n-steps", type=int, default=24)
    parser.add_argument(
        "--run-baseline", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--baseline-y-over-span", type=float, default=10.0)
    parser.add_argument(
        "--compute-wbar", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--resume-skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse case directories that already contain summary.json.",
    )
    parser.add_argument(
        "--clean-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove incomplete per-case output directories before rerunning them.",
    )
    parser.add_argument("--max-abs-x-over-span", type=float, default=20.0)
    parser.add_argument("--max-abs-speed-mps", type=float, default=50.0)
    parser.add_argument(
        "--early-stop-converged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Stop a case cleanly once rear-wing streamwise clamp thrust converges."),
    )
    parser.add_argument("--early-stop-min-time-s", type=float, default=12.0)
    parser.add_argument("--early-stop-window-s", type=float, default=2.0)
    parser.add_argument("--early-stop-rel-change-tol", type=float, default=0.01)
    parser.add_argument("--early-stop-rel-slope-tol", type=float, default=0.01)
    parser.add_argument("--early-stop-check-every-n-steps", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    """Run one parallel sweep."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.time_step_s <= 0.0:
        raise ValueError("--time-step-s must be positive.")
    if args.numba_threads_per_worker is not None and args.numba_threads_per_worker < 1:
        raise ValueError("--numba-threads-per-worker must be at least 1.")
    if args.max_tasks_per_child < 0:
        raise ValueError("--max-tasks-per-child must be non-negative.")
    if args.early_stop_check_every_n_steps < 1:
        raise ValueError("--early-stop-check-every-n-steps must be at least 1.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    numba_threads_per_worker = args.numba_threads_per_worker
    if numba_threads_per_worker is None:
        numba_threads_per_worker = max(1, cpu_count // args.workers)
    total_requested_threads = numba_threads_per_worker * args.workers
    print(
        "Parallel sweep configuration: "
        f"workers={args.workers}, numba_threads_per_worker={numba_threads_per_worker}, "
        f"total_requested_threads={total_requested_threads}, cpu_count={cpu_count}, "
        f"start_method={args.start_method}, "
        f"max_tasks_per_child={args.max_tasks_per_child}",
        flush=True,
    )

    x_values = sweep.parse_float_tuple(args.x_over_span)
    y_values = sweep.parse_float_tuple(args.y_over_span)
    z_values = sweep.parse_float_tuple(args.z_over_span)
    prescribed_num_steps = int(round(args.prescribed_time_s / args.time_step_s))
    total_num_steps = int(round(args.total_time_s / args.time_step_s))
    free_num_steps = max(1, total_num_steps - prescribed_num_steps)
    final_average_num_steps = max(
        1, int(round(args.final_average_window_s / args.time_step_s))
    )

    payloads: list[dict[str, Any]] = []
    cached_summaries: list[dict[str, Any]] = []

    def add_payload(payload: dict[str, Any]) -> None:
        output_dir = Path(payload["output_dir"])
        if args.resume_skip_existing:
            existing_summary = _load_existing_summary(
                output_dir,
                is_baseline=bool(payload.get("is_baseline", False)),
            )
            if existing_summary is not None:
                cached_summaries.append(existing_summary)
                return
        payload["clean_output_dir"] = bool(args.clean_incomplete)
        payloads.append(payload)

    if args.run_baseline:
        add_payload(
            {
                "output_dir": str(output_root / "baseline_far_lateral"),
                "x_over_span": 0.0,
                "y_over_span": args.baseline_y_over_span,
                "z_over_span": 0.0,
                "prescribed_num_steps": prescribed_num_steps,
                "free_num_steps": free_num_steps,
                "time_step_s": args.time_step_s,
                "final_average_num_steps": final_average_num_steps,
                "history_stride": args.history_stride,
                "save_every_n_steps": args.save_every_n_steps,
                "compute_wbar": args.compute_wbar,
                "aircraft_model": sweep.DEFAULT_AIRCRAFT_MODEL,
                "angle_of_attack_deg": args.angle_of_attack_deg,
                "max_abs_x_over_span": args.max_abs_x_over_span,
                "max_abs_speed_mps": args.max_abs_speed_mps,
                "early_stop_converged": args.early_stop_converged,
                "early_stop_min_time_s": args.early_stop_min_time_s,
                "early_stop_window_s": args.early_stop_window_s,
                "early_stop_rel_change_tol": args.early_stop_rel_change_tol,
                "early_stop_rel_slope_tol": args.early_stop_rel_slope_tol,
                "early_stop_check_every_n_steps": args.early_stop_check_every_n_steps,
                "numba_threads_per_worker": numba_threads_per_worker,
                "baseline_power_W": None,
                "is_baseline": True,
            }
        )
    for z_over_span, y_over_span, x_over_span in product(z_values, y_values, x_values):
        add_payload(
            {
                "output_dir": str(
                    output_root / sweep.run_label(x_over_span, y_over_span, z_over_span)
                ),
                "x_over_span": x_over_span,
                "y_over_span": y_over_span,
                "z_over_span": z_over_span,
                "prescribed_num_steps": prescribed_num_steps,
                "free_num_steps": free_num_steps,
                "time_step_s": args.time_step_s,
                "final_average_num_steps": final_average_num_steps,
                "history_stride": args.history_stride,
                "save_every_n_steps": args.save_every_n_steps,
                "compute_wbar": args.compute_wbar,
                "aircraft_model": sweep.DEFAULT_AIRCRAFT_MODEL,
                "angle_of_attack_deg": args.angle_of_attack_deg,
                "max_abs_x_over_span": args.max_abs_x_over_span,
                "max_abs_speed_mps": args.max_abs_speed_mps,
                "early_stop_converged": args.early_stop_converged,
                "early_stop_min_time_s": args.early_stop_min_time_s,
                "early_stop_window_s": args.early_stop_window_s,
                "early_stop_rel_change_tol": args.early_stop_rel_change_tol,
                "early_stop_rel_slope_tol": args.early_stop_rel_slope_tol,
                "early_stop_check_every_n_steps": args.early_stop_check_every_n_steps,
                "numba_threads_per_worker": numba_threads_per_worker,
                "baseline_power_W": None,
                "is_baseline": False,
            }
        )

    all_summaries: list[dict[str, Any]] = list(cached_summaries)
    summaries: list[dict[str, Any]] = [
        summary
        for summary in cached_summaries
        if not bool(summary.get("_parallel_is_baseline", False))
    ]
    if cached_summaries:
        print(
            f"Reusing {len(cached_summaries)} existing completed summaries.",
            flush=True,
        )
    total_payload_count = len(cached_summaries) + len(payloads)
    mp_context = mp.get_context(args.start_method)
    if payloads:
        max_tasks_per_child = (
            None if args.max_tasks_per_child == 0 else args.max_tasks_per_child
        )
        with mp_context.Pool(
            processes=args.workers,
            maxtasksperchild=max_tasks_per_child,
        ) as pool:
            for index, summary in enumerate(
                pool.imap_unordered(_run_case, payloads),
                start=len(cached_summaries) + 1,
            ):
                all_summaries.append(summary)
                if not summary["_parallel_is_baseline"]:
                    summaries.append(summary)
                print(
                    f"[{index}/{total_payload_count}] "
                    f"X/B={summary['x_over_span_initial']:.3g}, "
                    f"Y/B={summary['y_over_span_prescribed']:.3g}, "
                    f"Z/B={summary['z_over_span_prescribed']:.3g}: "
                    f"{summary['run_status']}",
                    flush=True,
                )

    baseline_summary = next(
        (summary for summary in all_summaries if summary["_parallel_is_baseline"]),
        None,
    )
    if baseline_summary is not None:
        baseline_power_W = np.asarray(
            baseline_summary["final_window_mean_required_streamwise_power_W"],
            dtype=float,
        )
        summaries = [
            _add_baseline_deltas(summary, baseline_power_W) for summary in summaries
        ]

    summaries.sort(
        key=lambda summary: (
            summary["z_over_span_prescribed"],
            summary["y_over_span_prescribed"],
            summary["x_over_span_initial"],
        )
    )
    csv_path = sweep.write_sweep_csv(output_root=output_root, summaries=summaries)
    map_paths = sweep.save_sweep_maps(output_root=output_root, summaries=summaries)
    theory_panel_path = sweep.save_theory_validation_9panel(
        output_root=output_root,
        summaries=summaries,
    )
    sweep_summary = {
        "case": "fixed_formation_parallel_energy_sweep",
        "aircraft_model": sweep.DEFAULT_AIRCRAFT_MODEL,
        "angle_of_attack_deg": args.angle_of_attack_deg,
        "prescribed_streamwise_speed_mps": sweep.DEFAULT_STREAMWISE_SPEED_MPS,
        "wake_model": "free",
        "prescribed_wake": False,
        "x_over_span_values": list(x_values),
        "y_over_span_values": list(y_values),
        "z_over_span_values": list(z_values),
        "span_m": sweep.FULL_SPAN_M,
        "root_chord_m": sweep.ROOT_CHORD_M,
        "tip_chord_m": sweep.TIP_CHORD_M,
        "total_time_s": args.total_time_s,
        "prescribed_time_s": args.prescribed_time_s,
        "time_step_s": args.time_step_s,
        "final_average_window_s": args.final_average_window_s,
        "workers": args.workers,
        "numba_threads_per_worker": numba_threads_per_worker,
        "max_tasks_per_child": args.max_tasks_per_child,
        "total_requested_threads": total_requested_threads,
        "cpu_count": cpu_count,
        "start_method": args.start_method,
        "early_stop_converged": args.early_stop_converged,
        "early_stop_min_time_s": args.early_stop_min_time_s,
        "early_stop_window_s": args.early_stop_window_s,
        "early_stop_rel_change_tol": args.early_stop_rel_change_tol,
        "early_stop_rel_slope_tol": args.early_stop_rel_slope_tol,
        "early_stop_check_every_n_steps": args.early_stop_check_every_n_steps,
        "baseline_summary": (
            None
            if baseline_summary is None
            else {
                key: value
                for key, value in baseline_summary.items()
                if not key.startswith("_parallel_")
            }
        ),
        "num_cases": len(summaries),
        "num_ok_cases": sum(summary["run_status"] == "ok" for summary in summaries),
        "num_skipped_initial_collision_cases": sum(
            summary["run_status"] == "skipped_initial_collision"
            for summary in summaries
        ),
        "sweep_summary_csv": str(csv_path),
        "sweep_maps": map_paths,
        "theory_validation_9panel": (
            None if theory_panel_path is None else str(theory_panel_path)
        ),
        "case_summaries": [
            {
                key: value
                for key, value in summary.items()
                if not key.startswith("_parallel_")
            }
            for summary in summaries
        ],
    }
    ff_utils.write_json(output_root / "sweep_summary.json", sweep_summary)


if __name__ == "__main__":
    main()
