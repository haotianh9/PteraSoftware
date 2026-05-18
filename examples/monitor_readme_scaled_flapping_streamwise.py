"""Print live speed/load diagnostics for the README-scaled flapping run."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "free_flight_cases"
    / "readme_scaled_flapping_streamwise"
    / "free_x"
)


def _format_vector(values: np.ndarray, unit: str) -> str:
    """Return a compact vector string."""
    return f"[{values[0]: .6g}, {values[1]: .6g}, {values[2]: .6g}] {unit}"


def _latest_npz(directory: Path, pattern: str = "step_*.npz") -> Path | None:
    """Return the most recent matching snapshot path."""
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def print_latest(run_dir: Path, steps_per_flap: int, prescribed_steps: int) -> None:
    """Print the latest streamed solver and clamp diagnostics."""
    streamed_dir = run_dir / "streamed_history"
    dense_path = streamed_dir / "dense_speed_force_history.csv"
    if dense_path.exists():
        with dense_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if rows:
            row = rows[-1]
            print(f"dense_diagnostics: {dense_path}")
            print(
                f"global_step={int(row['global_step'])}  "
                f"t={float(row['time_over_flap_period']):.3f} periods  "
                f"free_phase_step={int(row['local_step'])}"
            )
            print(f"x={float(row['x_m']): .6g} m")
            print(f"Ux={float(row['ux_mps']): .6g} m/s")
            print(
                "raw_F_E="
                f"[{float(row['raw_fx_N']): .6g}, "
                f"{float(row['raw_fy_N']): .6g}, "
                f"{float(row['raw_fz_N']): .6g}] N"
            )
            print(
                "projected_F_E="
                f"[{float(row['projected_fx_N']): .6g}, "
                f"{float(row['projected_fy_N']): .6g}, "
                f"{float(row['projected_fz_N']): .6g}] N"
            )
            print(
                "clamp_F_yz="
                f"[{float(row['clamp_fy_N']): .6g}, "
                f"{float(row['clamp_fz_N']): .6g}] N"
            )
            print(
                "clamp_M_E="
                f"[{float(row['clamp_mx_Nm']): .6g}, "
                f"{float(row['clamp_my_Nm']): .6g}, "
                f"{float(row['clamp_mz_Nm']): .6g}] N m"
            )
            return

    solver_path = _latest_npz(streamed_dir)
    if solver_path is None:
        print(f"No solver snapshots found under {streamed_dir}")
        return

    with np.load(solver_path) as data:
        step = int(data["step"][0])
        period_time = step / steps_per_flap
        velocity_E = np.asarray(data["velocity_E__E"], dtype=float)
        speed = float(np.linalg.norm(velocity_E))
        position_E = np.asarray(data["position_E_E"], dtype=float)
        aero_forces_E = np.asarray(data["aero_forces_E"], dtype=float)
        net_forces_E = np.asarray(data["net_forces_before_gating_E"], dtype=float)
        gated_forces_E = np.asarray(data["forces_passed_to_mujoco_E"], dtype=float)

    print(f"solver_snapshot: {solver_path}")
    print(
        f"global_step={step}  t={period_time:.3f} periods  "
        f"free_phase_step={max(step - prescribed_steps, 0)}"
    )
    print(f"x={position_E[0]: .6g} m")
    print(f"U_E={_format_vector(velocity_E, 'm/s')}  |U|={speed:.6g} m/s")
    print(f"aero_F_E={_format_vector(aero_forces_E, 'N')}")
    print(f"net_F_E={_format_vector(net_forces_E, 'N')}")
    print(f"solver_gated_F_before_projection={_format_vector(gated_forces_E, 'N')}")

    clamp_dir = streamed_dir / "clamp_loads"
    clamp_path = _latest_npz(clamp_dir)
    if clamp_path is None:
        print(f"No clamp-load snapshots found yet under {clamp_dir}")
        return

    with np.load(clamp_path) as data:
        local_step = int(data["step"][0])
        raw_forces_E = np.asarray(data["raw_forces_E_N"], dtype=float)
        projected_forces_E = np.asarray(data["projected_forces_E_N"], dtype=float)
        clamp_forces_E = np.asarray(data["clamp_forces_E_N"], dtype=float)
        clamp_moments_E = np.asarray(data["clamp_moments_E_Cg_Nm"], dtype=float)

    print(f"clamp_snapshot: {clamp_path}")
    print(
        f"clamp_local_step={local_step}  "
        f"approx_global_step={prescribed_steps + local_step}"
    )
    print(f"raw_F_E={_format_vector(raw_forces_E, 'N')}")
    print(f"projected_F_E={_format_vector(projected_forces_E, 'N')}")
    print(f"clamp_F_E={_format_vector(clamp_forces_E, 'N')}")
    print(f"clamp_M_E={_format_vector(clamp_moments_E, 'N m')}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read live speed/load snapshots from a running flapping case."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run output directory containing streamed_history/.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=48,
        help="Steps per flapping period used by the run.",
    )
    parser.add_argument(
        "--prescribed-steps",
        type=int,
        default=144,
        help="Prescribed warmup steps before streamwise-free motion.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep printing until interrupted.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=60.0,
        help="Polling interval when --watch is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the monitor."""
    args = parse_args()
    while True:
        print_latest(
            run_dir=args.run_dir,
            steps_per_flap=args.steps_per_flap,
            prescribed_steps=args.prescribed_steps,
        )
        if not args.watch:
            return
        print("-" * 72, flush=True)
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
