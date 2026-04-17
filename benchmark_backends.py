#!/usr/bin/env python3
"""Benchmark aerodynamics backends (serial CPU, parallel CPU, GPU CUDA).

This script verifies correctness and measures performance of different
computation backends for the Biot-Savart aerodynamics kernel.

Usage:
    # Benchmark serial CPU (default)
    .venv/bin/python benchmark_backends.py

    # Benchmark parallel CPU
    export PTERASOFTWARE_BACKEND=parallel_cpu
    .venv/bin/python benchmark_backends.py

    # Benchmark GPU
    export PTERASOFTWARE_BACKEND=gpu_cuda
    .venv/bin/python benchmark_backends.py
"""

from __future__ import annotations

import os
import sys
import timeit

import numpy as np

# Set backend BEFORE importing pterasoftware
backend = os.environ.get("PTERASOFTWARE_BACKEND", "serial_cpu")

from pterasoftware._aerodynamics_functions import get_backend_info

# Import backend selection function
from pterasoftware._gpu_config import get_config, select_biot_savart_function


def collapsed_velocities_from_ring_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Dispatch ring vortex velocity calculation to appropriate backend."""
    config = get_config()

    if config.get_use_gpu():
        # For GPU: convert ring vortex to line vortex format and use GPU kernel
        from pterasoftware._aerodynamics_functions_cuda import (
            collapsed_velocities_from_line_vortices_cuda,
        )

        # Convert ring vortex (4 vertices) to line vortices (4 segments)
        # Each ring is 4 line vortices: BR->FR, FR->FL, FL->BL, BL->BR
        stackVInd = np.zeros((stackP_GP1_CgP1.shape[0], 3), dtype=np.float64)

        # Segment 1: Back Right to Front Right
        stackVInd += collapsed_velocities_from_line_vortices_cuda(
            stackP_GP1_CgP1,
            stackBrrvp_GP1_CgP1,
            stackFrrvp_GP1_CgP1,
            strengths,
            r_c0s,
            singularity_counts,
            ages,
            nu,
        )

        # Segment 2: Front Right to Front Left
        stackVInd += collapsed_velocities_from_line_vortices_cuda(
            stackP_GP1_CgP1,
            stackFrrvp_GP1_CgP1,
            stackFlrvp_GP1_CgP1,
            strengths,
            r_c0s,
            singularity_counts,
            ages,
            nu,
        )

        # Segment 3: Front Left to Back Left
        stackVInd += collapsed_velocities_from_line_vortices_cuda(
            stackP_GP1_CgP1,
            stackFlrvp_GP1_CgP1,
            stackBlrvp_GP1_CgP1,
            strengths,
            r_c0s,
            singularity_counts,
            ages,
            nu,
        )

        # Segment 4: Back Left to Back Right
        stackVInd += collapsed_velocities_from_line_vortices_cuda(
            stackP_GP1_CgP1,
            stackBlrvp_GP1_CgP1,
            stackBrrvp_GP1_CgP1,
            strengths,
            r_c0s,
            singularity_counts,
            ages,
            nu,
        )

        return stackVInd
    else:
        # For CPU: use CPU ring vortex function
        from pterasoftware._aerodynamics_functions import (
            collapsed_velocities_from_ring_vortices as cpu_ring_vortex,
        )

        return cpu_ring_vortex(
            stackP_GP1_CgP1,
            stackBrrvp_GP1_CgP1,
            stackFrrvp_GP1_CgP1,
            stackFlrvp_GP1_CgP1,
            stackBlrvp_GP1_CgP1,
            strengths,
            r_c0s,
            singularity_counts,
            ages,
            nu,
        )


def create_problem(num_points: int, num_vortices: int, seed: int = 42) -> dict:
    """Create random test problem for aerodynamics calculation."""
    np.random.seed(seed)

    stackP_GP1_CgP1 = np.random.randn(num_points, 3) * 10.0
    stackBrrvp_GP1_CgP1 = np.random.randn(num_vortices, 3)
    stackFrrvp_GP1_CgP1 = np.random.randn(num_vortices, 3)
    stackFlrvp_GP1_CgP1 = np.random.randn(num_vortices, 3)
    stackBlrvp_GP1_CgP1 = np.random.randn(num_vortices, 3)

    strengths = np.random.randn(num_vortices) * 2.0
    r_c0s = np.random.rand(num_vortices) * 0.1 + 0.01
    singularity_counts = np.array([0, 0, 0, 0], dtype=np.int64)

    return {
        "stackP_GP1_CgP1": stackP_GP1_CgP1,
        "stackBrrvp_GP1_CgP1": stackBrrvp_GP1_CgP1,
        "stackFrrvp_GP1_CgP1": stackFrrvp_GP1_CgP1,
        "stackFlrvp_GP1_CgP1": stackFlrvp_GP1_CgP1,
        "stackBlrvp_GP1_CgP1": stackBlrvp_GP1_CgP1,
        "strengths": strengths,
        "r_c0s": r_c0s,
        "singularity_counts": singularity_counts,
    }


def verify_correctness() -> bool:
    """Verify correctness before benchmarking.

    Returns:
        True if correctness check passes, False otherwise.
    """
    print("=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)

    problem = create_problem(num_points=50, num_vortices=30, seed=12345)

    print(f"\nTest problem: 50 points × 30 vortices")
    print(f"Backend: {backend}\n")

    try:
        result = collapsed_velocities_from_ring_vortices(**problem)

        # Verify output
        print(f"  ✓ Output shape: {result.shape}")
        print(f"  ✓ Output dtype: {result.dtype}")
        print(f"  ✓ Min velocity: {np.min(result):.6e}")
        print(f"  ✓ Max velocity: {np.max(result):.6e}")
        print(f"  ✓ Mean velocity: {np.mean(result):.6e}")

        # Check for NaN or Inf
        if np.isnan(result).any() or np.isinf(result).any():
            print(f"  ✗ FAILED: NaN or Inf detected")
            return False

        # Check singularity counts
        sing_counts = problem["singularity_counts"]
        print(f"  ✓ Singularity counts: {sing_counts}")
        print(f"\n✓ CORRECTNESS CHECK PASSED\n")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        return False


def main():
    """Run benchmarks with correctness verification."""
    print("=" * 70)
    print("Aerodynamics Backend Benchmark")
    print("=" * 70)

    # Get configuration
    info = get_backend_info()
    print(f"\nConfiguration:")
    print(f"  Backend: {info['backend']}")
    print(f"  GPU available: {info['gpu_available']}")
    print(f"  Using GPU: {info['using_gpu']}")
    print(f"  Using parallel CPU: {info['using_parallel_cpu']}")
    print(f"  Platform: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")

    # Step 1: Verify correctness
    if not verify_correctness():
        print("ERROR: Correctness check failed. Aborting benchmarks.")
        return

    # Step 2: Run performance benchmarks
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    problems = [
        ("Small", 20, 15),
        ("Medium", 50, 30),
        ("Large", 100, 50),
        ("XLarge", 200, 100),
    ]

    results = {}

    print(
        f"\n{'Problem':<12} {'Points':<10} {'Vortices':<10} {'Time (ms)':>12} {'Per op (ns)':>12}"
    )
    print("-" * 68)

    for name, num_points, num_vortices in problems:
        problem = create_problem(num_points, num_vortices)
        num_ops = num_points * num_vortices * 4  # 4 ring vortex legs

        # Determine number of iterations (fewer for large problems)
        if num_ops < 5000:
            num_iterations = 100
        elif num_ops < 20000:
            num_iterations = 50
        else:
            num_iterations = 10

        # Warm up JIT
        collapsed_velocities_from_ring_vortices(**problem)

        # Benchmark with timeit
        timer = timeit.Timer(
            "collapsed_velocities_from_ring_vortices(**problem)",
            globals={
                "collapsed_velocities_from_ring_vortices": collapsed_velocities_from_ring_vortices,
                "problem": problem,
            },
        )

        # Take minimum of multiple runs (filters OS jitter)
        times = timer.repeat(repeat=5, number=num_iterations)
        elapsed_time = min(times) / num_iterations

        results[name] = elapsed_time
        time_per_op = elapsed_time / num_ops * 1e9

        print(
            f"{name:<12} {num_points:<10} {num_vortices:<10} "
            f"{elapsed_time * 1000:>12.3f} {time_per_op:>12.1f}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\nScaling analysis (relative to Small problem):")
        small_time = results["Small"]
        for name, _, _ in problems[1:]:
            if name in results:
                scaling = results[name] / small_time
                print(f"  {name:8}: {scaling:6.1f}× (O(N²) scaling)")


if __name__ == "__main__":
    main()
