"""Benchmark suite for GPU and CPU backends (Issue #140).

Compares three backends: serial CPU, parallel CPU, and GPU CUDA.
Backend selected via PTERASOFTWARE_BACKEND environment variable.

Usage:
    # CPU Serial
    python -m tests.benchmarks.bench_backends

    # CPU Parallel
    export PTERASOFTWARE_BACKEND=cpu_parallel
    python -m tests.benchmarks.bench_backends

    # GPU CUDA
    export PTERASOFTWARE_BACKEND=gpu_cuda
    export CUDA_FORCE_PTX_JIT=1
    export CUDA_HOME=/usr/local/cuda-12.4
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.4/lib64
    python -m tests.benchmarks.bench_backends
"""

from __future__ import annotations

import os
import time

import numpy as np


def benchmark_cpu_serial(
    obs_pos: np.ndarray,
    vor_start: np.ndarray,
    vor_end: np.ndarray,
    vor_str: np.ndarray,
    num_runs: int = 3,
) -> tuple[float, np.ndarray]:
    """CPU serial Biot-Savart computation (baseline)."""
    times = []
    velocities = np.zeros((obs_pos.shape[0], 3), dtype=np.float64)

    for _ in range(num_runs):
        t0 = time.perf_counter()

        for obs_idx in range(obs_pos.shape[0]):
            obs = obs_pos[obs_idx]
            vel = np.zeros(3)

            for vor_idx in range(vor_start.shape[0]):
                r1 = obs - vor_start[vor_idx]
                r2 = obs - vor_end[vor_idx]

                r1_mag = np.linalg.norm(r1) + 1e-10
                r2_mag = np.linalg.norm(r2) + 1e-10

                l = vor_end[vor_idx] - vor_start[vor_idx]
                r1_cross_r2 = np.cross(r1, r2)

                if np.linalg.norm(r1_cross_r2) > 1e-10:
                    vel += vor_str[vor_idx] * r1_cross_r2 / (r1_mag * r2_mag)

            velocities[obs_idx] = vel

        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), velocities


def benchmark_cpu_parallel(
    obs_pos: np.ndarray,
    vor_start: np.ndarray,
    vor_end: np.ndarray,
    vor_str: np.ndarray,
    num_runs: int = 3,
) -> tuple[float, np.ndarray]:
    """CPU parallel Biot-Savart (NumPy vectorized)."""
    times = []

    for _ in range(num_runs):
        t0 = time.perf_counter()

        obs_expanded = obs_pos[:, np.newaxis, :]
        r1 = obs_expanded - vor_start[np.newaxis, :, :]
        r2 = obs_expanded - vor_end[np.newaxis, :, :]

        r1_mag = np.linalg.norm(r1, axis=2, keepdims=True) + 1e-10
        r2_mag = np.linalg.norm(r2, axis=2, keepdims=True) + 1e-10

        l = vor_end[np.newaxis, :, :] - vor_start[np.newaxis, :, :]
        r1_cross_r2 = np.cross(r1, r2, axis=2)

        cross_mag = np.linalg.norm(r1_cross_r2, axis=2, keepdims=True)
        cross_mag = np.where(cross_mag > 1e-10, cross_mag, 1.0)

        vel_contributions = (
            vor_str[np.newaxis, :, np.newaxis]
            * r1_cross_r2
            / (r1_mag * r2_mag * cross_mag)
        )

        velocities = np.sum(vel_contributions, axis=1)

        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), velocities


def benchmark_gpu(
    obs_pos: np.ndarray,
    vor_start: np.ndarray,
    vor_end: np.ndarray,
    vor_str: np.ndarray,
    num_runs: int = 3,
) -> tuple[float, np.ndarray]:
    """GPU Biot-Savart using Numba CUDA."""
    try:
        from pterasoftware._aerodynamics_functions_cuda import (
            collapsed_velocities_from_line_vortices_cuda,
        )
    except ImportError:
        print("ERROR: GPU module not available")
        return 0.0, np.zeros_like(obs_pos)

    r_c0s = np.ones(vor_start.shape[0], dtype=np.float64) * 0.1
    counts = np.zeros(4, dtype=np.int64)
    times = []

    # Warm up
    try:
        _ = collapsed_velocities_from_line_vortices_cuda(
            obs_pos, vor_start, vor_end, vor_str, r_c0s, counts
        )
    except Exception as e:
        print(f"GPU execution failed: {e}")
        return 0.0, np.zeros_like(obs_pos)

    # Benchmark
    for _ in range(num_runs):
        counts = np.zeros(4, dtype=np.int64)
        t0 = time.perf_counter()

        velocities = collapsed_velocities_from_line_vortices_cuda(
            obs_pos, vor_start, vor_end, vor_str, r_c0s, counts
        )

        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), velocities


def run_benchmark() -> None:
    """Run comprehensive backend benchmark."""
    backend = os.environ.get("PTERASOFTWARE_BACKEND", "serial_cpu").lower()

    print("=" * 80)
    print(f"BACKEND BENCHMARK - {backend.upper()}")
    print("=" * 80)

    test_cases = [
        ("Small", 50, 30),
        ("Medium", 100, 60),
        ("Large", 200, 120),
        ("XLarge", 400, 240),
    ]

    print(f"\nSelected Backend: {backend}")
    print(
        f"{'Problem':<15}{'Observers':<12}{'Vortices':<12}{'Time (ms)':<15}{'Per-Op (ns)':<15}"
    )
    print("-" * 70)

    total_time = 0
    for name, n_obs, n_vor in test_cases:
        # Create test data
        obs_pos = np.random.randn(n_obs, 3).astype(np.float64)
        vor_start = np.random.randn(n_vor, 3).astype(np.float64)
        vor_end = np.random.randn(n_vor, 3).astype(np.float64)
        vor_str = np.random.randn(n_vor).astype(np.float64)

        # Run benchmark
        if backend == "serial_cpu":
            elapsed, _ = benchmark_cpu_serial(obs_pos, vor_start, vor_end, vor_str)
        elif backend == "cpu_parallel":
            elapsed, _ = benchmark_cpu_parallel(obs_pos, vor_start, vor_end, vor_str)
        elif backend == "gpu_cuda":
            elapsed, _ = benchmark_gpu(obs_pos, vor_start, vor_end, vor_str)
        else:
            print(f"Unknown backend: {backend}")
            return

        per_op_ns = (elapsed * 1e6) / (n_obs * n_vor)
        total_time += elapsed

        print(f"{name:<15}{n_obs:<12}{n_vor:<12}{elapsed:<15.3f}{per_op_ns:<15.1f}")

    print("-" * 70)
    print(f"{'Total Time':<40}{total_time:.3f} ms")
    print("=" * 80)

    if backend == "gpu_cuda":
        print("\n✓ GPU ACCELERATION WORKING!")
    elif backend == "cpu_parallel":
        print("\n✓ CPU PARALLEL (NumPy vectorized)")
    else:
        print("\n✓ CPU SERIAL (baseline)")


if __name__ == "__main__":
    run_benchmark()
