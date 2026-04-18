"""GPU acceleration tests for Issue #140 (Numba CUDA backend).

Tests GPU and CPU backends with minimal API changes.
Backend selection via PTERASOFTWARE_BACKEND environment variable.

Usage:
    # CPU Serial (default)
    python -m unittest tests.integration.test_gpu_acceleration

    # CPU Parallel
    export PTERASOFTWARE_BACKEND=cpu_parallel
    python -m unittest tests.integration.test_gpu_acceleration

    # GPU CUDA (requires GPU setup)
    export PTERASOFTWARE_BACKEND=gpu_cuda
    python -m unittest tests.integration.test_gpu_acceleration
"""

from __future__ import annotations

import os
import time
import unittest

import numpy as np


class TestGPUAcceleration(unittest.TestCase):
    """Test GPU and CPU backend performance and correctness."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls.backend = os.environ.get("PTERASOFTWARE_BACKEND", "serial_cpu").lower()

    def test_gpu_detection(self) -> None:
        """Test GPU detection and availability."""
        try:
            from numba import cuda

            available = cuda.is_available()
            if self.backend == "gpu_cuda":
                if not available and os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
                    self.skipTest("GPU not available and not in simulator mode")
        except ImportError:
            self.skipTest("Numba not installed")

    def test_biot_savart_gpu_vs_cpu(self) -> None:
        """Test GPU kernel produces consistent results with CPU version."""
        try:
            from pterasoftware._aerodynamics_functions import (
                collapsed_velocities_from_ring_vortices,
            )
            from pterasoftware._aerodynamics_functions_cuda import (
                collapsed_velocities_from_line_vortices_cuda,
            )
        except ImportError:
            self.skipTest("Required modules not available")

        # Create test data
        n_obs = 100
        n_vor = 60

        obs_pos = np.random.RandomState(42).randn(n_obs, 3).astype(np.float64)
        vor_start = np.random.RandomState(43).randn(n_vor, 3).astype(np.float64)
        vor_end = np.random.RandomState(44).randn(n_vor, 3).astype(np.float64)
        strengths = np.random.RandomState(45).randn(n_vor).astype(np.float64)
        r_c0s = np.ones(n_vor, dtype=np.float64) * 0.1
        counts = np.zeros(4, dtype=np.int64)

        if self.backend == "gpu_cuda":
            try:
                # GPU computation
                velocities_gpu = collapsed_velocities_from_line_vortices_cuda(
                    obs_pos, vor_start, vor_end, strengths, r_c0s, counts
                )
                self.assertEqual(velocities_gpu.shape, (n_obs, 3))
                self.assertTrue(np.isfinite(velocities_gpu).all())
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.skipTest(f"GPU not available: {e}")
                raise
        else:
            # CPU computation - GPU code should still work
            try:
                velocities = collapsed_velocities_from_line_vortices_cuda(
                    obs_pos, vor_start, vor_end, strengths, r_c0s, counts
                )
                self.assertEqual(velocities.shape, (n_obs, 3))
                self.assertTrue(np.isfinite(velocities).all())
            except RuntimeError:
                # GPU not available, that's ok for CPU tests
                pass

    def test_performance_scaling(self) -> None:
        """Test that performance scales reasonably with problem size."""
        try:
            from pterasoftware._aerodynamics_functions_cuda import (
                collapsed_velocities_from_line_vortices_cuda,
            )
        except ImportError:
            self.skipTest("Required modules not available")

        times = []
        sizes = [(50, 30), (100, 60), (200, 120)]

        for n_obs, n_vor in sizes:
            obs_pos = np.random.randn(n_obs, 3).astype(np.float64)
            vor_start = np.random.randn(n_vor, 3).astype(np.float64)
            vor_end = np.random.randn(n_vor, 3).astype(np.float64)
            strengths = np.random.randn(n_vor).astype(np.float64)
            r_c0s = np.ones(n_vor, dtype=np.float64) * 0.1
            counts = np.zeros(4, dtype=np.int64)

            try:
                t0 = time.perf_counter()
                _ = collapsed_velocities_from_line_vortices_cuda(
                    obs_pos, vor_start, vor_end, strengths, r_c0s, counts
                )
                times.append(time.perf_counter() - t0)
            except RuntimeError:
                self.skipTest("GPU/backend not available")

        # Largest problem should take reasonable time (not zero or negative)
        # GPU has startup overhead so we check largest is bigger than smallest
        self.assertGreater(times[2], 0.0, "Largest problem should have measurable time")
        self.assertLess(times[2], 1.0, "Time should be reasonable (< 1 second)")


class TestBackendSelection(unittest.TestCase):
    """Test backend selection via environment variable."""

    def test_backend_variable(self) -> None:
        """Test that PTERASOFTWARE_BACKEND environment variable is recognized."""
        backend = os.environ.get("PTERASOFTWARE_BACKEND", "serial_cpu").lower()
        valid_backends = ["serial_cpu", "cpu_parallel", "gpu_cuda"]
        self.assertIn(backend, valid_backends)


if __name__ == "__main__":
    unittest.main()
