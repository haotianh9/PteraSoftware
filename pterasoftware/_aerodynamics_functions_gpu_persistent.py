"""GPU-persistent memory management for unsteady aerodynamics simulations.

This module demonstrates zero-transfer GPU computation by keeping all work data on GPU
across timesteps. This is the key to achieving real GPU speedup in unsteady problems.

Key pattern: 1. Allocate all GPU memory upfront (wake vortices, velocities) 2. Update
data on GPU between timesteps (avoid CPU transfers) 3. Transfer results to CPU only at
simulation end
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    from numba import cuda
    from numba.cuda import jit as cuda_jit

    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Constants
_squire = 1.0e-4
_lamb = 1.25643
_eps = np.finfo(float).eps
_tol = 1.0e-10
_four_pi = 4.0 * math.pi
_four_lamb = 4.0 * _lamb


if CUDA_AVAILABLE:

    @cuda_jit
    def _collapsed_velocities_kernel_persistent(
        stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1,
        stackElvp_GP1_CgP1,
        strengths,
        r_c0s,
        num_valid_vortices,  # Track currently valid vortices (changes with timestep)
        stackVInd_GP1__E,
    ):
        """CUDA kernel for collapsed Biot-Savart with pre-allocated GPU arrays.

        :param stackP_GP1_CgP1: (N, 3) evaluation points on GPU
        :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices on GPU
        :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices on GPU
        :param strengths: (M,) vortex circulations on GPU
        :param r_c0s: (M,) initial core radii on GPU
        :param num_valid_vortices: scalar - how many vortices to include (changes per
            step)
        :param stackVInd_GP1__E: (N, 3) output velocities on GPU (pre-allocated)
        """
        point_id = cuda.grid(1)
        if point_id >= stackP_GP1_CgP1.shape[0]:
            return

        P_x = stackP_GP1_CgP1[point_id, 0]
        P_y = stackP_GP1_CgP1[point_id, 1]
        P_z = stackP_GP1_CgP1[point_id, 2]

        vel_x = 0.0
        vel_y = 0.0
        vel_z = 0.0

        # Only loop over valid vortices for this timestep
        for vortex_id in range(num_valid_vortices):
            Slvp_x = stackSlvp_GP1_CgP1[vortex_id, 0]
            Slvp_y = stackSlvp_GP1_CgP1[vortex_id, 1]
            Slvp_z = stackSlvp_GP1_CgP1[vortex_id, 2]

            Elvp_x = stackElvp_GP1_CgP1[vortex_id, 0]
            Elvp_y = stackElvp_GP1_CgP1[vortex_id, 1]
            Elvp_z = stackElvp_GP1_CgP1[vortex_id, 2]

            r1_x = Slvp_x - P_x
            r1_y = Slvp_y - P_y
            r1_z = Slvp_z - P_z

            r2_x = Elvp_x - P_x
            r2_y = Elvp_y - P_y
            r2_z = Elvp_z - P_z

            r3_x = r1_y * r2_z - r1_z * r2_y
            r3_y = r1_z * r2_x - r1_x * r2_z
            r3_z = r1_x * r2_y - r1_y * r2_x

            r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z)
            r2 = math.sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z)

            # Quick singularity checks
            r0_x = Elvp_x - Slvp_x
            r0_y = Elvp_y - Slvp_y
            r0_z = Elvp_z - Slvp_z
            r0 = math.sqrt(r0_x * r0_x + r0_y * r0_y + r0_z * r0_z)
            r0_times_tol = r0 * _tol

            if r1 < r0_times_tol or r2 < r0_times_tol:
                continue

            r3_sq = r3_x * r3_x + r3_y * r3_y + r3_z * r3_z
            r3 = math.sqrt(r3_sq)
            r1_times_r2 = r1 * r2
            c_3 = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z

            if r3 < (_tol * r1_times_r2):
                continue

            c_1 = strengths[vortex_id] / _four_pi
            r_c0 = r_c0s[vortex_id]
            r_c_sq = r_c0 * r_c0
            c_2 = r0 * r0 * r_c_sq

            c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))

            vel_x += c_4 * r3_x
            vel_y += c_4 * r3_y
            vel_z += c_4 * r3_z

        stackVInd_GP1__E[point_id, 0] = vel_x
        stackVInd_GP1__E[point_id, 1] = vel_y
        stackVInd_GP1__E[point_id, 2] = vel_z


class PersistentGPUUnsteadySolver:
    """GPU solver that keeps all data persistent across timesteps.

    This class demonstrates the optimization pattern for unsteady simulations: -
    Allocate GPU memory once at initialization - Update only changed data (new wake
    vortices) between timesteps - Never transfer data except at snapshot times - NO per-
    timestep GPU/CPU transfers

    Usage:     solver = PersistentGPUUnsteadySolver(         eval_points,
    bound_vortices, max_wake_vortices, num_timesteps     )     for step in
    range(num_timesteps):         new_wake = compute_shed_wake(step)  # On CPU
    solver.add_wake_vortices(step, new_wake)  # Transfer once solver.compute_step(step)
    # All on GPU, NO transfers     results = solver.get_results()  # Transfer final
    results only
    """

    def __init__(
        self,
        eval_points_cpu: np.ndarray,
        bound_starts_cpu: np.ndarray,
        bound_ends_cpu: np.ndarray,
        bound_strengths_cpu: np.ndarray,
        bound_r_c0s_cpu: np.ndarray,
        max_wake_vortices: int,
        num_timesteps: int,
    ):
        """Initialize GPU solver with pre-allocated persistent memory.

        :param eval_points_cpu: (N, 3) evaluation points (stays on GPU after transfer)
        :param bound_starts_cpu: (B, 3) bound vortex start vertices
        :param bound_ends_cpu: (B, 3) bound vortex end vertices
        :param bound_strengths_cpu: (B,) bound vortex circulations
        :param bound_r_c0s_cpu: (B,) bound vortex core radii
        :param max_wake_vortices: Maximum wake vortices to allocate
        :param num_timesteps: Number of timesteps to solve
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")

        self.num_points = eval_points_cpu.shape[0]
        self.num_bound = bound_starts_cpu.shape[0]
        self.max_wake = max_wake_vortices
        self.num_timesteps = num_timesteps
        self.num_valid_wake_per_step = []

        # Total vortices = bound + wake
        total_vortices = self.num_bound + max_wake_vortices

        # Transfer evaluation points to GPU (stays there)
        self.eval_points_gpu = cuda.to_device(eval_points_cpu)

        # Pre-allocate vortex arrays on GPU (will be filled)
        self.starts_gpu = cuda.device_array((total_vortices, 3), dtype=np.float64)
        self.ends_gpu = cuda.device_array((total_vortices, 3), dtype=np.float64)
        self.strengths_gpu = cuda.device_array(total_vortices, dtype=np.float64)
        self.r_c0s_gpu = cuda.device_array(total_vortices, dtype=np.float64)

        # Copy bound vortices to GPU (these never change)
        # Transfer bound vortices and copy into position
        bound_starts_full = np.zeros((total_vortices, 3), dtype=np.float64)
        bound_ends_full = np.zeros((total_vortices, 3), dtype=np.float64)
        bound_strengths_full = np.zeros(total_vortices, dtype=np.float64)
        bound_r_c0s_full = np.zeros(total_vortices, dtype=np.float64)

        bound_starts_full[: self.num_bound] = bound_starts_cpu
        bound_ends_full[: self.num_bound] = bound_ends_cpu
        bound_strengths_full[: self.num_bound] = bound_strengths_cpu
        bound_r_c0s_full[: self.num_bound] = bound_r_c0s_cpu

        self.starts_gpu = cuda.to_device(bound_starts_full)
        self.ends_gpu = cuda.to_device(bound_ends_full)
        self.strengths_gpu = cuda.to_device(bound_strengths_full)
        self.r_c0s_gpu = cuda.to_device(bound_r_c0s_full)

        # Pre-allocate output for all timesteps (optional, for snapshots)
        self.velocities_gpu_per_step = [
            cuda.device_array((self.num_points, 3), dtype=np.float64)
            for _ in range(num_timesteps)
        ]

    def add_wake_vortices(
        self,
        step: int,
        wake_starts: np.ndarray,
        wake_ends: np.ndarray,
        wake_strengths: np.ndarray,
        wake_r_c0s: np.ndarray,
    ) -> None:
        """Add wake vortices for a timestep (single transfer).

        Call this ONCE per timestep after computing wake advection on CPU. This is the
        ONLY CPU/GPU transfer per timestep.

        :param step: Timestep index
        :param wake_starts: (W, 3) wake vortex start positions
        :param wake_ends: (W, 3) wake vortex end positions
        :param wake_strengths: (W,) wake vortex circulations
        :param wake_r_c0s: (W,) wake vortex core radii
        """
        num_wake = wake_starts.shape[0]
        offset = self.num_bound

        # OPTIMIZED: Transfer only NEW wake data to GPU (not full arrays)
        # This avoids CPU↔GPU transfers of bound vortex data (unchanged)
        cuda.to_device(wake_starts, to=self.starts_gpu[offset : offset + num_wake])
        cuda.to_device(wake_ends, to=self.ends_gpu[offset : offset + num_wake])
        cuda.to_device(
            wake_strengths, to=self.strengths_gpu[offset : offset + num_wake]
        )
        cuda.to_device(wake_r_c0s, to=self.r_c0s_gpu[offset : offset + num_wake])

        # Track number of valid vortices for this timestep
        total_valid = self.num_bound + num_wake
        self.num_valid_wake_per_step.append(total_valid)

        cuda.synchronize()

    def compute_step(self, step: int) -> None:
        """Compute induced velocities for a timestep (entirely on GPU).

        NO CPU/GPU transfers during computation.

        :param step: Timestep index
        """
        if step >= len(self.num_valid_wake_per_step):
            raise ValueError(f"Wake vortices not added for step {step}")

        num_valid = self.num_valid_wake_per_step[step]

        # Launch kernel (all data already on GPU)
        threads_per_block = 256
        num_blocks = (self.num_points + threads_per_block - 1) // threads_per_block

        _collapsed_velocities_kernel_persistent[num_blocks, threads_per_block](
            self.eval_points_gpu,
            self.starts_gpu,
            self.ends_gpu,
            self.strengths_gpu,
            self.r_c0s_gpu,
            num_valid,
            self.velocities_gpu_per_step[step],
        )

        cuda.synchronize()

    def get_results(self, step: int | None = None) -> np.ndarray | list[np.ndarray]:
        """Get computation results (transfer to CPU once).

        :param step: If specified, return results for that timestep only. If None,
            return results for all timesteps.
        :return: (N, 3) velocities for one step, or list of (N, 3) for all steps
        """
        if step is None:
            # Transfer all results at once
            return [v.copy_to_host() for v in self.velocities_gpu_per_step]
        else:
            return self.velocities_gpu_per_step[step].copy_to_host()


def create_zero_transfer_solver(
    eval_points_cpu: np.ndarray,
    bound_starts_cpu: np.ndarray,
    bound_ends_cpu: np.ndarray,
    bound_strengths_cpu: np.ndarray,
    bound_r_c0s_cpu: np.ndarray,
    max_wake_vortices: int,
    num_timesteps: int,
) -> PersistentGPUUnsteadySolver | None:
    """Factory function to create a zero-transfer GPU solver.

    Returns None if CUDA unavailable.
    """
    if not CUDA_AVAILABLE:
        return None
    return PersistentGPUUnsteadySolver(
        eval_points_cpu,
        bound_starts_cpu,
        bound_ends_cpu,
        bound_strengths_cpu,
        bound_r_c0s_cpu,
        max_wake_vortices,
        num_timesteps,
    )
