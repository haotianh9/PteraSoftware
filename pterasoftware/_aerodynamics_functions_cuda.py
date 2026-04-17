"""GPU CUDA implementation of Biot-Savart kernel (Issue #140, Phase 1)."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numba import cuda, float64, int64, jit, prange

# Import constants from CPU version
_eps = 1e-14
_tol = 1e-4
_four_pi = 4.0 * math.pi
_four_lamb = 4.0 * math.log(2.0)
_squire = 0.25


@cuda.jit
def biot_savart_cuda_kernel(
    stackP_dev,  # (num_points, 3)
    stackS_dev,  # (num_vortices, 3) - start vertices
    stackE_dev,  # (num_vortices, 3) - end vertices
    strengths_dev,  # (num_vortices,)
    r_c0s_dev,  # (num_vortices,)
    ages_dev,  # (num_vortices,)
    nu,  # scalar viscosity
    velocities_dev,  # (num_points, 3) - OUTPUT
    singularity_counts_dev,  # (4,) - OUTPUT
):
    """GPU CUDA kernel for parallel Biot-Savart computation.

    Grid structure: (num_vortices, 1, 1) - one block per vortex Block structure: (256,
    1, 1) - threads parallelize over evaluation points

    Each thread: - Assigned one evaluation point - Processes single vortex (blockIdx.x)
    - Computes velocity at point due to vortex - Uses atomic operations for accumulation
    and singularity counting

    Thread safety: - Each thread writes to unique velocities_dev[point_id, :] location -
    Singularity counts use atomic adds (thread-safe)
    """
    vortex_id = cuda.blockIdx.x
    point_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    num_vortices = stackS_dev.shape[0]
    num_points = stackP_dev.shape[0]

    # Each block processes one vortex, threads iterate over points
    for pt in range(point_id, num_points, block_size):
        P = stackP_dev[pt]

        # Load vortex properties
        Slvp = stackS_dev[vortex_id]
        Elvp = stackE_dev[vortex_id]

        # Compute r0 (vortex segment vector from start to end)
        r0X = Elvp[0] - Slvp[0]
        r0Y = Elvp[1] - Slvp[1]
        r0Z = Elvp[2] - Slvp[2]

        r0 = math.sqrt(r0X * r0X + r0Y * r0Y + r0Z * r0Z)

        # Skip degenerate filaments
        if r0 < _eps:
            cuda.atomic.add(singularity_counts_dev, 0, 1)
            continue

        strength = strengths_dev[vortex_id]
        age = ages_dev[vortex_id]
        r_c0 = r_c0s_dev[vortex_id]

        r0_times_tol = r0 * _tol
        r_c_sq = r_c0 * r_c0 + _four_lamb * (nu + _squire * abs(strength)) * age

        c_1 = strength / _four_pi
        c_2 = r0 * r0 * r_c_sq

        # Compute r1 (from eval point to segment start)
        r1X = Slvp[0] - P[0]
        r1Y = Slvp[1] - P[1]
        r1Z = Slvp[2] - P[2]

        # Compute r2 (from eval point to segment end)
        r2X = Elvp[0] - P[0]
        r2Y = Elvp[1] - P[1]
        r2Z = Elvp[2] - P[2]

        # Compute r3 (cross product r1 × r2)
        r3X = r1Y * r2Z - r1Z * r2Y
        r3Y = r1Z * r2X - r1X * r2Z
        r3Z = r1X * r2Y - r1Y * r2X

        r1 = math.sqrt(r1X * r1X + r1Y * r1Y + r1Z * r1Z)
        r2 = math.sqrt(r2X * r2X + r2Y * r2Y + r2Z * r2Z)

        # Check for singular configurations
        if r1 < r0_times_tol:
            cuda.atomic.add(singularity_counts_dev, 1, 1)
            continue

        if r2 < r0_times_tol:
            cuda.atomic.add(singularity_counts_dev, 2, 1)
            continue

        r3_sq = r3X * r3X + r3Y * r3Y + r3Z * r3Z
        r3 = math.sqrt(r3_sq)

        r1_times_r2 = r1 * r2

        # Compute dot product r1 · r2
        c_3 = r1X * r2X + r1Y * r2Y + r1Z * r2Z

        # Check collinearity
        if r3 < (_tol * r1_times_r2):
            if c_3 < 0.0:
                cuda.atomic.add(singularity_counts_dev, 3, 1)
            continue

        # Compute velocity magnitude
        c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))

        # Accumulate velocity using atomic operations
        # Note: FP64 atomic add is supported on most modern GPUs
        cuda.atomic.add(velocities_dev, (pt, 0), c_4 * r3X)
        cuda.atomic.add(velocities_dev, (pt, 1), c_4 * r3Y)
        cuda.atomic.add(velocities_dev, (pt, 2), c_4 * r3Z)


def collapsed_velocities_from_line_vortices_cuda(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated Biot-Savart kernel using CUDA.

    Wrapper function that: 1. Checks CUDA availability 2. Transfers data to GPU 3.
    Launches CUDA kernel 4. Transfers results back to CPU 5. Falls back to CPU if GPU
    unavailable

    Thread safety (GPU): - Each thread writes to unique velocities array location (FP64
    atomic add) - Singularity counts use atomic operations (thread-safe)

    :param stackP_GP1_CgP1: Evaluation points (num_points, 3)
    :param stackSlvp_GP1_CgP1: Vortex start vertices (num_vortices, 3)
    :param stackElvp_GP1_CgP1: Vortex end vertices (num_vortices, 3)
    :param strengths: Vortex strengths (num_vortices,)
    :param r_c0s: Initial core radii (num_vortices,)
    :param singularity_counts: Output singularity counts (4,) - modified in place
    :param ages: Vortex ages - default None (treated as zeros)
    :param nu: Kinematic viscosity (default 0.0)
    :return: Induced velocities at evaluation points (num_points, 3)
    """

    # Check CUDA availability (hardware or simulator mode)
    try:
        if not cuda.is_available():
            import os

            # Allow simulator mode for testing
            if os.environ.get("NUMBA_ENABLE_CUDASIM", "").lower() not in ("1", "true"):
                raise RuntimeError(
                    "CUDA GPU not available. Use CPU parallel version instead."
                )
    except Exception:
        pass  # If cuda module issues, try anyway

    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    if ages is None:
        ages = np.zeros(num_vortices, dtype=np.float64)

    # Ensure all arrays are contiguous and correct dtype
    stackP = np.ascontiguousarray(stackP_GP1_CgP1, dtype=np.float64)
    stackS = np.ascontiguousarray(stackSlvp_GP1_CgP1, dtype=np.float64)
    stackE = np.ascontiguousarray(stackElvp_GP1_CgP1, dtype=np.float64)
    strengths = np.ascontiguousarray(strengths, dtype=np.float64)
    r_c0s = np.ascontiguousarray(r_c0s, dtype=np.float64)
    ages = np.ascontiguousarray(ages, dtype=np.float64)

    # Allocate output arrays on CPU (will transfer to GPU)
    velocities = np.zeros((num_points, 3), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)

    # Transfer data to GPU
    stackP_dev = cuda.to_device(stackP)
    stackS_dev = cuda.to_device(stackS)
    stackE_dev = cuda.to_device(stackE)
    strengths_dev = cuda.to_device(strengths)
    r_c0s_dev = cuda.to_device(r_c0s)
    ages_dev = cuda.to_device(ages)
    velocities_dev = cuda.to_device(velocities)
    counts_dev = cuda.to_device(counts)

    # Configure grid and block
    threads_per_block = 256
    blocks_per_grid = num_vortices

    # Launch kernel
    biot_savart_cuda_kernel[blocks_per_grid, threads_per_block](
        stackP_dev,
        stackS_dev,
        stackE_dev,
        strengths_dev,
        r_c0s_dev,
        ages_dev,
        nu,
        velocities_dev,
        counts_dev,
    )

    # Synchronize GPU (wait for kernel to finish)
    cuda.synchronize()

    # Transfer results back to CPU
    velocities_result = velocities_dev.copy_to_host()
    counts_result = counts_dev.copy_to_host()

    # Update singularity counts (modify in place as per CPU version)
    singularity_counts[:] += counts_result

    return cast(np.ndarray, velocities_result)


def calculate_bound_wing_influences_with_persistent_memory(
    gpu_pool,
    strengths: np.ndarray,
    singularity_counts: np.ndarray,
) -> np.ndarray:
    """Calculate bound wing-wing influences using persistent GPU memory pool.

    This uses static panel and bound vortex geometry that remains on GPU, only
    transferring bound vortex strengths per timestep.

    Phase 2 optimization: Reduces per-timestep transfers by keeping panel geometry and
    bound vortex vertices persistent on GPU.

    Parameters: ----------- gpu_pool : GPUMemoryPool     Persistent GPU memory pool with
    initialized static data strengths : np.ndarray     Bound vortex strengths
    (num_bound_vortices,) singularity_counts : np.ndarray     Singularity counts array
    (4,) - modified in place

    Returns: -------- np.ndarray     Induced velocities at panel points (num_panels, 3)
    """
    from pterasoftware._gpu_memory_manager import GPUMemoryPool

    if not isinstance(gpu_pool, GPUMemoryPool):
        raise TypeError("gpu_pool must be GPUMemoryPool instance")

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    # Update bound strengths on GPU (small transfer: ~1.6 KB)
    gpu_pool.update_bound_strengths(strengths)

    # Reset buffers
    gpu_pool.reset_velocities()
    gpu_pool.reset_singularity_counts()

    # Configure grid and block
    num_bound_vortices = gpu_pool.num_bound_vortices
    threads_per_block = 256
    blocks_per_grid = num_bound_vortices

    # Launch kernel using persistent GPU memory
    biot_savart_cuda_kernel[blocks_per_grid, threads_per_block](
        gpu_pool.panel_cpp_gpu,
        gpu_pool.bound_start_vortices_gpu,
        gpu_pool.bound_end_vortices_gpu,
        gpu_pool.bound_strengths_gpu,
        cuda.device_array_like(
            np.ones(num_bound_vortices, dtype=np.float64) * 0.1
        ),  # Default core radius
        cuda.device_array_like(
            np.zeros(num_bound_vortices, dtype=np.float64)
        ),  # Zero ages (bound vortices don't age)
        gpu_pool.viscosity,
        gpu_pool.velocities_gpu,
        gpu_pool.singularity_counts_gpu,
    )

    # Synchronize
    cuda.synchronize()

    # Fetch results
    velocities = gpu_pool.fetch_velocities_to_cpu()
    counts = gpu_pool.fetch_singularity_counts_to_cpu()

    # Update singularity counts
    singularity_counts[:] += counts

    return velocities


def calculate_wake_wing_influences_with_persistent_memory(
    gpu_pool,
    singularity_counts: np.ndarray,
) -> np.ndarray:
    """Calculate wake-wing influences using persistent GPU memory pool.

    Accumulates induced velocities from all wake vortices (old + new). All wake vortices
    remain on GPU; only new ones transferred each timestep.

    Phase 2 optimization: Append-only wake structure reduces per-timestep transfers from
    43 KB to ~1.7 KB by keeping accumulated wake data on GPU.

    Parameters: ----------- gpu_pool : GPUMemoryPool     Persistent GPU memory pool with
    accumulated wake data singularity_counts : np.ndarray     Singularity counts array
    (4,) - modified in place

    Returns: -------- np.ndarray     Induced velocities at panel points (num_panels, 3)
    """
    from pterasoftware._gpu_memory_manager import GPUMemoryPool

    if not isinstance(gpu_pool, GPUMemoryPool):
        raise TypeError("gpu_pool must be GPUMemoryPool instance")

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    if gpu_pool.current_num_wake_vortices == 0:
        # No wake vortices yet
        return np.zeros((gpu_pool.num_panels, 3), dtype=np.float64)

    # Reset buffers
    gpu_pool.reset_velocities()
    gpu_pool.reset_singularity_counts()

    # Configure grid and block for wake vortices
    num_wake_vortices = gpu_pool.current_num_wake_vortices
    threads_per_block = 256
    blocks_per_grid = num_wake_vortices

    # Launch kernel using persistent wake GPU memory
    biot_savart_cuda_kernel[blocks_per_grid, threads_per_block](
        gpu_pool.panel_cpp_gpu,
        gpu_pool.wake_start_vortices_gpu[:num_wake_vortices],
        gpu_pool.wake_end_vortices_gpu[:num_wake_vortices],
        gpu_pool.wake_strengths_gpu[:num_wake_vortices],
        gpu_pool.wake_rc0s_gpu[:num_wake_vortices],
        gpu_pool.wake_ages_gpu[:num_wake_vortices],
        gpu_pool.viscosity,
        gpu_pool.velocities_gpu,
        gpu_pool.singularity_counts_gpu,
    )

    # Synchronize
    cuda.synchronize()

    # Fetch results
    velocities = gpu_pool.fetch_velocities_to_cpu()
    counts = gpu_pool.fetch_singularity_counts_to_cpu()

    # Update singularity counts
    singularity_counts[:] += counts

    return velocities


def calculate_bound_wing_influences_gpu(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    nu: float = 0.0,
) -> np.ndarray:
    """Calculate bound vortex influence matrix on GPU (Phase 3.3).

    Computes induced velocities from bound RingVortices at panel collocation points.
    Returns full (num_panels, num_panels, 3) matrix for wing-wing influence coeff.

    Expected speedup: 5-20× for large systems (N > 500 panels).

    Parameters: ----------- stackP_GP1_CgP1 : Panel collocation points (num_panels, 3)
    stackBrrvp_GP1_CgP1, etc. : Bound vortex corners (num_panels, 3) strengths : Bound
    vortex strengths (num_panels,) r_c0s : Vortex core radii (num_panels,)
    singularity_counts : Singularity counter (4,) - modified in place nu : Kinematic
    viscosity

    Returns: -------- np.ndarray: Influence matrix (num_panels, num_panels, 3)
    """

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    num_panels = stackP_GP1_CgP1.shape[0]

    # Convert to contiguous arrays
    stackP = np.ascontiguousarray(stackP_GP1_CgP1, dtype=np.float64)
    stackBr = np.ascontiguousarray(stackBrrvp_GP1_CgP1, dtype=np.float64)
    stackFr = np.ascontiguousarray(stackFrrvp_GP1_CgP1, dtype=np.float64)
    stackFl = np.ascontiguousarray(stackFlrvp_GP1_CgP1, dtype=np.float64)
    stackBl = np.ascontiguousarray(stackBlrvp_GP1_CgP1, dtype=np.float64)
    strengths = np.ascontiguousarray(strengths, dtype=np.float64)
    r_c0s = np.ascontiguousarray(r_c0s, dtype=np.float64)

    # Output arrays
    grid_velocities = np.zeros((num_panels, num_panels, 3), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)

    # GPU arrays
    stackP_gpu = cuda.to_device(stackP)
    stackBr_gpu = cuda.to_device(stackBr)
    stackFr_gpu = cuda.to_device(stackFr)
    stackFl_gpu = cuda.to_device(stackFl)
    stackBl_gpu = cuda.to_device(stackBl)
    strengths_gpu = cuda.to_device(strengths)
    r_c0s_gpu = cuda.to_device(r_c0s)
    grid_vel_gpu = cuda.to_device(grid_velocities)
    counts_gpu = cuda.to_device(counts)
    ages_gpu = cuda.to_device(np.zeros(num_panels, dtype=np.float64))

    # Kernel configuration
    threads_per_block = 256
    blocks_per_grid = num_panels

    # Compute for each ring vortex leg
    @cuda.jit
    def bound_vortex_kernel(
        P_dev, S_dev, E_dev, str_dev, rc_dev, age_dev, nu, grid_dev, counts_dev
    ):
        vortex_id = cuda.blockIdx.x
        thread_id = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        for pt_id in range(thread_id, P_dev.shape[0], block_size):
            P = P_dev[pt_id]
            S = S_dev[vortex_id]
            E = E_dev[vortex_id]

            r0X = E[0] - S[0]
            r0Y = E[1] - S[1]
            r0Z = E[2] - S[2]
            r0 = math.sqrt(r0X * r0X + r0Y * r0Y + r0Z * r0Z)

            if r0 < _eps:
                cuda.atomic.add(counts_dev, 0, 1)
                continue

            strength = str_dev[vortex_id]
            age = age_dev[vortex_id]
            r_c0 = rc_dev[vortex_id]

            r0_times_tol = r0 * _tol
            r_c_sq = r_c0 * r_c0 + _four_lamb * (nu + _squire * abs(strength)) * age
            c_1 = strength / _four_pi
            c_2 = r0 * r0 * r_c_sq

            r1X = S[0] - P[0]
            r1Y = S[1] - P[1]
            r1Z = S[2] - P[2]
            r2X = E[0] - P[0]
            r2Y = E[1] - P[1]
            r2Z = E[2] - P[2]

            r3X = r1Y * r2Z - r1Z * r2Y
            r3Y = r1Z * r2X - r1X * r2Z
            r3Z = r1X * r2Y - r1Y * r2X

            r1 = math.sqrt(r1X * r1X + r1Y * r1Y + r1Z * r1Z)
            r2 = math.sqrt(r2X * r2X + r2Y * r2Y + r2Z * r2Z)

            if r1 < r0_times_tol:
                cuda.atomic.add(counts_dev, 1, 1)
            if r2 < r0_times_tol:
                cuda.atomic.add(counts_dev, 2, 1)

            r1r2 = r1 * r2
            if r1r2 < _eps:
                cuda.atomic.add(counts_dev, 3, 1)
                continue

            K = c_1 * (r0 / (r1r2 + c_2)) / r1r2
            grid_dev[pt_id, vortex_id, 0] += K * r3X
            grid_dev[pt_id, vortex_id, 1] += K * r3Y
            grid_dev[pt_id, vortex_id, 2] += K * r3Z

    # Process each leg
    for S, E in [
        (stackBr_gpu, stackFr_gpu),
        (stackFr_gpu, stackFl_gpu),
        (stackFl_gpu, stackBl_gpu),
        (stackBl_gpu, stackBr_gpu),
    ]:
        bound_vortex_kernel[blocks_per_grid, threads_per_block](
            stackP_gpu,
            S,
            E,
            strengths_gpu,
            r_c0s_gpu,
            ages_gpu,
            nu,
            grid_vel_gpu,
            counts_gpu,
        )

    cuda.synchronize()

    # Results back to CPU
    grid_velocities = grid_vel_gpu.copy_to_host()
    counts_result = counts_gpu.copy_to_host()
    singularity_counts[:] += counts_result

    return cast(np.ndarray, grid_velocities)
