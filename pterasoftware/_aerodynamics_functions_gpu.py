"""GPU-accelerated Biot-Savart kernels using Numba CUDA.

This module provides GPU implementations of the Biot-Savart kernels for computing
induced velocities from line vortices. These kernels are designed to be drop-in
replacements for the CPU parallel versions when CUDA is available.

Note: GPU kernels use simplified singularity handling (per-vortex validity flags)
instead of atomic event counting to improve GPU performance.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import cuda
    from numba.cuda import jit as cuda_jit

    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Constants (duplicated here to avoid circular imports)
_squire = 1.0e-4
_lamb = 1.25643
_eps = np.finfo(float).eps
_tol = 1.0e-10
_four_pi = 4.0 * math.pi
_four_lamb = 4.0 * _lamb


if CUDA_AVAILABLE:

    @cuda_jit
    def _collapsed_velocities_vortex_parallel_kernel(
        stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1,
        stackElvp_GP1_CgP1,
        strengths,
        r_c0s,
        stackVInd_GP1__E,
        vortex_valid,
        vortex_c1,
        vortex_c2,
        vortex_r0_times_tol,
    ):
        """CUDA kernel for collapsed Biot-Savart using VORTEX parallelization.

        Each thread processes one VORTEX across ALL evaluation points. Uses atomicAdd to
        accumulate contributions. This is more efficient when number of vortices is
        large (typical unsteady problem).

        :param stackP_GP1_CgP1: (N, 3) evaluation points on GPU
        :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices on GPU
        :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices on GPU
        :param strengths: (M,) vortex circulations on GPU
        :param r_c0s: (M,) initial core radii on GPU
        :param stackVInd_GP1__E: (N, 3) output velocities on GPU (initialized to 0)
        :param vortex_valid: (M,) per-vortex validity flags on GPU
        :param vortex_c1: (M,) pre-computed strength / (4*pi) on GPU
        :param vortex_c2: (M,) pre-computed r0^2 * r_c^2 on GPU
        :param vortex_r0_times_tol: (M,) pre-computed r0 * tolerance on GPU
        """
        vortex_id = cuda.grid(1)
        if vortex_id >= stackSlvp_GP1_CgP1.shape[0]:
            return

        if not vortex_valid[vortex_id]:
            return

        num_points = stackP_GP1_CgP1.shape[0]

        Slvp_x = stackSlvp_GP1_CgP1[vortex_id, 0]
        Slvp_y = stackSlvp_GP1_CgP1[vortex_id, 1]
        Slvp_z = stackSlvp_GP1_CgP1[vortex_id, 2]

        Elvp_x = stackElvp_GP1_CgP1[vortex_id, 0]
        Elvp_y = stackElvp_GP1_CgP1[vortex_id, 1]
        Elvp_z = stackElvp_GP1_CgP1[vortex_id, 2]

        r0_times_tol = vortex_r0_times_tol[vortex_id]
        c_1 = vortex_c1[vortex_id]
        c_2 = vortex_c2[vortex_id]

        # Loop over all evaluation points
        for point_id in range(num_points):
            P_x = stackP_GP1_CgP1[point_id, 0]
            P_y = stackP_GP1_CgP1[point_id, 1]
            P_z = stackP_GP1_CgP1[point_id, 2]

            # r1: from P to start vertex
            r1_x = Slvp_x - P_x
            r1_y = Slvp_y - P_y
            r1_z = Slvp_z - P_z

            # r2: from P to end vertex
            r2_x = Elvp_x - P_x
            r2_y = Elvp_y - P_y
            r2_z = Elvp_z - P_z

            # r3: cross product r1 x r2
            r3_x = r1_y * r2_z - r1_z * r2_y
            r3_y = r1_z * r2_x - r1_x * r2_z
            r3_z = r1_x * r2_y - r1_y * r2_x

            r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z)
            r2 = math.sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z)

            # Singularity checks
            if r1 < r0_times_tol or r2 < r0_times_tol:
                continue

            r3_sq = r3_x * r3_x + r3_y * r3_y + r3_z * r3_z
            r3 = math.sqrt(r3_sq)

            r1_times_r2 = r1 * r2
            c_3 = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z

            # Collinearity check
            if r3 < (_tol * r1_times_r2):
                continue

            # Biot-Savart calculation
            c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))

            # Accumulate contribution using atomicAdd
            cuda.atomic.add(stackVInd_GP1__E, (point_id, 0), c_4 * r3_x)
            cuda.atomic.add(stackVInd_GP1__E, (point_id, 1), c_4 * r3_y)
            cuda.atomic.add(stackVInd_GP1__E, (point_id, 2), c_4 * r3_z)

    @cuda_jit
    def _expanded_velocities_vortex_parallel_kernel(
        stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1,
        stackElvp_GP1_CgP1,
        strengths,
        r_c0s,
        gridVInd_GP1__E,
        vortex_valid,
        vortex_c1,
        vortex_c2,
        vortex_r0_times_tol,
    ):
        """CUDA kernel for expanded Biot-Savart using VORTEX parallelization.

        Each thread processes one VORTEX across ALL evaluation points. Writes directly
        to grid (no atomicAdd needed, different from collapsed).

        :param stackP_GP1_CgP1: (N, 3) evaluation points on GPU
        :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices on GPU
        :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices on GPU
        :param strengths: (M,) vortex circulations on GPU
        :param r_c0s: (M,) initial core radii on GPU
        :param gridVInd_GP1__E: (N, M, 3) output velocities on GPU
        :param vortex_valid: (M,) per-vortex validity flags on GPU
        :param vortex_c1: (M,) pre-computed strength / (4*pi) on GPU
        :param vortex_c2: (M,) pre-computed r0^2 * r_c^2 on GPU
        :param vortex_r0_times_tol: (M,) pre-computed r0 * tolerance on GPU
        """
        vortex_id = cuda.grid(1)
        if vortex_id >= stackSlvp_GP1_CgP1.shape[0]:
            return

        if not vortex_valid[vortex_id]:
            return

        num_points = stackP_GP1_CgP1.shape[0]

        Slvp_x = stackSlvp_GP1_CgP1[vortex_id, 0]
        Slvp_y = stackSlvp_GP1_CgP1[vortex_id, 1]
        Slvp_z = stackSlvp_GP1_CgP1[vortex_id, 2]

        Elvp_x = stackElvp_GP1_CgP1[vortex_id, 0]
        Elvp_y = stackElvp_GP1_CgP1[vortex_id, 1]
        Elvp_z = stackElvp_GP1_CgP1[vortex_id, 2]

        r0_times_tol = vortex_r0_times_tol[vortex_id]
        c_1 = vortex_c1[vortex_id]
        c_2 = vortex_c2[vortex_id]

        # Loop over all evaluation points
        for point_id in range(num_points):
            P_x = stackP_GP1_CgP1[point_id, 0]
            P_y = stackP_GP1_CgP1[point_id, 1]
            P_z = stackP_GP1_CgP1[point_id, 2]

            # r1: from P to start vertex
            r1_x = Slvp_x - P_x
            r1_y = Slvp_y - P_y
            r1_z = Slvp_z - P_z

            # r2: from P to end vertex
            r2_x = Elvp_x - P_x
            r2_y = Elvp_y - P_y
            r2_z = Elvp_z - P_z

            # r3: cross product r1 x r2
            r3_x = r1_y * r2_z - r1_z * r2_y
            r3_y = r1_z * r2_x - r1_x * r2_z
            r3_z = r1_x * r2_y - r1_y * r2_x

            r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z)
            r2 = math.sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z)

            # Singularity checks
            if r1 < r0_times_tol or r2 < r0_times_tol:
                gridVInd_GP1__E[point_id, vortex_id, 0] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 1] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 2] = 0.0
                continue

            r3_sq = r3_x * r3_x + r3_y * r3_y + r3_z * r3_z
            r3 = math.sqrt(r3_sq)

            r1_times_r2 = r1 * r2
            c_3 = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z

            # Collinearity check
            if r3 < (_tol * r1_times_r2):
                gridVInd_GP1__E[point_id, vortex_id, 0] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 1] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 2] = 0.0
                continue

            # Biot-Savart calculation
            c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))

            gridVInd_GP1__E[point_id, vortex_id, 0] = c_4 * r3_x
            gridVInd_GP1__E[point_id, vortex_id, 1] = c_4 * r3_y
            gridVInd_GP1__E[point_id, vortex_id, 2] = c_4 * r3_z

    @cuda_jit
    def _collapsed_velocities_from_line_vortices_kernel(
        stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1,
        stackElvp_GP1_CgP1,
        strengths,
        r_c0s,
        stackVInd_GP1__E,
        vortex_valid,
        vortex_c1,
        vortex_c2,
        vortex_r0_times_tol,
    ):
        """CUDA kernel for collapsed Biot-Savart calculations.

        Each thread computes the induced velocity at one evaluation point from all
        vortices.

        :param stackP_GP1_CgP1: (N, 3) evaluation points
        :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
        :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
        :param strengths: (M,) vortex circulations
        :param r_c0s: (M,) initial core radii
        :param stackVInd_GP1__E: (N, 3) output velocities (pre-allocated)
        :param vortex_valid: (M,) per-vortex validity flags
        :param vortex_c1: (M,) pre-computed strength / (4*pi)
        :param vortex_c2: (M,) pre-computed r0^2 * r_c^2
        :param vortex_r0_times_tol: (M,) pre-computed r0 * tolerance
        """
        point_id = cuda.grid(1)
        if point_id >= stackP_GP1_CgP1.shape[0]:
            return

        num_vortices = stackSlvp_GP1_CgP1.shape[0]

        P_x = stackP_GP1_CgP1[point_id, 0]
        P_y = stackP_GP1_CgP1[point_id, 1]
        P_z = stackP_GP1_CgP1[point_id, 2]

        vel_x = 0.0
        vel_y = 0.0
        vel_z = 0.0

        for vortex_id in range(num_vortices):
            if not vortex_valid[vortex_id]:
                continue

            Slvp_x = stackSlvp_GP1_CgP1[vortex_id, 0]
            Slvp_y = stackSlvp_GP1_CgP1[vortex_id, 1]
            Slvp_z = stackSlvp_GP1_CgP1[vortex_id, 2]

            Elvp_x = stackElvp_GP1_CgP1[vortex_id, 0]
            Elvp_y = stackElvp_GP1_CgP1[vortex_id, 1]
            Elvp_z = stackElvp_GP1_CgP1[vortex_id, 2]

            # r1: from P to start vertex
            r1_x = Slvp_x - P_x
            r1_y = Slvp_y - P_y
            r1_z = Slvp_z - P_z

            # r2: from P to end vertex
            r2_x = Elvp_x - P_x
            r2_y = Elvp_y - P_y
            r2_z = Elvp_z - P_z

            # r3: cross product r1 x r2
            r3_x = r1_y * r2_z - r1_z * r2_y
            r3_y = r1_z * r2_x - r1_x * r2_z
            r3_z = r1_x * r2_y - r1_y * r2_x

            r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z)
            r2 = math.sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z)

            # Singularity checks
            r0_times_tol = vortex_r0_times_tol[vortex_id]
            if r1 < r0_times_tol or r2 < r0_times_tol:
                continue

            r3_sq = r3_x * r3_x + r3_y * r3_y + r3_z * r3_z
            r3 = math.sqrt(r3_sq)

            r1_times_r2 = r1 * r2
            c_3 = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z

            # Collinearity check
            if r3 < (_tol * r1_times_r2):
                continue

            # Biot-Savart calculation
            c_4 = (
                vortex_c1[vortex_id]
                * (r1 + r2)
                * (r1_times_r2 - c_3)
                / (r1_times_r2 * (r3_sq + vortex_c2[vortex_id]))
            )

            vel_x += c_4 * r3_x
            vel_y += c_4 * r3_y
            vel_z += c_4 * r3_z

        stackVInd_GP1__E[point_id, 0] = vel_x
        stackVInd_GP1__E[point_id, 1] = vel_y
        stackVInd_GP1__E[point_id, 2] = vel_z

    @cuda_jit
    def _expanded_velocities_from_line_vortices_kernel(
        stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1,
        stackElvp_GP1_CgP1,
        strengths,
        r_c0s,
        gridVInd_GP1__E,
        vortex_valid,
        vortex_c1,
        vortex_c2,
        vortex_r0_times_tol,
    ):
        """CUDA kernel for expanded Biot-Savart calculations.

        Each thread computes the induced velocity at one point due to one vortex.

        :param stackP_GP1_CgP1: (N, 3) evaluation points
        :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
        :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
        :param strengths: (M,) vortex circulations
        :param r_c0s: (M,) initial core radii
        :param gridVInd_GP1__E: (N, M, 3) output velocities (pre-allocated)
        :param vortex_valid: (M,) per-vortex validity flags
        :param vortex_c1: (M,) pre-computed strength / (4*pi)
        :param vortex_c2: (M,) pre-computed r0^2 * r_c^2
        :param vortex_r0_times_tol: (M,) pre-computed r0 * tolerance
        """
        # Grid structure: use 2D grid (point_id, vortex_id)
        point_id = cuda.grid(1)

        if point_id >= stackP_GP1_CgP1.shape[0]:
            return

        num_vortices = stackSlvp_GP1_CgP1.shape[0]

        P_x = stackP_GP1_CgP1[point_id, 0]
        P_y = stackP_GP1_CgP1[point_id, 1]
        P_z = stackP_GP1_CgP1[point_id, 2]

        for vortex_id in range(num_vortices):
            if not vortex_valid[vortex_id]:
                gridVInd_GP1__E[point_id, vortex_id, 0] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 1] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 2] = 0.0
                continue

            Slvp_x = stackSlvp_GP1_CgP1[vortex_id, 0]
            Slvp_y = stackSlvp_GP1_CgP1[vortex_id, 1]
            Slvp_z = stackSlvp_GP1_CgP1[vortex_id, 2]

            Elvp_x = stackElvp_GP1_CgP1[vortex_id, 0]
            Elvp_y = stackElvp_GP1_CgP1[vortex_id, 1]
            Elvp_z = stackElvp_GP1_CgP1[vortex_id, 2]

            # r1: from P to start vertex
            r1_x = Slvp_x - P_x
            r1_y = Slvp_y - P_y
            r1_z = Slvp_z - P_z

            # r2: from P to end vertex
            r2_x = Elvp_x - P_x
            r2_y = Elvp_y - P_y
            r2_z = Elvp_z - P_z

            # r3: cross product r1 x r2
            r3_x = r1_y * r2_z - r1_z * r2_y
            r3_y = r1_z * r2_x - r1_x * r2_z
            r3_z = r1_x * r2_y - r1_y * r2_x

            r1 = math.sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z)
            r2 = math.sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z)

            # Singularity checks
            r0_times_tol = vortex_r0_times_tol[vortex_id]
            if r1 < r0_times_tol or r2 < r0_times_tol:
                gridVInd_GP1__E[point_id, vortex_id, 0] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 1] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 2] = 0.0
                continue

            r3_sq = r3_x * r3_x + r3_y * r3_y + r3_z * r3_z
            r3 = math.sqrt(r3_sq)

            r1_times_r2 = r1 * r2
            c_3 = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z

            # Collinearity check
            if r3 < (_tol * r1_times_r2):
                gridVInd_GP1__E[point_id, vortex_id, 0] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 1] = 0.0
                gridVInd_GP1__E[point_id, vortex_id, 2] = 0.0
                continue

            # Biot-Savart calculation
            c_4 = (
                vortex_c1[vortex_id]
                * (r1 + r2)
                * (r1_times_r2 - c_3)
                / (r1_times_r2 * (r3_sq + vortex_c2[vortex_id]))
            )

            gridVInd_GP1__E[point_id, vortex_id, 0] = c_4 * r3_x
            gridVInd_GP1__E[point_id, vortex_id, 1] = c_4 * r3_y
            gridVInd_GP1__E[point_id, vortex_id, 2] = c_4 * r3_z


def _collapsed_velocities_from_line_vortices_cuda(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated collapsed Biot-Savart kernel using Numba CUDA.

    Computes induced velocities using CUDA if available, otherwise falls back to
    returning None to indicate failure.

    Note: This version uses per-vortex validity flags instead of atomic counting for
    singularities to improve GPU performance.

    :param stackP_GP1_CgP1: (N, 3) evaluation points
    :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
    :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
    :param strengths: (M,) vortex circulations
    :param r_c0s: (M,) initial core radii
    :param singularity_counts: (4,) singularity event counter (not updated in GPU
        version)
    :param ages: (M,) vortex ages (currently ignored in GPU version)
    :param nu: kinematic viscosity (currently ignored in GPU version)
    :return: (N, 3) induced velocities, or None if CUDA unavailable
    """
    if not CUDA_AVAILABLE:
        return None

    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Pre-compute per-vortex quantities on CPU, then transfer to GPU
    vortex_valid = np.empty(num_vortices, dtype=np.bool_)
    vortex_c1 = np.empty(num_vortices, dtype=np.float64)
    vortex_c2 = np.empty(num_vortices, dtype=np.float64)
    vortex_r0_times_tol = np.empty(num_vortices, dtype=np.float64)

    degenerate_count = 0
    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        if r0 < _eps:
            vortex_valid[vortex_id] = False
            degenerate_count += 1
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Simplified core radius (ignoring age and viscous growth for GPU version)
        r_c_sq = r_c0**2.0

        vortex_r0_times_tol[vortex_id] = r0 * _tol
        vortex_c1[vortex_id] = strength / _four_pi
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Update singularity counter for degenerate filaments
    singularity_counts[0] += degenerate_count

    # Transfer data to GPU
    stackP_d = cuda.to_device(stackP_GP1_CgP1)
    stackSlvp_d = cuda.to_device(stackSlvp_GP1_CgP1)
    stackElvp_d = cuda.to_device(stackElvp_GP1_CgP1)
    strengths_d = cuda.to_device(strengths)
    r_c0s_d = cuda.to_device(r_c0s)

    vortex_valid_d = cuda.to_device(vortex_valid)
    vortex_c1_d = cuda.to_device(vortex_c1)
    vortex_c2_d = cuda.to_device(vortex_c2)
    vortex_r0_times_tol_d = cuda.to_device(vortex_r0_times_tol)

    # Allocate output on GPU
    stackVInd_GP1__E_d = cuda.device_array((num_points, 3), dtype=np.float64)

    # Determine grid/block dimensions
    # Using 256 threads per block (efficient for most GPUs)
    threads_per_block = 256
    num_blocks = (num_points + threads_per_block - 1) // threads_per_block

    # Launch kernel
    _collapsed_velocities_from_line_vortices_kernel[num_blocks, threads_per_block](
        stackP_d,
        stackSlvp_d,
        stackElvp_d,
        strengths_d,
        r_c0s_d,
        stackVInd_GP1__E_d,
        vortex_valid_d,
        vortex_c1_d,
        vortex_c2_d,
        vortex_r0_times_tol_d,
    )

    # Transfer result back to CPU
    stackVInd_GP1__E = stackVInd_GP1__E_d.copy_to_host()

    return stackVInd_GP1__E


def _expanded_velocities_from_line_vortices_cuda(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated expanded Biot-Savart kernel using Numba CUDA.

    Computes induced velocities using CUDA if available, otherwise falls back to
    returning None to indicate failure.

    Note: This version uses per-vortex validity flags instead of atomic counting for
    singularities to improve GPU performance.

    :param stackP_GP1_CgP1: (N, 3) evaluation points
    :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
    :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
    :param strengths: (M,) vortex circulations
    :param r_c0s: (M,) initial core radii
    :param singularity_counts: (4,) singularity event counter (not updated in GPU
        version)
    :param ages: (M,) vortex ages (currently ignored in GPU version)
    :param nu: kinematic viscosity (currently ignored in GPU version)
    :return: (N, M, 3) induced velocities, or None if CUDA unavailable
    """
    if not CUDA_AVAILABLE:
        return None

    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Pre-compute per-vortex quantities on CPU, then transfer to GPU
    vortex_valid = np.empty(num_vortices, dtype=np.bool_)
    vortex_c1 = np.empty(num_vortices, dtype=np.float64)
    vortex_c2 = np.empty(num_vortices, dtype=np.float64)
    vortex_r0_times_tol = np.empty(num_vortices, dtype=np.float64)

    degenerate_count = 0
    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        if r0 < _eps:
            vortex_valid[vortex_id] = False
            degenerate_count += 1
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Simplified core radius (ignoring age and viscous growth for GPU version)
        r_c_sq = r_c0**2.0

        vortex_r0_times_tol[vortex_id] = r0 * _tol
        vortex_c1[vortex_id] = strength / _four_pi
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Update singularity counter for degenerate filaments
    singularity_counts[0] += degenerate_count

    # Transfer data to GPU
    stackP_d = cuda.to_device(stackP_GP1_CgP1)
    stackSlvp_d = cuda.to_device(stackSlvp_GP1_CgP1)
    stackElvp_d = cuda.to_device(stackElvp_GP1_CgP1)
    strengths_d = cuda.to_device(strengths)
    r_c0s_d = cuda.to_device(r_c0s)

    vortex_valid_d = cuda.to_device(vortex_valid)
    vortex_c1_d = cuda.to_device(vortex_c1)
    vortex_c2_d = cuda.to_device(vortex_c2)
    vortex_r0_times_tol_d = cuda.to_device(vortex_r0_times_tol)

    # Allocate output on GPU
    gridVInd_GP1__E_d = cuda.device_array(
        (num_points, num_vortices, 3), dtype=np.float64
    )

    # Determine grid/block dimensions
    threads_per_block = 256
    num_blocks = (num_points + threads_per_block - 1) // threads_per_block

    # Launch kernel
    _expanded_velocities_from_line_vortices_kernel[num_blocks, threads_per_block](
        stackP_d,
        stackSlvp_d,
        stackElvp_d,
        strengths_d,
        r_c0s_d,
        gridVInd_GP1__E_d,
        vortex_valid_d,
        vortex_c1_d,
        vortex_c2_d,
        vortex_r0_times_tol_d,
    )

    # Transfer result back to CPU
    gridVInd_GP1__E = gridVInd_GP1__E_d.copy_to_host()

    return gridVInd_GP1__E


def _collapsed_velocities_from_line_vortices_cuda_vortex_parallel(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated collapsed Biot-Savart kernel using vortex-parallel kernels.

    STRATEGY 2: Parallelizes over VORTICES (more efficient for unsteady problems). Each
    thread processes one vortex across ALL evaluation points using atomicAdd.

    Uses tuples for atomicAdd indexing: cuda.atomic.add(array, (idx, jdx), value)

    :param stackP_GP1_CgP1: (N, 3) evaluation points
    :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
    :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
    :param strengths: (M,) vortex circulations
    :param r_c0s: (M,) initial core radii
    :param singularity_counts: (4,) singularity event counter (not updated in GPU
        version)
    :param ages: (M,) vortex ages (currently ignored in GPU version)
    :param nu: kinematic viscosity (currently ignored in GPU version)
    :return: (N, 3) induced velocities, or None if CUDA unavailable
    """
    if not CUDA_AVAILABLE:
        return None

    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Pre-compute per-vortex quantities on CPU, then transfer to GPU
    vortex_valid = np.empty(num_vortices, dtype=np.bool_)
    vortex_c1 = np.empty(num_vortices, dtype=np.float64)
    vortex_c2 = np.empty(num_vortices, dtype=np.float64)
    vortex_r0_times_tol = np.empty(num_vortices, dtype=np.float64)

    degenerate_count = 0
    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        if r0 < _eps:
            vortex_valid[vortex_id] = False
            degenerate_count += 1
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Simplified core radius (ignoring age and viscous growth for GPU version)
        r_c_sq = r_c0**2.0

        vortex_r0_times_tol[vortex_id] = r0 * _tol
        vortex_c1[vortex_id] = strength / _four_pi
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Update singularity counter for degenerate filaments
    singularity_counts[0] += degenerate_count

    # Transfer data to GPU
    stackP_d = cuda.to_device(stackP_GP1_CgP1)
    stackSlvp_d = cuda.to_device(stackSlvp_GP1_CgP1)
    stackElvp_d = cuda.to_device(stackElvp_GP1_CgP1)
    strengths_d = cuda.to_device(strengths)
    r_c0s_d = cuda.to_device(r_c0s)

    vortex_valid_d = cuda.to_device(vortex_valid)
    vortex_c1_d = cuda.to_device(vortex_c1)
    vortex_c2_d = cuda.to_device(vortex_c2)
    vortex_r0_times_tol_d = cuda.to_device(vortex_r0_times_tol)

    # Allocate output on GPU, initialized to zero for atomicAdd
    stackVInd_GP1__E_d = cuda.to_device(np.zeros((num_points, 3), dtype=np.float64))

    # Determine grid/block dimensions - VORTEX parallelization
    threads_per_block = 256
    num_blocks = (num_vortices + threads_per_block - 1) // threads_per_block

    # Launch kernel
    _collapsed_velocities_vortex_parallel_kernel[num_blocks, threads_per_block](
        stackP_d,
        stackSlvp_d,
        stackElvp_d,
        strengths_d,
        r_c0s_d,
        stackVInd_GP1__E_d,
        vortex_valid_d,
        vortex_c1_d,
        vortex_c2_d,
        vortex_r0_times_tol_d,
    )

    # Transfer result back to CPU
    stackVInd_GP1__E = stackVInd_GP1__E_d.copy_to_host()

    return stackVInd_GP1__E


def _expanded_velocities_from_line_vortices_cuda_vortex_parallel(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated expanded Biot-Savart kernel using vortex-parallel kernels.

    STRATEGY 2: Parallelizes over VORTICES (more efficient for unsteady problems). Each
    thread processes one vortex across ALL evaluation points.

    :param stackP_GP1_CgP1: (N, 3) evaluation points
    :param stackSlvp_GP1_CgP1: (M, 3) vortex start vertices
    :param stackElvp_GP1_CgP1: (M, 3) vortex end vertices
    :param strengths: (M,) vortex circulations
    :param r_c0s: (M,) initial core radii
    :param singularity_counts: (4,) singularity event counter (not updated in GPU
        version)
    :param ages: (M,) vortex ages (currently ignored in GPU version)
    :param nu: kinematic viscosity (currently ignored in GPU version)
    :return: (N, M, 3) induced velocities, or None if CUDA unavailable
    """
    if not CUDA_AVAILABLE:
        return None

    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Pre-compute per-vortex quantities on CPU, then transfer to GPU
    vortex_valid = np.empty(num_vortices, dtype=np.bool_)
    vortex_c1 = np.empty(num_vortices, dtype=np.float64)
    vortex_c2 = np.empty(num_vortices, dtype=np.float64)
    vortex_r0_times_tol = np.empty(num_vortices, dtype=np.float64)

    degenerate_count = 0
    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        if r0 < _eps:
            vortex_valid[vortex_id] = False
            degenerate_count += 1
            continue

        vortex_valid[vortex_id] = True

        strength = strengths[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Simplified core radius (ignoring age and viscous growth for GPU version)
        r_c_sq = r_c0**2.0

        vortex_r0_times_tol[vortex_id] = r0 * _tol
        vortex_c1[vortex_id] = strength / _four_pi
        vortex_c2[vortex_id] = r0**2.0 * r_c_sq

    # Update singularity counter for degenerate filaments
    singularity_counts[0] += degenerate_count

    # Transfer data to GPU
    stackP_d = cuda.to_device(stackP_GP1_CgP1)
    stackSlvp_d = cuda.to_device(stackSlvp_GP1_CgP1)
    stackElvp_d = cuda.to_device(stackElvp_GP1_CgP1)
    strengths_d = cuda.to_device(strengths)
    r_c0s_d = cuda.to_device(r_c0s)

    vortex_valid_d = cuda.to_device(vortex_valid)
    vortex_c1_d = cuda.to_device(vortex_c1)
    vortex_c2_d = cuda.to_device(vortex_c2)
    vortex_r0_times_tol_d = cuda.to_device(vortex_r0_times_tol)

    # Allocate output on GPU
    gridVInd_GP1__E_d = cuda.device_array(
        (num_points, num_vortices, 3), dtype=np.float64
    )

    # Determine grid/block dimensions - VORTEX parallelization
    threads_per_block = 256
    num_blocks = (num_vortices + threads_per_block - 1) // threads_per_block

    # Launch kernel
    _expanded_velocities_vortex_parallel_kernel[num_blocks, threads_per_block](
        stackP_d,
        stackSlvp_d,
        stackElvp_d,
        strengths_d,
        r_c0s_d,
        gridVInd_GP1__E_d,
        vortex_valid_d,
        vortex_c1_d,
        vortex_c2_d,
        vortex_r0_times_tol_d,
    )

    # Transfer result back to CPU
    gridVInd_GP1__E = gridVInd_GP1__E_d.copy_to_host()

    return gridVInd_GP1__E
