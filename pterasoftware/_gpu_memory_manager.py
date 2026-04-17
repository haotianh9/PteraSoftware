"""GPU Memory Manager for persistent data pooling across timesteps (Phase 2).

Manages persistent GPU arrays to minimize PCIe transfers: - Static data (panel geometry,
bound vortex vertices) transferred once - Wake vortices use append-only structure
(transfer only new data) - Age updates performed on GPU (no CPU transfer needed)

Transfer optimization: - Initialization: ~20 KB once - Per-timestep: ~1.7 KB (bound
strengths + new wake vortices) - Compared to naive approach: 43 KB → 1.7 KB per step
"""

import math

import numpy as np
from numba import cuda, float64, int64

# Import constants from CPU version
_eps = 1e-14
_tol = 1e-4
_four_pi = 4.0 * math.pi
_four_lamb = 4.0 * math.log(2.0)
_squire = 0.25


class GPUMemoryPool:
    """Persistent GPU memory manager for aerodynamic simulations.

    Keeps static data (panel geometry, bound vortices) on GPU across all timesteps. Wake
    vortices use append-only structure (new data appended each timestep).

    Transfer strategy: - Initialization: Transfer all static data once (panel geometry,
    bound vortices) - Per-timestep: Only transfer new wake vortices and bound strengths
    - Age updates: Performed on GPU (cheap)
    """

    def __init__(
        self,
        num_panels: int,
        num_bound_vortices: int,
        max_num_steps: int,
        num_spanwise_panels: int,
        viscosity: float = 1e-5,
    ):
        """Initialize GPU memory pool with pre-allocated arrays.

        Parameters: ----------- num_panels : int     Number of evaluation points (panel
        collocation points) num_bound_vortices : int     Number of bound vortices
        (typically 4*num_panels or num_panels) max_num_steps : int     Maximum number of
        timesteps in simulation num_spanwise_panels : int     Spanwise panels (new wake
        vortices per timestep) viscosity : float     Physical viscosity parameter (nu)
        """
        if not cuda.is_available():
            raise RuntimeError("CUDA not available. GPU memory manager requires GPU.")

        self.num_panels = num_panels
        self.num_bound_vortices = num_bound_vortices
        self.max_num_steps = max_num_steps
        self.num_spanwise_panels = num_spanwise_panels
        self.viscosity = viscosity

        # Estimate max wake vortices
        max_wake_vortices = max_num_steps * num_spanwise_panels

        # Static data on GPU (transferred once at initialization)
        self.panel_cpp_gpu = cuda.device_array((num_panels, 3), dtype=np.float64)
        self.panel_normals_gpu = cuda.device_array((num_panels, 3), dtype=np.float64)

        # Bound vortex geometry (4 corners per panel)
        self.bound_start_vortices_gpu = cuda.device_array(
            (num_bound_vortices, 3), dtype=np.float64
        )
        self.bound_end_vortices_gpu = cuda.device_array(
            (num_bound_vortices, 3), dtype=np.float64
        )

        # Dynamic data on GPU (append-only structure for wake)
        self.wake_start_vortices_gpu = cuda.device_array(
            (max_wake_vortices, 3), dtype=np.float64
        )
        self.wake_end_vortices_gpu = cuda.device_array(
            (max_wake_vortices, 3), dtype=np.float64
        )
        self.wake_strengths_gpu = cuda.device_array(max_wake_vortices, dtype=np.float64)
        self.wake_ages_gpu = cuda.device_array(max_wake_vortices, dtype=np.float64)
        self.wake_rc0s_gpu = cuda.device_array(max_wake_vortices, dtype=np.float64)

        # Bound strengths (changes each timestep)
        self.bound_strengths_gpu = cuda.device_array(
            num_bound_vortices, dtype=np.float64
        )

        # Computation results buffers
        self.velocities_gpu = cuda.device_array((num_panels, 3), dtype=np.float64)
        self.singularity_counts_gpu = cuda.device_array(4, dtype=np.int64)

        # Tracking
        self.current_num_wake_vortices = 0
        self.is_initialized = False

    def initialize_static_data(
        self,
        panel_cpp: np.ndarray,
        panel_normals: np.ndarray,
        bound_start: np.ndarray,
        bound_end: np.ndarray,
    ) -> None:
        """Transfer static panel and bound vortex geometry to GPU (one-time operation).

        Parameters: ----------- panel_cpp : np.ndarray     Panel collocation points
        (num_panels, 3) panel_normals : np.ndarray     Panel normal vectors (num_panels,
        3) bound_start : np.ndarray     Start vertices of bound vortices
        (num_bound_vortices, 3) bound_end : np.ndarray     End vertices of bound
        vortices (num_bound_vortices, 3)
        """
        if self.is_initialized:
            raise RuntimeError("Static data already initialized")

        # Transfer panel geometry (one-time)
        cuda.to_device(panel_cpp, to=self.panel_cpp_gpu)
        cuda.to_device(panel_normals, to=self.panel_normals_gpu)

        # Transfer bound vortex geometry (one-time, static)
        cuda.to_device(bound_start, to=self.bound_start_vortices_gpu)
        cuda.to_device(bound_end, to=self.bound_end_vortices_gpu)

        self.is_initialized = True

    def update_bound_strengths(self, bound_strengths: np.ndarray) -> None:
        """Transfer bound vortex strengths to GPU (per-timestep operation).

        Only ~1.6 KB transfer for typical cases.

        Parameters: ----------- bound_strengths : np.ndarray     Bound vortex strengths
        (num_bound_vortices,)
        """
        if bound_strengths.shape[0] != self.num_bound_vortices:
            raise ValueError(
                f"Expected {self.num_bound_vortices} strengths, "
                f"got {bound_strengths.shape[0]}"
            )
        cuda.to_device(bound_strengths, to=self.bound_strengths_gpu)

    def append_wake_vortices(
        self,
        wake_start: np.ndarray,
        wake_end: np.ndarray,
        strengths: np.ndarray,
        ages: np.ndarray,
        rc0s: np.ndarray,
    ) -> None:
        """Append new wake vortices to GPU (append-only structure).

        Only transfers new vortices added in this timestep (~48 bytes per spanwise).

        Parameters: ----------- wake_start : np.ndarray     Start vertices of NEW wake
        vortices (num_new, 3) wake_end : np.ndarray     End vertices of NEW wake
        vortices (num_new, 3) strengths : np.ndarray     Strengths of NEW wake vortices
        (num_new,) ages : np.ndarray     Ages of NEW wake vortices (num_new,) rc0s :
        np.ndarray     Core radius parameters of NEW wake vortices (num_new,)
        """
        num_new = wake_start.shape[0]

        if (
            self.current_num_wake_vortices + num_new
            > self.wake_start_vortices_gpu.shape[0]
        ):
            raise RuntimeError(
                f"Wake vortex array overflow: "
                f"current={self.current_num_wake_vortices}, "
                f"new={num_new}, "
                f"max={self.wake_start_vortices_gpu.shape[0]}"
            )

        # Append to GPU arrays (only new data transferred)
        start_idx = self.current_num_wake_vortices
        end_idx = start_idx + num_new

        cuda.to_device(wake_start, to=self.wake_start_vortices_gpu[start_idx:end_idx])
        cuda.to_device(wake_end, to=self.wake_end_vortices_gpu[start_idx:end_idx])
        cuda.to_device(strengths, to=self.wake_strengths_gpu[start_idx:end_idx])
        cuda.to_device(ages, to=self.wake_ages_gpu[start_idx:end_idx])
        cuda.to_device(rc0s, to=self.wake_rc0s_gpu[start_idx:end_idx])

        self.current_num_wake_vortices = end_idx

    def update_wake_ages(self, delta_time: float) -> None:
        """Update wake vortex ages on GPU (GPU-side computation, no transfer).

        This is a cheap operation performed entirely on GPU.

        Parameters: ----------- delta_time : float     Time step increment
        """
        if self.current_num_wake_vortices == 0:
            return

        _update_ages_kernel[(self.current_num_wake_vortices + 255) // 256, 256](
            self.wake_ages_gpu, delta_time, self.current_num_wake_vortices
        )

    def get_velocities_gpu(self) -> np.ndarray:
        """Get GPU array pointer to velocities (for kernel output)."""
        return self.velocities_gpu

    def get_singularity_counts_gpu(self) -> np.ndarray:
        """Get GPU array pointer to singularity counts (for kernel output)."""
        return self.singularity_counts_gpu

    def fetch_velocities_to_cpu(self) -> np.ndarray:
        """Transfer velocity results from GPU to CPU.

        Returns: -------- np.ndarray     Velocities on CPU (num_panels, 3)
        """
        return self.velocities_gpu.copy_to_host()

    def fetch_singularity_counts_to_cpu(self) -> np.ndarray:
        """Transfer singularity counts from GPU to CPU.

        Returns: -------- np.ndarray     Singularity counts on CPU (4,)
        """
        return self.singularity_counts_gpu.copy_to_host()

    def reset_velocities(self) -> None:
        """Reset velocity buffer for next computation."""
        self.velocities_gpu.fill(0.0)

    def reset_singularity_counts(self) -> None:
        """Reset singularity count buffer for next computation."""
        self.singularity_counts_gpu.fill(0)

    def clear_wake_data(self) -> None:
        """Clear accumulated wake data (e.g., for new simulation)."""
        self.current_num_wake_vortices = 0


@cuda.jit
def _update_ages_kernel(ages_gpu, delta_time, num_wake_vortices):
    """GPU kernel to update wake vortex ages in-place.

    This runs on GPU, no CPU transfer needed for age updates.

    Parameters: ----------- ages_gpu : np.ndarray     Wake vortex ages on GPU (updated
    in-place) delta_time : float     Time step increment num_wake_vortices : int Number
    of wake vortices to update
    """
    idx = cuda.grid(1)
    if idx < num_wake_vortices:
        ages_gpu[idx] += delta_time
