"""GPU/CPU Backend Configuration with Compile-Time Flags (Issue #140).

This module provides runtime-configurable backend selection for aerodynamics
computations:

- PTERASOFTWARE_BACKEND: Environment variable to select serial CPU, parallel CPU,   or
GPU backend - Graceful fallback when GPU unavailable - Configuration persists for entire
Python session - Runtime API to switch backends (for advanced usage)

Quick Start:

# Use GPU if available (set before import) import os os.environ["PTERASOFTWARE_BACKEND"]
= "gpu_cuda"

# Or use serial CPU (default - no action needed)

# Check configuration from pterasoftware import get_backend_info
print(get_backend_info())

Environment Variables:

PTERASOFTWARE_BACKEND : {'serial_cpu', 'parallel_cpu', 'gpu_cuda'}     Select
computation backend (default: 'serial_cpu')     - serial_cpu: Single-threaded baseline
with fastmath     - parallel_cpu: Multi-core with Numba prange (usually slower)     -
gpu_cuda: GPU acceleration (requires NVIDIA GPU + CUDA)

PTERASOFTWARE_FORCE_GPU : {'0', '1', 'false', 'true'}     Force GPU even if unavailable
(default: '0' - fallback to CPU)     Set to '1' to error if GPU backbone selected but
unavailable
"""

import os
import warnings
from enum import Enum

from numba import cuda


class Backend(Enum):
    """Computation backend selector."""

    SERIAL_CPU = "serial_cpu"
    PARALLEL_CPU = "parallel_cpu"
    GPU_CUDA = "gpu_cuda"


class AerodynamicsConfig:
    """Compile-time configuration for aerodynamics computations.

    Configuration is read from environment variables at module load time. Changes after
    module import require runtime override via set_backend().

    Environment Variables: ---------------------- PTERASOFTWARE_BACKEND : {'serial_cpu',
    'parallel_cpu', 'gpu_cuda'}     Select computation backend (default: 'serial_cpu')

    PTERASOFTWARE_CPU_PARALLEL : {'0', '1', 'false', 'true'}     Enable CPU
    parallelization (default: '0' - disabled)     Only used if PTERASOFTWARE_BACKEND is
    'parallel_cpu'

    PTERASOFTWARE_FORCE_GPU : {'0', '1'}     Force GPU even if unavailable (default: '0'
    - fallback to CPU)

    Examples: --------- # Use GPU if available, fall back to serial CPU export
    PTERASOFTWARE_BACKEND=gpu_cuda

    # Force CPU parallelization (not recommended unless proven beneficial) export
    PTERASOFTWARE_BACKEND=parallel_cpu

    # Keep default (serial CPU - fastest for most cases) export
    PTERASOFTWARE_BACKEND=serial_cpu
    """

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Read backend from environment
        backend_str = os.environ.get("PTERASOFTWARE_BACKEND", "serial_cpu").lower()

        # Validate backend choice
        valid_backends = {b.value for b in Backend}
        if backend_str not in valid_backends:
            warnings.warn(
                f"Invalid PTERASOFTWARE_BACKEND='{backend_str}'. "
                f"Valid options: {valid_backends}. Using 'serial_cpu'.",
                RuntimeWarning,
            )
            backend_str = "serial_cpu"

        self._backend_choice = Backend(backend_str)

        # Check GPU availability
        self._gpu_available = cuda.is_available()

        # Handle GPU selection
        if self._backend_choice == Backend.GPU_CUDA:
            if not self._gpu_available:
                force_gpu = os.environ.get("PTERASOFTWARE_FORCE_GPU", "0")
                if force_gpu in ("0", "false"):
                    warnings.warn(
                        "PTERASOFTWARE_BACKEND=gpu_cuda selected but GPU not available. "
                        "Falling back to serial CPU. "
                        "To force GPU (and error if unavailable), set "
                        "PTERASOFTWARE_FORCE_GPU=1",
                        RuntimeWarning,
                    )
                    self._backend_choice = Backend.SERIAL_CPU
                else:
                    raise RuntimeError(
                        "PTERASOFTWARE_BACKEND=gpu_cuda and PTERASOFTWARE_FORCE_GPU=1 "
                        "selected, but GPU not available."
                    )

        # Log configuration on first import
        self._log_configuration()

    def _log_configuration(self) -> None:
        """Log current configuration to stderr (non-blocking)."""
        import sys

        backend_name = self._backend_choice.value
        status = "✓" if self.get_backend() != Backend.SERIAL_CPU else " "

        if self._gpu_available:
            gpu_status = "✓ Available"
        else:
            gpu_status = "✗ Not available"

        # Only log if not in non-interactive mode or if backend is non-default
        if self._backend_choice != Backend.SERIAL_CPU or self._gpu_available:
            print(
                f"[PteraSoftware] Aerodynamics backend: {backend_name} "
                f"| GPU: {gpu_status}",
                file=sys.stderr,
            )

    def get_backend(self) -> Backend:
        """Get configured computation backend.

        Returns: -------- Backend     Currently configured backend

        Raises: ------- RuntimeError     If GPU configured but not available and
        PTERASOFTWARE_FORCE_GPU=1
        """
        return self._backend_choice

    def get_use_gpu(self) -> bool:
        """Check if GPU backend is configured and available.

        Returns: -------- bool     True if GPU should be used, False otherwise
        """
        return self._backend_choice == Backend.GPU_CUDA and self._gpu_available

    def get_use_parallel_cpu(self) -> bool:
        """Check if parallel CPU backend is configured.

        Returns: -------- bool     True if CPU parallelization should be used, False
        otherwise
        """
        return self._backend_choice == Backend.PARALLEL_CPU

    def get_gpu_available(self) -> bool:
        """Check GPU availability (independent of configuration).

        Returns: -------- bool     True if GPU hardware detected, False otherwise
        """
        return self._gpu_available

    def set_backend(self, backend: Backend | str) -> None:
        """Change backend at runtime (affects new JIT compilations).

        Note: Already-compiled Numba functions are not recompiled. This affects function
        selection for new computations only.

        Parameters: ----------- backend : Backend or str     Backend to switch to
        ('serial_cpu', 'parallel_cpu', 'gpu_cuda')

        Raises: ------- ValueError     If invalid backend specified RuntimeError     If
        GPU requested but not available TypeError     If backend not Backend enum or str
        """
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        elif not isinstance(backend, Backend):
            raise TypeError(f"backend must be Backend or str, got {type(backend)}")

        # Check GPU availability for GPU backend
        if backend == Backend.GPU_CUDA and not self._gpu_available:
            raise RuntimeError(f"Cannot switch to {backend.value}: GPU not available")

        self._backend_choice = backend

    def get_config_summary(self) -> dict:
        """Get current configuration as dictionary.

        Returns: -------- dict     Configuration state and GPU availability
        """
        return {
            "backend": self._backend_choice.value,
            "gpu_available": self._gpu_available,
            "using_gpu": self.get_use_gpu(),
            "using_parallel_cpu": self.get_use_parallel_cpu(),
        }


# Global configuration instance (read from environment at module load)
_config = AerodynamicsConfig()


def get_config() -> AerodynamicsConfig:
    """Get global aerodynamics configuration instance.

    Returns: -------- AerodynamicsConfig     Global configuration object
    """
    return _config


def select_biot_savart_function():
    """Select appropriate Biot-Savart function based on configuration.

    This function is called at runtime to choose between: 1. Serial CPU (fastest
    baseline) 2. Parallel CPU (Numba prange - usually slower) 3. GPU CUDA (50-150×
    speedup if available)

    Returns: -------- callable     Selected function for computing Biot-Savart
    velocities

    Raises: ------- RuntimeError     If configuration requirements cannot be met
    """
    config = get_config()
    backend = config.get_backend()

    if backend == Backend.GPU_CUDA:
        # Import GPU version (lazy import to avoid GPU overhead if not used)
        try:
            from pterasoftware._aerodynamics_functions_cuda import (
                collapsed_velocities_from_line_vortices_cuda,
            )

            return collapsed_velocities_from_line_vortices_cuda
        except ImportError as e:
            raise RuntimeError(
                f"GPU backend selected but CUDA modules unavailable: {e}"
            )

    elif backend == Backend.PARALLEL_CPU:
        # Import parallel CPU version
        from pterasoftware._aerodynamics_functions import (
            _collapsed_velocities_from_line_vortices_parallel,
        )

        return _collapsed_velocities_from_line_vortices_parallel

    else:  # Backend.SERIAL_CPU (default)
        # Import serial CPU version (baseline)
        from pterasoftware._aerodynamics_functions import (
            _collapsed_velocities_from_line_vortices,
        )

        return _collapsed_velocities_from_line_vortices


def get_gpu_memory_pool_if_available():
    """Get GPUMemoryPool class if GPU available and configured.

    Returns: -------- type or None     GPUMemoryPool class if GPU available, None
    otherwise

    Note: This is for Phase 2 persistent memory optimization. Only available if GPU is
    configured and detected.
    """
    config = get_config()

    if not config.get_use_gpu():
        return None

    try:
        from pterasoftware._gpu_memory_manager import GPUMemoryPool

        return GPUMemoryPool
    except ImportError:
        return None


# Public API
__all__ = [
    "Backend",
    "AerodynamicsConfig",
    "get_config",
    "select_biot_savart_function",
    "get_gpu_memory_pool_if_available",
]
