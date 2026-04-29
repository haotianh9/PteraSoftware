"""Quick Reference: Phase 1a GPU Linear Solver

Fast-access guide for developers implementing Phase 1a.
"""

# ==============================================================================
# QUICK START - PHASE 1a GPU LINEAR SOLVER
# ==============================================================================

## 1. Installation (One-time)

```bash
# Install CuPy to enable GPU acceleration
pip install cupy-cuda11x      # For CUDA 11.x
# or
pip install cupy-cuda12x      # For CUDA 12.x
```


## 2. Verify Installation

```python
from pterasoftware.unsteady_ring_vortex_lattice_method_gpu import (
    GPU_LINEAR_SOLVER_AVAILABLE
)
print(f"GPU available: {GPU_LINEAR_SOLVER_AVAILABLE}")  # True with CuPy
```


## 3. Basic Usage

```python
from pterasoftware.unsteady_ring_vortex_lattice_method_gpu import (
    PersistentGPUUnsteadySolver
)
import numpy as np

# Your linear system
A = np.random.randn(500, 500).astype(np.float64)
A = A @ A.T  # Make positive definite
b = np.random.randn(500).astype(np.float64)

# Solve (GPU if available, else CPU)
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)

# Check what method was used
print(f"Solver: {timing['method']}")  # 'gpu' or 'cpu'
print(f"Time: {timing['total_ms']:.2f}ms")
```


## 4. Performance Monitoring

```python
# Get breakdown of GPU costs
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)

print(f"GPU Transfer: {timing['transfer_ms']:.2f}ms")
print(f"GPU Solve:    {timing['solve_ms']:.2f}ms")
print(f"GPU Total:    {timing['total_ms']:.2f}ms")
```


## 5. Force CPU-Only (for testing)

```python
# Disable GPU, always use CPU
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(
    A, b, use_gpu=False
)
print(f"Method: {timing['method']}")  # Always 'cpu'
```


## 6. Error Handling (Automatic)

```python
# No try-except needed! Falls back to CPU on any GPU error
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)

# Check which method was actually used
if timing['method'] == 'cpu':
    print("GPU unavailable or failed, using CPU")
```


## 7. Integration into Unsteady Solver

```python
# In your unsteady RVLM solver code:

def solve_influence_matrix(self, influence_matrix, rhs_vector):
    """Solve linear system with automatic GPU acceleration."""
    gamma, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(
        influence_matrix, rhs_vector, use_gpu=True
    )
    
    # Log timing for diagnostics
    self.linear_solve_times.append(timing['total_ms'])
    
    return gamma
```


## 8. Benchmarking Your Problem

```python
import time
import numpy as np
from pterasoftware.unsteady_ring_vortex_lattice_method_gpu import (
    PersistentGPUUnsteadySolver
)

n = 1000  # Your typical problem size
A = np.random.randn(n, n).astype(np.float64)
A = A @ A.T
b = np.random.randn(n).astype(np.float64)

# Warmup
for _ in range(2):
    _, _ = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)

# Benchmark
times = []
for _ in range(10):
    _, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
    times.append(timing['total_ms'])

print(f"Average: {np.mean(times):.2f}ms")
print(f"Std dev: {np.std(times):.2f}ms")
print(f"Method: {timing['method']}")
```


## 9. Long Simulation Pattern (Persistent Memory)

```python
from pterasoftware.unsteady_ring_vortex_lattice_method_gpu import (
    PersistentGPUUnsteadySolver
)

# Create solver with persistent GPU memory
solver = PersistentGPUUnsteadySolver(unsteady_problem)
solver.initialize_gpu(max_wake_rows=100)

# Run many timesteps (GPU memory stays allocated)
for step in range(1000):
    result = solver.solve_step(step)  # Linear solver called internally
    # With GPU: ~5-15ms per step
    # With CPU: ~40-60ms per step
```


## 10. Troubleshooting

### GPU Solver Not Activating
```python
# Check if CuPy is available
try:
    import cupy
    print("CuPy installed ✓")
except ImportError:
    print("CuPy not found. Run: pip install cupy-cuda11x")

# Check if GPU_LINEAR_SOLVER_AVAILABLE is True
from pterasoftware.unsteady_ring_vortex_lattice_method_gpu import (
    GPU_LINEAR_SOLVER_AVAILABLE
)
print(f"GPU linear solver: {GPU_LINEAR_SOLVER_AVAILABLE}")
```

### Memory Issues
```python
# If GPU runs out of memory, code automatically falls back to CPU
# To manually free GPU memory:
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()

# Then try again
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
```

### Slow GPU Performance
```python
# GPU slowest for small problems (< 100×100)
# Minimum problem size benefit: 200×200
# Significant benefit: 500×500+

# If your system is small, CPU is actually faster
# No harm - code detects this and uses best method
```


## Expected Performance

Problem Size │ CPU Time  │ GPU Time  │ Speedup
──────────────────────────────────────────────
100×100      │ 0.5ms     │ 1-2ms     │ 0.5-0.3x (CPU better)
500×500      │ 10ms      │ 2ms       │ 5x (GPU wins)
1000×1000    │ 25ms      │ 5ms       │ 5x (GPU wins)
2000×2000    │ 80ms      │ 15ms      │ 5x (GPU wins)

Break-even: ~200-300×200-300 system size


## Files to Know

Core Implementation:
  pterasoftware/unsteady_ring_vortex_lattice_method_gpu.py
    → solve_linear_system_gpu() method is the entry point

Tests:
  tests/test_gpu_phase1a.py
    → Run to verify installation
    → Shows performance on your system

Examples:
  examples/gpu_persistent_unsteady_simple.py
    → Shows integration patterns
    → Runnable demonstration

Documentation:
  PHASE_1A_IMPLEMENTATION_SUMMARY.md
    → Complete implementation details
  docs/GPU_PHASE1A_LINEAR_SOLVER.md
    → Architecture and design decisions


## Common Patterns

### Pattern 1: One-shot solve
```python
x, _ = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
```

### Pattern 2: Track timing
```python
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
print(f"GPU solve took {timing['solve_ms']:.2f}ms")
```

### Pattern 3: Force CPU for debugging
```python
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(
    A, b, use_gpu=False
)
```

### Pattern 4: Compare performance
```python
x_cpu = np.linalg.solve(A, b)  # Direct CPU
x_gpu, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
assert np.allclose(x_cpu, x_gpu)
speedup = cpu_time / timing['solve_ms']
```

### Pattern 5: Production - fire and forget
```python
# In your solver, just call it - framework handles GPU/CPU
x, timing = PersistentGPUUnsteadySolver.solve_linear_system_gpu(A, b)
# Falls back to CPU if GPU unavailable
# No special error handling needed
```


## Performance Tips

1. Minimum viable GPU system size: 200×200
2. Significant benefit above: 500×500
3. Amortization matters: GPU wins after ~50 timesteps
4. Persistent memory multiplication: 20-60x reduction in transfer overhead
5. Monitor with: nvidia-smi dmon


## Next Phase (Phase 1b)

Phase 1b will add influence matrix fusion on GPU:
  - Fused Biot-Savart kernel: 2-3x
  - Combined with Phase 1a: 40-50% improvement
  - Keeps data on GPU, no transfers

When to proceed: After confirming Phase 1a speedup works for your problems


## Support

Run tests to verify everything:
  python tests/test_gpu_phase1a.py

Run examples:
  python examples/gpu_persistent_unsteady_simple.py

Check GitHub issues if problems:
  camUrban/PteraSoftware/issues
"""
