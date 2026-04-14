"""
Custom GPU kernels for sync metrics.

Provides Metal (Apple Silicon) and CUDA (NVIDIA) implementations
for metrics that cannot be efficiently expressed with torch operations
(e.g., PLI, wPLI — non-linear per-timepoint operations).
"""

# Metal availability (Apple Silicon via PyObjC)
try:
    import Metal as _Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# CUDA availability (NVIDIA via CuPy)
try:
    import cupy as _cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

__all__ = ["METAL_AVAILABLE", "CUPY_AVAILABLE"]
