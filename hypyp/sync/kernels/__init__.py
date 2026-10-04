"""
Custom GPU kernels for sync metrics.

Provides Metal (Apple Silicon) and CUDA (NVIDIA) implementations
for metrics that cannot be efficiently expressed with torch operations
(e.g., PLI, wPLI — non-linear per-timepoint operations).
"""

import warnings

# A package that is not installed is skipped silently; one that is installed
# but fails to load (a missing shared library, for instance) disables the
# backend with a warning instead of making `import hypyp` fail.

# Metal availability (Apple Silicon via PyObjC)
try:
    import Metal as _Metal

    METAL_AVAILABLE = True
except Exception as exc:
    METAL_AVAILABLE = False
    if not (isinstance(exc, ModuleNotFoundError) and exc.name == "Metal"):
        warnings.warn(
            "The Metal bindings are installed but could not be loaded "
            f"({type(exc).__name__}: {exc}). The Metal backend is disabled.",
            UserWarning,
        )

# CUDA availability (NVIDIA via CuPy)
try:
    import cupy as _cp

    CUPY_AVAILABLE = True
except Exception as exc:
    CUPY_AVAILABLE = False
    if not (isinstance(exc, ModuleNotFoundError) and exc.name == "cupy"):
        warnings.warn(
            "cupy is installed but could not be loaded "
            f"({type(exc).__name__}: {exc}). The CUDA kernel backend is disabled.",
            UserWarning,
        )

__all__ = ["METAL_AVAILABLE", "CUPY_AVAILABLE"]
