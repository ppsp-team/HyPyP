"""
Custom GPU kernels for sync metrics.

Provides Metal (Apple Silicon) and CUDA (NVIDIA) implementations
for metrics that cannot be efficiently expressed with torch operations
(e.g., PLI, wPLI — non-linear per-timepoint operations).
"""

import warnings

# Metal availability (Apple Silicon via PyObjC)
try:
    import Metal as _Metal

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
except Exception as exc:
    # Installed but unusable (a missing shared library, for instance):
    # disable the backend instead of making `import hypyp` fail
    warnings.warn(
        "The Metal bindings are installed but could not be imported "
        f"({type(exc).__name__}: {exc}). The Metal backend is disabled.",
        UserWarning,
    )
    METAL_AVAILABLE = False

# CUDA availability (NVIDIA via CuPy)
try:
    import cupy as _cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
except Exception as exc:
    warnings.warn(
        "cupy is installed but could not be imported "
        f"({type(exc).__name__}: {exc}). The CUDA kernel backend is disabled.",
        UserWarning,
    )
    CUPY_AVAILABLE = False

__all__ = ["METAL_AVAILABLE", "CUPY_AVAILABLE"]
