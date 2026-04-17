#!/usr/bin/env python
# coding=utf-8

"""
Base classes and helper functions for connectivity metrics.

| Option | Description |
| ------ | ----------- |
| title           | base.py |
| authors         | HyPyP Team |
| date            | 2026-01-30 |
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# Check optional dependency availability
try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Custom kernel backends
from .kernels import METAL_AVAILABLE, CUPY_AVAILABLE


# ---------------------------------------------------------------------------
# Benchmark-driven GPU backend priority for optimization='auto'
# ---------------------------------------------------------------------------
# Compiled from Mac M4 Max (131 rows) and Narval A100 (111 rows) benchmarks.
# Format: {metric_name: {platform: [gpu_backend_1, gpu_backend_2]}}
# First available GPU backend in the list wins.
#
# 'auto' selects the best GPU backend only. Users choose CPU strategies
# explicitly: optimization=None (numpy) or optimization='numba'.
#
# The priority can be overridden per-call via the `priority` parameter:
#   get_metric('plv', optimization='auto', priority=['metal', 'torch'])
#
# Rationale:
#   MPS — einsum metrics: torch wins (batched matrix ops via Apple MPS).
#          sign-based/accorr: Metal custom kernels win (sign() and circular
#          correlation are not vectorizable; torch OOMs at ≥512ch for PLI/wPLI).
#          No Metal kernels for einsum metrics (torch_mps dominates at all scales).
#   CUDA — cuda_kernel first for all metrics: torch_cuda is faster at small/
#          medium scale but OOMs at realistic_hd (512ch) due to large
#          intermediate tensors. cuda_kernel computes pairwise without
#          materializing the full output tensor.
AUTO_PRIORITY = {
    # einsum metrics — torch wins on MPS, cuda_kernel safe-first on CUDA
    # (torch OOMs at ≥512ch on CUDA; cuda_kernel computes pairwise)
    'plv':     {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    'ccorr':   {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    'coh':     {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    'imcoh':   {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    'envcorr': {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    'powcorr': {'mps': ['torch'], 'cuda': ['cuda_kernel', 'torch']},
    # sign-based — custom kernels beat torch on both platforms
    'pli':     {'mps': ['metal', 'torch'], 'cuda': ['cuda_kernel', 'torch']},
    'wpli':    {'mps': ['metal', 'torch'], 'cuda': ['cuda_kernel', 'torch']},
    # accorr — Metal wins on MPS (circular correlation), cuda_kernel safe on CUDA
    'accorr':  {'mps': ['metal', 'torch'], 'cuda': ['cuda_kernel', 'torch']},
}


def multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate efficiently.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate

    Notes
    -----
    Implements: product = (real x real.T + imag x imag.T) - i(real x imag.T - imag x real.T)
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def multiply_conjugate_time(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate without collapsing time dimension.

    Similar to multiply_conjugate, but preserves the time dimension, which is
    needed for certain connectivity metrics like wPLI.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate with time dimension preserved
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def multiply_product(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of two complex arrays (not conjugate) efficiently.

    Unlike multiply_conjugate, this computes z1 * z2 instead of z1 * conj(z2).
    Used in the adjusted circular correlation (accorr) metric.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array with itself (non-conjugate)
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) - \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) + 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) + \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def multiply_conjugate_torch(c, s):
    """
    Compute z * conj(z) using torch tensors, collapsing time dimension.

    Torch equivalent of :func:`multiply_conjugate`. Uses the einsum convention
    ``e=epoch, f=freq, i=ch_row, j=ch_col, t=time``.

    Parameters
    ----------
    c : torch.Tensor
        Real part, shape (E, F, C, T).
    s : torch.Tensor
        Imaginary part, shape (E, F, C, T).

    Returns
    -------
    torch.Tensor
        Complex product, shape (E, F, C, C).
    """
    formula = 'efit,efjt->efij'
    import torch
    return (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * \
           (torch.einsum(formula, c, s) - torch.einsum(formula, s, c))


def multiply_conjugate_time_torch(c, s):
    """
    Compute z * conj(z) using torch tensors, preserving time dimension.

    Torch equivalent of :func:`multiply_conjugate_time`. Produces a 5D tensor
    ``(E, F, C, C, T)`` — can be very large for high channel counts.

    Parameters
    ----------
    c : torch.Tensor
        Real part, shape (E, F, C, T).
    s : torch.Tensor
        Imaginary part, shape (E, F, C, T).

    Returns
    -------
    torch.Tensor
        Complex product, shape (E, F, C, C, T).
    """
    formula = 'efit,efjt->efijt'
    import torch
    return (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * \
           (torch.einsum(formula, c, s) - torch.einsum(formula, s, c))


class BaseMetric(ABC):
    """
    Abstract base class for connectivity metrics.

    All connectivity metrics should inherit from this class and implement
    the compute method.

    Parameters
    ----------
    optimization : str, optional
        Optimization strategy for computation. Options:
        - None: standard numpy (default)
        - 'auto': best available (torch > numba > numpy)
        - 'numba': numba JIT compilation (falls back to numpy if unavailable)
        - 'torch': PyTorch with auto-detected GPU (falls back gracefully)

    Attributes
    ----------
    optimization : str or None
        The requested optimization.
    name : str
        The name of the metric (class attribute to be defined by subclasses).
    """

    name: str = "base"

    def __init__(self, optimization: Optional[str] = None,
                 priority: Optional[list] = None):
        self.optimization = optimization
        self._priority = priority
        self._backend, self._device = self._resolve_optimization(
            optimization, priority
        )

    @classmethod
    def _resolve_optimization(cls, optimization: Optional[str] = None,
                              priority: Optional[list] = None) -> tuple:
        """
        Resolves an optimization value to (backend, device).

        Implements fallback logic with warnings when requested
        optimization is not available.

        Parameters
        ----------
        optimization : str or None
            Requested optimization strategy:

            - ``None``: standard numpy, no acceleration (default).
            - ``'auto'``: best available backend, selected per-metric from
              the ``AUTO_PRIORITY`` table (compiled from benchmarks).
              See ``_resolve_auto`` for details.
            - ``'numba'``: JIT-compiled loops via numba. Falls back to numpy
              with a UserWarning if numba is not installed.
            - ``'torch'``: PyTorch tensors with auto-detected GPU (see
              ``_resolve_torch`` for device priority). Falls back to numpy
              with a UserWarning if torch is not installed.
            - ``'metal'``: Apple Metal compute shaders. Falls back to numpy
              with a UserWarning if PyObjC Metal is not available.
            - ``'cuda_kernel'``: Custom CUDA kernels via CuPy. Falls back
              to numpy with a UserWarning if CuPy is not available.
        priority : list of str, optional
            Custom backend priority list for ``'auto'`` mode. Overrides
            the default ``AUTO_PRIORITY`` table for this call.
            Example: ``['metal', 'torch', 'numba']``.

        Returns
        -------
        backend : str
            One of ``'numpy'``, ``'numba'``, ``'torch'``, ``'metal'``,
            ``'cuda_kernel'``.
        device : str
            One of ``'cpu'``, ``'mps'``, ``'cuda'``.

        Notes
        -----
        Fallback cascade for ``'auto'`` (per-metric, per-platform):
            Iterates ``AUTO_PRIORITY[metric][platform]`` and returns the
            first available backend. Falls back to numba → numpy if no
            GPU backend is available.

        Fallback cascade for explicit backends when unavailable:
            requested backend → numpy (with UserWarning)
        """
        if optimization is None:
            return 'numpy', 'cpu'

        if optimization == 'auto':
            return cls._resolve_auto(priority)

        if optimization == 'numba':
            if NUMBA_AVAILABLE:
                return 'numba', 'cpu'
            warnings.warn(
                "numba not installed, falling back to numpy. "
                "Install with: poetry install --with optim_numba",
                UserWarning, stacklevel=3
            )
            return 'numpy', 'cpu'

        if optimization == 'torch':
            if TORCH_AVAILABLE:
                return cls._resolve_torch()
            warnings.warn(
                "torch not installed, falling back to numpy. "
                "Install with: poetry install --with optim_torch",
                UserWarning, stacklevel=3
            )
            return 'numpy', 'cpu'

        if optimization == 'metal':
            if METAL_AVAILABLE:
                return 'metal', 'mps'
            warnings.warn(
                "PyObjC Metal not available, falling back to numpy. "
                "Install with: pip install pyobjc-framework-Metal",
                UserWarning, stacklevel=3
            )
            return 'numpy', 'cpu'

        if optimization == 'cuda_kernel':
            if CUPY_AVAILABLE:
                return 'cuda_kernel', 'cuda'
            warnings.warn(
                "CuPy not available, falling back to numpy. "
                "Install with: pip install cupy-cuda12x",
                UserWarning, stacklevel=3
            )
            return 'numpy', 'cpu'

        raise ValueError(
            f"Unknown optimization '{optimization}'. "
            f"Options: None, 'auto', 'numba', 'torch', 'metal', 'cuda_kernel'"
        )

    @classmethod
    def _resolve_auto(cls, priority: Optional[list] = None) -> tuple:
        """
        Benchmark-driven backend selection, per metric and platform.

        Uses the ``AUTO_PRIORITY`` table compiled from Mac M4 Max and
        Narval A100 benchmarks. Iterates the priority list and returns
        the first available backend.

        Parameters
        ----------
        priority : list of str, optional
            Custom priority list overriding ``AUTO_PRIORITY`` for this call.

        Returns
        -------
        backend : str
            Selected backend name.
        device : str
            Associated device (``'cpu'``, ``'mps'``, or ``'cuda'``).

        Notes
        -----
        Platform detection: MPS → 'mps', CUDA → 'cuda', else 'cpu'.
        On CPU-only machines, warns and falls back to numba → numpy.
        """
        if MPS_AVAILABLE:
            platform = 'mps'
        elif CUDA_AVAILABLE:
            platform = 'cuda'
        else:
            # No GPU — warn and fall back to CPU
            warnings.warn(
                "No GPU available. optimization='auto' selects the best GPU "
                "backend. Use optimization='numba' for CPU parallelism or "
                "optimization=None for numpy.",
                UserWarning, stacklevel=4
            )
            if NUMBA_AVAILABLE:
                return 'numba', 'cpu'
            return 'numpy', 'cpu'

        if priority is None:
            priority = AUTO_PRIORITY.get(cls.name, {}).get(platform, [])

        for backend in priority:
            if backend == 'torch' and TORCH_AVAILABLE:
                return cls._resolve_torch()
            if backend == 'metal' and METAL_AVAILABLE:
                return 'metal', 'mps'
            if backend == 'cuda_kernel' and CUPY_AVAILABLE:
                return 'cuda_kernel', 'cuda'

        # No GPU backend from priority list available — fall back
        warnings.warn(
            f"No GPU backend available for {cls.name!r} on platform "
            f"'{platform}'. Falling back to CPU.",
            UserWarning, stacklevel=4
        )
        if NUMBA_AVAILABLE:
            return 'numba', 'cpu'
        return 'numpy', 'cpu'

    @staticmethod
    def _resolve_torch() -> tuple:
        """
        Resolves the best available torch device, with warnings if no GPU found.

        Device priority: MPS > CUDA > CPU.

        MPS (Metal Performance Shaders) is Apple Silicon's GPU backend and is
        checked first. CUDA is checked second for NVIDIA GPUs on Linux/Windows.
        The two are mutually exclusive — a machine will have one or the other,
        never both. If neither is available, torch runs on CPU with a warning.

        Returns
        -------
        backend : str
            Always ``'torch'``.
        device : str
            One of ``'mps'``, ``'cuda'``, or ``'cpu'``.

        Notes
        -----
        MPS uses 32-bit float precision (``torch.float32``), so numerical
        results may differ from CPU/CUDA (64-bit) by up to ~1e-5. Tests
        should apply a looser tolerance when MPS is the active device.
        """
        if MPS_AVAILABLE:
            return 'torch', 'mps'
        if CUDA_AVAILABLE:
            return 'torch', 'cuda'
        warnings.warn(
            "No GPU found, using torch on CPU",
            UserWarning, stacklevel=4
        )
        return 'torch', 'cpu'

    @abstractmethod
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute the connectivity metric.

        Parameters
        ----------
        complex_signal : np.ndarray
            Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times).
        n_samp : int
            Number of time samples.
        transpose_axes : tuple
            Axes to transpose for matrix multiplication.

        Returns
        -------
        con : np.ndarray
            Connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        pass
