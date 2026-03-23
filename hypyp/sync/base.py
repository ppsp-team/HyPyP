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

    def __init__(self, optimization: Optional[str] = None):
        self.optimization = optimization
        self._backend, self._device = self._resolve_optimization(optimization)

    @staticmethod
    def _resolve_optimization(optimization: Optional[str] = None) -> tuple:
        """
        Resolves an optimization value to (backend, device).

        Implements fallback logic with warnings when requested
        optimization is not available.

        Parameters
        ----------
        optimization : str or None
            Requested optimization strategy:

            - ``None``: standard numpy, no acceleration (default).
            - ``'auto'``: best available backend — tries torch first, then
              numba, then falls back to numpy. No warning is emitted.
            - ``'numba'``: JIT-compiled loops via numba. Falls back to numpy
              with a UserWarning if numba is not installed.
            - ``'torch'``: PyTorch tensors with auto-detected GPU (see
              ``_resolve_torch`` for device priority). Falls back to numpy
              with a UserWarning if torch is not installed.

        Returns
        -------
        backend : str
            One of ``'numpy'``, ``'numba'``, ``'torch'``.
        device : str
            One of ``'cpu'``, ``'mps'``, ``'cuda'``.

        Notes
        -----
        Fallback cascade for ``'auto'``:
            torch (best available device) → numba → numpy

        Fallback cascade for ``'torch'`` or ``'numba'`` when unavailable:
            requested backend → numpy (with UserWarning)
        """
        if optimization is None:
            return 'numpy', 'cpu'

        if optimization == 'auto':
            if TORCH_AVAILABLE:
                return BaseMetric._resolve_torch()
            if NUMBA_AVAILABLE:
                return 'numba', 'cpu'
            return 'numpy', 'cpu'

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
                return BaseMetric._resolve_torch()
            warnings.warn(
                "torch not installed, falling back to numpy. "
                "Install with: poetry install --with optim_torch",
                UserWarning, stacklevel=3
            )
            return 'numpy', 'cpu'

        raise ValueError(
            f"Unknown optimization '{optimization}'. "
            f"Options: None, 'auto', 'numba', 'torch'"
        )

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
