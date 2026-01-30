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
from typing import Literal

import numpy as np


# Available backends
BACKENDS = Literal['numpy', 'numba', 'torch']


def detect_backend() -> str:
    """
    Detects the best available backend for computation.
    
    Returns
    -------
    str
        The name of the best available backend ('numpy', 'numba', or 'torch').
        Falls back to 'numpy' if no accelerated backend is available.
    """
    # Try torch first (GPU acceleration)
    try:
        import torch
        if torch.cuda.is_available():
            return 'torch'
    except ImportError:
        pass
    
    # Try numba (JIT compilation)
    try:
        import numba
        return 'numba'
    except ImportError:
        pass
    
    # Fallback to numpy
    return 'numpy'


def multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate efficiently.
    
    This helper function performs matrix multiplication between complex arrays
    represented by their real and imaginary parts, collapsing the last dimension.
    
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
    This function implements the formula:
    product = (real × real.T + imag × imag.T) - i(real × imag.T - imag × real.T)
    
    Using einsum for efficient computation without explicitly creating complex arrays.
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
    
    Notes
    -----
    This function uses a different einsum formula than multiply_conjugate:
    'jilm,jimk->jilkm' instead of 'jilm,jimk->jilk'
    
    This preserves the time dimension (m) in the output, which is necessary for 
    computing metrics that require individual time point values rather than 
    time-averaged products.
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
    
    This helper function performs matrix multiplication between complex arrays
    represented by their real and imaginary parts, collapsing the last dimension.
    Unlike multiply_conjugate, this computes z1 * z2 instead of z1 * conj(z2).
    
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
    
    Notes
    -----
    This function implements the formula for z1 * z2:
    product = (real × real.T - imag × imag.T) + i(real × imag.T + imag × real.T)
    
    Using einsum for efficient computation without explicitly creating complex arrays.
    This is used in the adjusted circular correlation (accorr) metric.
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
    backend : str, optional
        The computation backend to use ('numpy', 'numba', or 'torch').
        Default is 'numpy'.
    
    Attributes
    ----------
    backend : str
        The active computation backend.
    name : str
        The name of the metric (class attribute to be defined by subclasses).
    """
    
    name: str = "base"
    
    def __init__(self, backend: str = 'numpy'):
        if backend == 'auto':
            backend = detect_backend()
        
        # Validate backend availability
        if backend == 'numba':
            try:
                import numba
            except ImportError:
                warnings.warn(
                    f"Backend 'numba' unavailable, falling back to 'numpy'",
                    UserWarning,
                    stacklevel=2
                )
                backend = 'numpy'
        elif backend == 'torch':
            try:
                import torch
            except ImportError:
                warnings.warn(
                    f"Backend 'torch' unavailable, falling back to 'numpy'",
                    UserWarning,
                    stacklevel=2
                )
                backend = 'numpy'
        
        self.backend = backend
    
    @abstractmethod
    def compute(self, complex_signal: np.ndarray, n_samp: int, 
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute the connectivity metric.
        
        Parameters
        ----------
        complex_signal : np.ndarray
            Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times).
            Note: This is the already reshaped signal from compute_sync.
            
        n_samp : int
            Number of time samples.
            
        transpose_axes : tuple
            Axes to transpose for matrix multiplication.
        
        Returns
        -------
        con : np.ndarray
            Connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
            Note: This is the raw output before swapaxes and epochs_average.
        """
        pass
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of the metric."""
        raise NotImplementedError(
            f"NumPy backend not implemented for {self.__class__.__name__}"
        )
    
    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """Numba implementation of the metric."""
        raise NotImplementedError(
            f"Numba backend not implemented for {self.__class__.__name__}"
        )
    
    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """PyTorch implementation of the metric."""
        raise NotImplementedError(
            f"PyTorch backend not implemented for {self.__class__.__name__}"
        )
