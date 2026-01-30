#!/usr/bin/env python
# coding=utf-8

"""
Phase Locking Value (PLV) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate


class PLV(BaseMetric):
    """
    Phase Locking Value connectivity metric.
    
    PLV measures the consistency of phase differences between two signals
    across time, regardless of amplitude.
    
    Mathematical formulation:
        PLV = |⟨e^(i(φₓ-φᵧ))⟩|
    
    References
    ----------
    Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
    Measuring phase synchrony in brain signals. Human brain mapping, 8(4), 194-208.
    """
    
    name = "plv"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Phase Locking Value.
        
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
            PLV connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        if self.backend == 'numpy':
            return self._compute_numpy(complex_signal, n_samp, transpose_axes)
        elif self.backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        elif self.backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        else:
            return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of PLV."""
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = abs(dphi) / n_samp
        return con
