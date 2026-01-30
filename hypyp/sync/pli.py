#!/usr/bin/env python
# coding=utf-8

"""
Phase Lag Index (PLI) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate_time


class PLI(BaseMetric):
    """
    Phase Lag Index (PLI) connectivity metric.
    
    PLI measures the asymmetry of the distribution of instantaneous phase
    differences. It is insensitive to volume conduction as it ignores
    zero-lag interactions.
    
    Mathematical formulation:
        PLI = |⟨sign(Im(XY*))⟩|
    
    References
    ----------
    Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index:
    assessment of functional connectivity from multi channel EEG and MEG
    with diminished bias from common sources. Human Brain Mapping, 28(11),
    1178-1193.
    """
    
    name = "pli"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Phase Lag Index.
        
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
            PLI connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
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
        """NumPy implementation of Phase Lag Index."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.mean(np.sign(np.imag(dphi)), axis=4))
        return con
