#!/usr/bin/env python
# coding=utf-8

"""
Coherence (Coh) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate


class Coh(BaseMetric):
    """
    Coherence connectivity metric.
    
    Coherence measures the linear relationship between two signals in the
    frequency domain, normalized by their power.
    
    Mathematical formulation:
        Coh = |⟨XY*⟩|² / (⟨|X|²⟩⟨|Y|²⟩)
    
    References
    ----------
    Nunez, P. L., & Srinivasan, R. (2006). Electric fields of the brain:
    the neurophysics of EEG. Oxford University Press.
    """
    
    name = "coh"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Coherence.
        
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
            Coherence connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Coherence."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                               np.nansum(amp, axis=3)))
        return con
