#!/usr/bin/env python
# coding=utf-8

"""
Imaginary Coherence (ImCoh) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate


class ImCoh(BaseMetric):
    """
    Imaginary Coherence connectivity metric.
    
    Imaginary Coherence uses only the imaginary part of the cross-spectrum,
    making it robust to volume conduction effects.
    
    Mathematical formulation:
        ImCoh = Im(⟨XY*⟩) / √(⟨|X|²⟩⟨|Y|²⟩)
    
    References
    ----------
    Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).
    Identifying true brain interaction from EEG data using the imaginary part
    of coherency. Clinical Neurophysiology, 115(10), 2292-2307.
    """
    
    name = "imcoh"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Imaginary Coherence.
        
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
            Imaginary Coherence connectivity matrix with shape
            (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Imaginary Coherence."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                        np.nansum(amp, axis=3)))
        return con
