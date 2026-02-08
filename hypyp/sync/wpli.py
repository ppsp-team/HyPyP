#!/usr/bin/env python
# coding=utf-8

"""
Weighted Phase Lag Index (wPLI) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate_time


class WPLI(BaseMetric):
    """
    Weighted Phase Lag Index (wPLI) connectivity metric.
    
    wPLI is a modification of PLI that weights the contribution of each
    phase difference by its distance from the real axis. This reduces
    sensitivity to noise-induced perturbations of small phase differences.
    
    Mathematical formulation:
        wPLI = |⟨|Im(XY*)| sign(Im(XY*))⟩| / ⟨|Im(XY*)|⟩
    
    References
    ----------
    Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz,
    C. M. (2011). An improved index of phase-synchronization for electro-
    physiological data in the presence of volume-conduction, noise and
    sample-size bias. NeuroImage, 55(4), 1548-1565.
    """
    
    name = "wpli"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Weighted Phase Lag Index.
        
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
            wPLI connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Weighted Phase Lag Index."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        con_num = np.abs(np.mean(np.abs(np.imag(dphi)) * np.sign(np.imag(dphi)), axis=4))
        con_den = np.mean(np.abs(np.imag(dphi)), axis=4)
        con_den = np.where(con_den == 0, 1, con_den)
        con = con_num / con_den
        return con
