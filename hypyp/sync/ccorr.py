#!/usr/bin/env python
# coding=utf-8

"""
Circular Correlation (CCorr) connectivity metric.
"""

import numpy as np
from scipy.stats import circmean

from .base import BaseMetric


class CCorr(BaseMetric):
    """
    Circular Correlation connectivity metric.
    
    CCorr measures the circular correlation coefficient between the phases
    of two signals, using the circular mean for phase centering.
    
    References
    ----------
    Fisher, N. I. (1995). Statistical analysis of circular data. Cambridge University Press.
    """
    
    name = "ccorr"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Circular Correlation.
        
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
            CCorr connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of CCorr."""
        n_epoch = complex_signal.shape[0]
        n_freq = complex_signal.shape[1]
        n_ch_total = complex_signal.shape[2]
        
        angle = np.angle(complex_signal)
        mu_angle = circmean(angle, high=np.pi, low=-np.pi, axis=3).reshape(
            n_epoch, n_freq, n_ch_total, 1
        )
        angle = np.sin(angle - mu_angle)

        formula = 'nilm,nimk->nilk'
        con = np.abs(np.einsum(formula, angle, angle.transpose(transpose_axes)) /
                     np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), 
                                       np.sum(angle ** 2, axis=3))))
        return con
