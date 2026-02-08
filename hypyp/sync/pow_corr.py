#!/usr/bin/env python
# coding=utf-8

"""
Power Correlation (PowCorr) connectivity metric.
"""

import numpy as np

from .base import BaseMetric


class PowCorr(BaseMetric):
    """
    Power Correlation connectivity metric.
    
    Power Correlation measures the correlation between the power (squared
    amplitude) of two signals across time.
    
    Mathematical formulation:
        PowCorr = correlation(|X|², |Y|²) over time samples
    
    The implementation normalizes the power values by subtracting the mean
    and dividing by the product of standard deviations.
    
    References
    ----------
    Colclough, G. L., Woolrich, M. W., Tewarie, P. K., Brookes, M. J.,
    Quinn, A. J., & Smith, S. M. (2016). How reliable are MEG resting-state
    connectivity metrics? NeuroImage, 138, 284-293.
    """
    
    name = "powcorr"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Power Correlation.
        
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
            Power Correlation connectivity matrix with shape
            (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Power Correlation."""
        n_epoch, n_freq, n_ch_total = complex_signal.shape[:3]
        env = np.abs(complex_signal) ** 2
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, n_ch_total, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
              np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))
        return con
