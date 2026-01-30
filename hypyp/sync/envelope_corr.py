#!/usr/bin/env python
# coding=utf-8

"""
Envelope Correlation (EnvCorr) connectivity metric.
"""

import numpy as np

from .base import BaseMetric


class EnvCorr(BaseMetric):
    """
    Envelope Correlation connectivity metric.
    
    Envelope Correlation measures the correlation between the amplitude
    envelopes of two signals across time.
    
    Mathematical formulation:
        EnvCorr = correlation(|X|, |Y|) over time samples
    
    The implementation normalizes the amplitudes by subtracting the mean
    and dividing by the product of standard deviations.
    
    References
    ----------
    Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel, A. K.
    (2012). Large-scale cortical correlation structure of spontaneous
    oscillatory activity. Nature Neuroscience, 15(6), 884-890.
    """
    
    name = "envcorr"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Envelope Correlation.
        
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
            Envelope Correlation connectivity matrix with shape
            (n_epoch, n_freq, 2*n_ch, 2*n_ch).
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
        """NumPy implementation of Envelope Correlation."""
        n_epoch, n_freq, n_ch_total = complex_signal.shape[:3]
        env = np.abs(complex_signal)
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, n_ch_total, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
              np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))
        return con
