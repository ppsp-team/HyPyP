#!/usr/bin/env python
# coding=utf-8

"""
Adjusted Circular Correlation (ACorr) connectivity metric.
"""

import numpy as np
from tqdm import tqdm

from .base import BaseMetric, multiply_conjugate, multiply_product


class ACorr(BaseMetric):
    """
    Adjusted Circular Correlation connectivity metric.
    
    ACorr computes the circular correlation coefficient with optimized
    phase centering for each channel pair individually, providing a more
    accurate measure than standard circular correlation.
    
    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """
    
    name = "accorr"
    
    def __init__(self, backend: str = 'numpy', show_progress: bool = True):
        super().__init__(backend)
        self.show_progress = show_progress
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Adjusted Circular Correlation.
        
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
            ACorr connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
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
        """NumPy implementation of ACorr (hybrid approach)."""
        n_epochs = complex_signal.shape[0]
        n_freq = complex_signal.shape[1]
        n_ch_total = complex_signal.shape[2]
        
        # Numerator (vectorized)
        z = complex_signal / np.abs(complex_signal)
        c, s = np.real(z), np.imag(z)
        
        cross_conj = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        r_minus = np.abs(cross_conj)
        
        cross_prod = multiply_product(c, s, transpose_axes=transpose_axes)
        r_plus = np.abs(cross_prod)
        
        num = r_minus - r_plus
        
        # Denominator (loop)
        angle = np.angle(complex_signal)
        den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))
        
        total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
        pbar = tqdm(total=total_pairs, desc="    accorr (denominator)", 
                    disable=not self.show_progress, leave=False)
        
        for i in range(n_ch_total):
            for j in range(i, n_ch_total):
                alpha1 = angle[:, :, i, :]
                alpha2 = angle[:, :, j, :]
                
                phase_diff = alpha1 - alpha2
                phase_sum = alpha1 + alpha2
                
                mean_diff = np.angle(np.mean(np.exp(1j * phase_diff), axis=2, keepdims=True))
                mean_sum = np.angle(np.mean(np.exp(1j * phase_sum), axis=2, keepdims=True))
                
                n_adj = -1 * (mean_diff - mean_sum) / 2
                m_adj = mean_diff + n_adj
                
                x_sin = np.sin(alpha1 - m_adj)
                y_sin = np.sin(alpha2 - n_adj)
                
                den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
                den[:, :, i, j] = den_ij
                den[:, :, j, i] = den_ij
                
                pbar.update(1)
        
        pbar.close()
        
        den = np.where(den == 0, 1, den)
        con = num / den
        
        return con
