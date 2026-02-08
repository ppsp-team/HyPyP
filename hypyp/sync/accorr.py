#!/usr/bin/env python
# coding=utf-8

"""
Adjusted Circular Correlation (ACCorr) connectivity metric.

Optimizations (numba, torch) originally developed by @m2march
as part of BrainHack Montreal 2026 (see PR #246).
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from .base import (
    BaseMetric, multiply_conjugate, multiply_product,
    TORCH_AVAILABLE, MPS_AVAILABLE, NUMBA_AVAILABLE,
)

# Conditional imports for optional backends
if TORCH_AVAILABLE:
    import torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class ACCorr(BaseMetric):
    """
    Adjusted Circular Correlation connectivity metric.

    ACCorr computes the circular correlation coefficient with optimized
    phase centering for each channel pair individually, providing a more
    accurate measure than standard circular correlation.

    Parameters
    ----------
    optimization : str, optional
        Optimization strategy: None, 'auto', 'numba', 'torch'.
        See BaseMetric for fallback behavior.
    show_progress : bool, optional
        If True, display a progress bar during computation. Default is True.

    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """

    name = "accorr"

    def __init__(self, optimization: Optional[str] = None,
                 show_progress: bool = True):
        super().__init__(optimization)
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
            ACCorr connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        if self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        else:
            return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        NumPy implementation of ACCorr with precompute optimization.

        Pre-computes m_adj and n_adj for ALL pairs by reusing the cross_conj
        and cross_prod matrices, reducing computation in the denominator loop.
        """
        n_epochs, n_freq, n_ch_total, n_times = complex_signal.shape

        # Numerator (vectorized)
        z = complex_signal / np.abs(complex_signal)
        c, s = np.real(z), np.imag(z)

        cross_conj = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        cross_prod = multiply_product(c, s, transpose_axes=transpose_axes)

        r_minus = np.abs(cross_conj)
        r_plus = np.abs(cross_prod)
        num = r_minus - r_plus

        # Pre-compute m_adj and n_adj for ALL pairs
        mean_diff_all = np.angle(cross_conj / n_times)
        mean_sum_all = np.angle(cross_prod / n_times)

        n_adj_all = -1 * (mean_diff_all - mean_sum_all) / 2
        m_adj_all = mean_diff_all + n_adj_all

        # Denominator (lighter loop - just lookups)
        angle = np.angle(complex_signal)
        den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))

        total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
        pbar = tqdm(total=total_pairs, desc="    accorr (denominator)",
                    disable=not self.show_progress, leave=False)

        for i in range(n_ch_total):
            for j in range(i, n_ch_total):
                alpha1 = angle[:, :, i, :]
                alpha2 = angle[:, :, j, :]

                m_adj = m_adj_all[:, :, i, j, np.newaxis]
                n_adj = n_adj_all[:, :, i, j, np.newaxis]

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

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba-optimized implementation of ACCorr with precompute.

        Uses numba JIT compilation for the denominator loop.
        Note: parallelization is currently disabled due to a dependency conflict.
        """
        n_epochs, n_freq, n_ch_total, n_times = complex_signal.shape

        # Numerator (vectorized, same as numpy)
        z = complex_signal / np.abs(complex_signal)
        c, s = np.real(z), np.imag(z)

        cross_conj = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        cross_prod = multiply_product(c, s, transpose_axes=transpose_axes)

        r_minus = np.abs(cross_conj)
        r_plus = np.abs(cross_prod)
        num = r_minus - r_plus

        # Pre-compute m_adj and n_adj
        mean_diff_all = np.angle(cross_conj / n_times)
        mean_sum_all = np.angle(cross_prod / n_times)

        n_adj_all = -1 * (mean_diff_all - mean_sum_all) / 2
        m_adj_all = mean_diff_all + n_adj_all

        # Denominator via numba JIT
        angle = np.angle(complex_signal)
        den = _accorr_den_numba(n_epochs, n_freq, n_ch_total, angle,
                                m_adj_all, n_adj_all)

        den = np.where(den == 0, 1, den)
        con = num / den

        return con

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of ACCorr with precompute optimization.

        Uses torch tensor operations on the resolved device (cpu/mps/cuda).
        MPS uses float32 precision; cpu/cuda uses float64.
        """
        device = self._device

        if device == 'mps':
            float_type = torch.float32
            complex_type = torch.complex64
        else:
            float_type = torch.float64
            complex_type = torch.complex128

        complex_tensor = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        n_epochs, n_freq, n_ch_total, n_times = complex_tensor.shape

        # Numerator (vectorized with torch einsum)
        z = complex_tensor / torch.abs(complex_tensor)
        c, s = z.real, z.imag

        formula = 'efit,efjt->efij'
        cross_conj = (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * \
                     (torch.einsum(formula, c, s) - torch.einsum(formula, s, c))
        cross_prod = (torch.einsum(formula, c, c) - torch.einsum(formula, s, s)) + 1j * \
                     (torch.einsum(formula, c, s) + torch.einsum(formula, s, c))

        r_minus = torch.abs(cross_conj)
        r_plus = torch.abs(cross_prod)
        num = r_minus - r_plus

        # Pre-compute m_adj and n_adj
        mean_diff_all = torch.angle(cross_conj / n_times)
        mean_sum_all = torch.angle(cross_prod / n_times)

        n_adj_all = -0.5 * (mean_diff_all - mean_sum_all)
        m_adj_all = mean_diff_all + n_adj_all

        # Denominator - loop on device
        angle = torch.angle(complex_tensor)
        den = torch.zeros((n_epochs, n_freq, n_ch_total, n_ch_total),
                          device=device, dtype=float_type)

        total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
        pbar = tqdm(total=total_pairs, desc="    accorr_torch (denominator)",
                    disable=not self.show_progress, leave=False)

        for i in range(n_ch_total):
            for j in range(i, n_ch_total):
                alpha1 = angle[:, :, i, :]
                alpha2 = angle[:, :, j, :]

                m_adj = m_adj_all[:, :, i, j].unsqueeze(-1)
                n_adj = n_adj_all[:, :, i, j].unsqueeze(-1)

                x_sin = torch.sin(alpha1 - m_adj)
                y_sin = torch.sin(alpha2 - n_adj)

                den_ij = 2 * torch.sqrt(torch.sum(x_sin**2, dim=2) * torch.sum(y_sin**2, dim=2))
                den[:, :, i, j] = den_ij
                den[:, :, j, i] = den_ij

                pbar.update(1)

        pbar.close()

        den = torch.where(den == 0, torch.ones_like(den), den)
        con = num / den

        return con.cpu().numpy()


# Numba JIT-compiled helper (defined at module level for caching)
if NUMBA_AVAILABLE:
    # TODO(@m2march): research why parallelization is not working
    @njit(parallel=False, cache=True)
    def _accorr_den_numba(n_epochs, n_freq, n_ch_total, angle, m_adj_all, n_adj_all):
        """Numba JIT-compiled denominator calculation for accorr."""
        den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))

        for i in range(den.shape[2]):
            for j in range(i, den.shape[3]):
                alpha1 = angle[:, :, i, :]
                alpha2 = angle[:, :, j, :]

                m_adj = m_adj_all[:, :, i, j]
                n_adj = n_adj_all[:, :, i, j]

                x = alpha1.copy()
                for xi in range(x.shape[0]):
                    for xj in range(x.shape[1]):
                        for xk in range(x.shape[2]):
                            x[xi, xj, xk] -= m_adj[xi, xj]
                x_sin = np.sin(x)

                y = alpha2.copy()
                for yi in range(y.shape[0]):
                    for yj in range(y.shape[1]):
                        for yk in range(y.shape[2]):
                            y[yi, yj, yk] -= n_adj[yi, yj]
                y_sin = np.sin(y)

                den_ij = 2 * np.sqrt(np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2))
                den[:, :, i, j] = den_ij
                den[:, :, j, i] = den_ij

        return den
