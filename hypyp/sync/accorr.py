#!/usr/bin/env python
# coding=utf-8

"""
Adjusted Circular Correlation (ACCorr) connectivity metric.

ACCorr computes the circular correlation between two phase time-series with
per-pair phase centering, providing a more accurate inter-brain synchrony
estimate than standard circular correlation (ccorr).

Reference: Zimmermann et al. (2024). *Imaging Neuroscience*, 2.
https://doi.org/10.1162/imag_a_00350

Credits
-------
The ``precompute`` optimization strategy (vectorized numerator + loop denominator
with pre-computed per-pair adjustments) was contributed by **Martín A. Miguel**
([@m2march](https://github.com/m2march)) during BrainHack Montréal 2026 (PR #246).

The numba JIT and PyTorch GPU/MPS backends were also developed by @m2march and
integrated into the modular sync architecture in PR #250.
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
                 priority: Optional[list] = None,
                 show_progress: bool = True):
        super().__init__(optimization, priority)
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
        if self._backend == 'metal':
            return self._compute_metal(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        else:
            return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_metal(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """Metal compute shader for ACCorr on Apple Silicon GPU."""
        from .kernels.metal_accorr import accorr_metal
        return accorr_metal(complex_signal)

    def _compute_cuda(self, complex_signal: np.ndarray, n_samp: int,
                      transpose_axes: tuple) -> np.ndarray:
        """CUDA kernel for ACCorr on NVIDIA GPU."""
        from .kernels.cuda_accorr import accorr_cuda
        return accorr_cuda(complex_signal)

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

        Uses numba JIT compilation with prange parallelization for the
        denominator loop.
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

    # Memory threshold for vectorized denominator (bytes). If the 5D tensor
    # (E, F, C, C, T) would exceed this, fall back to the loop-based approach.
    _VRAM_THRESHOLD = 2 * 1024**3  # 2 GB

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of ACCorr with precompute optimization.

        Uses torch tensor operations on the resolved device (cpu/mps/cuda).
        MPS uses float32 precision; cpu/cuda uses float64.

        The denominator is fully vectorized via broadcasting when the
        intermediate 5D tensor fits in memory (< _VRAM_THRESHOLD). Otherwise,
        falls back to a per-pair loop on device.
        """
        device = self._device

        if device == 'mps':
            float_type = torch.float32
            complex_type = torch.complex64
            bytes_per_elem = 4
        else:
            float_type = torch.float64
            complex_type = torch.complex128
            bytes_per_elem = 8

        complex_tensor = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        n_epochs, n_freq, n_ch_total, n_times = complex_tensor.shape

        # Numerator (vectorized with torch einsum)
        z = complex_tensor / torch.abs(complex_tensor)
        c, s = z.real, z.imag

        # Factorized: 4 einsum shared between cross_conj and cross_prod
        formula = 'efit,efjt->efij'
        cc = torch.einsum(formula, c, c)
        ss = torch.einsum(formula, s, s)
        cs = torch.einsum(formula, c, s)
        sc = torch.einsum(formula, s, c)
        cross_conj = (cc + ss) - 1j * (cs - sc)
        cross_prod = (cc - ss) + 1j * (cs + sc)

        r_minus = torch.abs(cross_conj)
        r_plus = torch.abs(cross_prod)
        num = r_minus - r_plus

        # Pre-compute m_adj and n_adj
        mean_diff_all = torch.angle(cross_conj / n_times)
        mean_sum_all = torch.angle(cross_prod / n_times)

        n_adj_all = -0.5 * (mean_diff_all - mean_sum_all)
        m_adj_all = mean_diff_all + n_adj_all

        # Denominator — choose vectorized or loop based on memory
        angle = torch.angle(complex_tensor)
        tensor_5d_bytes = n_epochs * n_freq * n_ch_total * n_ch_total * n_times * bytes_per_elem
        use_vectorized = tensor_5d_bytes < self._VRAM_THRESHOLD

        if use_vectorized:
            den = self._den_vectorized(angle, m_adj_all, n_adj_all, device, float_type)
        else:
            den = self._den_loop(angle, m_adj_all, n_adj_all, device, float_type,
                                 n_epochs, n_freq, n_ch_total)

        den = torch.where(den == 0, torch.ones_like(den), den)
        con = num / den

        return con.cpu().numpy()

    def _den_vectorized(self, angle, m_adj_all, n_adj_all, device, float_type):
        """
        Fully vectorized denominator via broadcasting.

        Broadcasts angle (E,F,C,T) against m_adj_all (E,F,C,C) to compute
        sin(angle_i - m_adj_{ij}) for all pairs simultaneously.

        Shape flow:
            angle[:,:,:,None,:] - m_adj_all[:,:,:,:,None]  -> (E, F, C, C, T)
            sin -> square -> sum over T -> sqrt -> 2 * sqrt(prod)
        """
        # angle: (E, F, C, T) -> (E, F, C, 1, T)
        # m_adj_all: (E, F, C, C) -> (E, F, C, C, 1)
        x_sin = torch.sin(angle.unsqueeze(3) - m_adj_all.unsqueeze(-1))  # (E,F,C,C,T)
        y_sin = torch.sin(angle.unsqueeze(2) - n_adj_all.unsqueeze(-1))  # (E,F,C,C,T)

        sum_x2 = torch.sum(x_sin ** 2, dim=-1)  # (E, F, C, C)
        sum_y2 = torch.sum(y_sin ** 2, dim=-1)  # (E, F, C, C)

        return 2.0 * torch.sqrt(sum_x2 * sum_y2)

    def _den_loop(self, angle, m_adj_all, n_adj_all, device, float_type,
                  n_epochs, n_freq, n_ch_total):
        """
        Loop-based denominator (fallback for large data).

        Iterates over channel pairs when the vectorized 5D tensor
        would exceed the VRAM threshold.
        """
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
        return den


# Numba JIT-compiled helper (defined at module level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _accorr_den_numba(n_epochs, n_freq, n_ch_total, angle, m_adj_all, n_adj_all):
        """
        Numba JIT-compiled denominator calculation for accorr.

        Uses prange for parallel iteration over channel pairs. The inner
        subtraction uses explicit loops (numba-compatible) instead of
        .copy() + broadcasting which caused allocation issues with prange.
        """
        den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))

        for i in prange(n_ch_total):
            for j in range(i, n_ch_total):
                # Compute sum of sin^2 for x and y directly, no temp arrays
                for ei in range(n_epochs):
                    for fi in range(n_freq):
                        m = m_adj_all[ei, fi, i, j]
                        n = n_adj_all[ei, fi, i, j]
                        sum_x2 = 0.0
                        sum_y2 = 0.0
                        for ti in range(angle.shape[3]):
                            sx = np.sin(angle[ei, fi, i, ti] - m)
                            sy = np.sin(angle[ei, fi, j, ti] - n)
                            sum_x2 += sx * sx
                            sum_y2 += sy * sy
                        den[ei, fi, i, j] = 2.0 * np.sqrt(sum_x2 * sum_y2)
                        den[ei, fi, j, i] = den[ei, fi, i, j]

        return den
