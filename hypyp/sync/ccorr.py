#!/usr/bin/env python
# coding=utf-8

"""
Circular Correlation (CCorr) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


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
        if self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_cuda(self, complex_signal, n_samp, transpose_axes):
        """CUDA kernel for CCorr."""
        from .kernels.cuda_phase import ccorr_cuda
        return ccorr_cuda(complex_signal)
    
    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        NumPy implementation of CCorr.

        Uses inline circular mean (atan2(mean(sin), mean(cos))) instead of
        scipy.stats.circmean to remove the scipy dependency from this module.
        Mathematically identical: circmean(x, high=pi, low=-pi) == atan2(mean(sin(x)), mean(cos(x))).
        """
        n_epoch = complex_signal.shape[0]
        n_freq = complex_signal.shape[1]
        n_ch_total = complex_signal.shape[2]

        angle = np.angle(complex_signal)
        # Circular mean: atan2(mean(sin(angle)), mean(cos(angle)))
        # Equivalent to scipy.stats.circmean(angle, high=pi, low=-pi, axis=3)
        mu_angle = np.arctan2(
            np.mean(np.sin(angle), axis=3),
            np.mean(np.cos(angle), axis=3),
        ).reshape(n_epoch, n_freq, n_ch_total, 1)
        angle = np.sin(angle - mu_angle)

        formula = 'nilm,nimk->nilk'
        con = np.abs(np.einsum(formula, angle, angle.transpose(transpose_axes)) /
                     np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3),
                                       np.sum(angle ** 2, axis=3))))
        return con

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of CCorr using angle-free reformulation.

        Works with cos(φ)/sin(φ) directly, avoiding all transcendental
        functions inside the JIT-compiled loops. Uses prange for
        parallelization across epochs and symmetry exploitation (upper triangle).
        """
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        return _ccorr_numba_kernel(c, s)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of CCorr using angle-free reformulation.

        Instead of extracting phase angles (torch.angle + arctan2 + sin),
        works directly with cos(φ) and sin(φ) from the unit-phase signal.

        The circular centering sin(φ - μ) is reformulated as:
            d(t) = sin(φ(t)) * C̄ - cos(φ(t)) * S̄
        where C̄ = mean(cos(φ)), S̄ = mean(sin(φ)).
        The normalization factor R = √(C̄² + S̄²) cancels in the correlation.

        This eliminates all transcendental functions (angle, arctan2, sin)
        after the initial phase normalization, improving MPS float32 precision.
        """
        device = self._device
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)

        # Unit-phase signal: same as PLV
        phase = sig / torch.abs(sig)
        c, s = phase.real, phase.imag  # cos(φ), sin(φ)

        # Circular mean components (no atan2)
        C_bar = torch.mean(c, dim=3, keepdim=True)  # (E, F, C, 1)
        S_bar = torch.mean(s, dim=3, keepdim=True)  # (E, F, C, 1)

        # Centered signal: d(t) = s(t)*C_bar - c(t)*S_bar
        # R factor cancels in correlation, no division needed
        d = s * C_bar - c * S_bar  # (E, F, C, T)

        # Correlation via einsum
        formula = 'efit,efjt->efij'
        num = torch.einsum(formula, d, d)
        sum_sq = torch.sum(d ** 2, dim=3)
        den = torch.sqrt(torch.einsum('efi,efj->efij', sum_sq, sum_sq))

        con = torch.abs(num / den)
        return con.cpu().numpy()

    def _compute_torch_cpu_circmean(self, complex_signal: np.ndarray, n_samp: int,
                                     transpose_axes: tuple) -> np.ndarray:
        """
        Hybrid approach: circular mean in float64 on CPU, correlation on GPU.

        Computes the precision-sensitive circular mean (arctan2) in float64
        on CPU, then transfers the centered signal to GPU for the einsum.
        Kept for comparison with the angle-free reformulation.
        """
        device = self._device
        float_type = torch.float32 if device == 'mps' else torch.float64

        # Step 1: Circular mean in float64 on CPU (precision-critical)
        angle = np.angle(complex_signal)
        mu_angle = np.arctan2(
            np.mean(np.sin(angle), axis=3),
            np.mean(np.cos(angle), axis=3),
        ).reshape(complex_signal.shape[0], complex_signal.shape[1],
                  complex_signal.shape[2], 1)
        centered = np.sin(angle - mu_angle)  # float64, precise

        # Step 2: Transfer centered signal to GPU for einsum
        d = torch.from_numpy(centered).to(device=device, dtype=float_type)

        formula = 'efit,efjt->efij'
        num = torch.einsum(formula, d, d)
        sum_sq = torch.sum(d ** 2, dim=3)
        den = torch.sqrt(torch.einsum('efi,efj->efij', sum_sq, sum_sq))

        con = torch.abs(num / den)
        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _ccorr_numba_kernel(c, s):
        """
        Angle-free CCorr kernel: no transcendental functions inside loops.

        Uses d_i(t) = s_i(t)*C_bar_i - c_i(t)*S_bar_i for circular centering.
        The normalization factor R cancels in the correlation.
        Exploits symmetry: CCorr(i,j) == CCorr(j,i).
        """
        n_ep, n_freq, n_ch, n_t = c.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                # Pre-compute mean(cos) and mean(sin) per channel
                C_bar = np.zeros(n_ch)
                S_bar = np.zeros(n_ch)
                for ch in range(n_ch):
                    c_sum = 0.0
                    s_sum = 0.0
                    for t in range(n_t):
                        c_sum += c[e, f, ch, t]
                        s_sum += s[e, f, ch, t]
                    C_bar[ch] = c_sum / n_t
                    S_bar[ch] = s_sum / n_t

                # Correlation for upper triangle + diagonal
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        num = 0.0
                        den_i = 0.0
                        den_j = 0.0
                        for t in range(n_t):
                            # d(t) = s(t)*C_bar - c(t)*S_bar
                            di = s[e, f, i, t] * C_bar[i] - c[e, f, i, t] * S_bar[i]
                            dj = s[e, f, j, t] * C_bar[j] - c[e, f, j, t] * S_bar[j]
                            num += di * dj
                            den_i += di * di
                            den_j += dj * dj
                        denom = np.sqrt(den_i * den_j)
                        if denom > 0:
                            val = np.abs(num) / denom
                        else:
                            val = 0.0
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val  # symmetry

        return con
