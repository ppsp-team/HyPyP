#!/usr/bin/env python
# coding=utf-8

"""
Power Correlation (PowCorr) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


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
        if self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_cuda(self, complex_signal, n_samp, transpose_axes):
        """CUDA kernel for Power Correlation."""
        from .kernels.cuda_amplitude import powcorr_cuda
        return powcorr_cuda(complex_signal)

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

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of Power Correlation.

        Fuses mean-centering, Pearson numerator, and denominator into a
        single loop pass with parallel epoch processing. Zero intermediate
        tensor allocations — accumulates in CPU registers.
        """
        env = np.abs(complex_signal) ** 2
        return _powcorr_numba_kernel(env)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of Power Correlation.

        Computes power as abs(sig)² then Pearson correlation entirely in
        float32 (MPS) or float64 (CPU/CUDA). No complex arithmetic on GPU
        after envelope extraction.

        Uses L2-normalization instead of separate numerator/denominator to
        avoid float32 underflow: power values are already small (~|z|²),
        and the denominator sqrt(sum(dev²_i) * sum(dev²_j)) involves 4th
        powers that underflow in float32.
        """
        device = self._device
        float_type = torch.float32 if device == 'mps' else torch.float64
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        # Power: |z|² via abs then square
        env = (torch.abs(sig).to(dtype=float_type)) ** 2
        del sig  # free complex tensor

        # Center the power
        mu = torch.mean(env, dim=3, keepdim=True)
        env = env - mu

        # L2-normalize per channel: env_hat = env / ||env||
        # Then dot(env_hat_i, env_hat_j) = Pearson correlation directly
        norm = torch.sqrt(torch.sum(env ** 2, dim=3, keepdim=True))
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        env = env / norm

        con = torch.einsum('efit,efjt->efij', env, env)
        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _powcorr_numba_kernel(env):
        """
        Fused Pearson correlation on power envelopes.

        For each (epoch, freq):
          1. Pre-compute per-channel mean and sum-of-squared-deviations
          2. Pearson correlation for upper triangle, copy by symmetry
        """
        n_ep, n_freq, n_ch, n_t = env.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                # Pre-compute mean and sum_sq per channel
                mu = np.zeros(n_ch)
                ss = np.zeros(n_ch)
                for ch in range(n_ch):
                    s = 0.0
                    for t in range(n_t):
                        s += env[e, f, ch, t]
                    mu[ch] = s / n_t
                    sq = 0.0
                    for t in range(n_t):
                        d = env[e, f, ch, t] - mu[ch]
                        sq += d * d
                    ss[ch] = sq

                # Pearson correlation for upper triangle
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        num = 0.0
                        for t in range(n_t):
                            num += (env[e, f, i, t] - mu[i]) * (env[e, f, j, t] - mu[j])
                        denom = np.sqrt(ss[i] * ss[j])
                        val = num / denom if denom > 0 else 0.0
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val  # symmetry

        return con
