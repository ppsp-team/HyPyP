#!/usr/bin/env python
# coding=utf-8

"""
Phase Locking Value (PLV) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from .base import multiply_conjugate_torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class PLV(BaseMetric):
    """
    Phase Locking Value connectivity metric.
    
    PLV measures the consistency of phase differences between two signals
    across time, regardless of amplitude.
    
    Mathematical formulation:
        PLV = |⟨e^(i(φₓ-φᵧ))⟩|
    
    References
    ----------
    Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
    Measuring phase synchrony in brain signals. Human brain mapping, 8(4), 194-208.
    """
    
    name = "plv"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Phase Locking Value.
        
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
            PLV connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        if self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_cuda(self, complex_signal, n_samp, transpose_axes):
        """CUDA kernel for PLV."""
        from .kernels.cuda_phase import plv_cuda
        return plv_cuda(complex_signal)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of PLV."""
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = abs(dphi) / n_samp
        return con

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of PLV with parallel epoch processing.

        Fuses the 4 einsum operations of multiply_conjugate into a single
        loop pass, avoiding intermediate tensor allocations. Uses prange
        for parallelization across epochs.

        This is significantly faster than numpy for PLV because:
        1. Zero intermediate allocations (numpy creates 4 temporary tensors)
        2. Single-pass accumulation in CPU registers
        3. prange parallelizes across epochs
        """
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        return _plv_numba_kernel(c, s, n_samp)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of PLV.

        Uses multiply_conjugate_torch to offload the 4 einsum operations
        to GPU. The einsum contracts the time dimension, so the output is
        directly (E, F, C, C) — no 5D intermediate tensor.

        MPS uses float32; CPU/CUDA uses float64.
        """
        device = self._device
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)

        # Normalize to unit magnitude (phase only)
        phase = sig / torch.abs(sig)
        c, s = phase.real, phase.imag

        # Cross-spectrum with time contraction: (E, F, C, C)
        dphi = multiply_conjugate_torch(c, s)

        con = torch.abs(dphi) / n_samp
        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _plv_numba_kernel(c, s, n_samp):
        """
        Fused PLV computation: multiply_conjugate + abs in a single pass.

        Computes |sum_t(z_i(t) * conj(z_j(t)))| / T for all (i,j) pairs,
        where z = c + i*s (unit-magnitude phase signal).

        Parallelized over epochs with prange.
        """
        n_ep, n_freq, n_ch, n_t = c.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                for i in range(n_ch):
                    for j in range(n_ch):
                        re_sum = 0.0
                        im_sum = 0.0
                        for t in range(n_t):
                            # z_i * conj(z_j) = (c_i*c_j + s_i*s_j) + i*(s_i*c_j - c_i*s_j)
                            re_sum += c[e, f, i, t] * c[e, f, j, t] + s[e, f, i, t] * s[e, f, j, t]
                            im_sum += s[e, f, i, t] * c[e, f, j, t] - c[e, f, i, t] * s[e, f, j, t]
                        con[e, f, i, j] = np.sqrt(re_sum**2 + im_sum**2) / n_samp

        return con
