#!/usr/bin/env python
# coding=utf-8

"""
Coherence (Coh) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from .base import multiply_conjugate_torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class Coh(BaseMetric):
    """
    Coherence connectivity metric.
    
    Coherence measures the linear relationship between two signals in the
    frequency domain, normalized by their power.
    
    Mathematical formulation:
        Coh = |⟨XY*⟩|² / (⟨|X|²⟩⟨|Y|²⟩)
    
    References
    ----------
    Nunez, P. L., & Srinivasan, R. (2006). Electric fields of the brain:
    the neurophysics of EEG. Oxford University Press.
    """
    
    name = "coh"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Coherence.
        
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
            Coherence connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        if self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_cuda(self, complex_signal, n_samp, transpose_axes):
        """CUDA kernel for Coherence."""
        from .kernels.cuda_amplitude import coh_cuda
        return coh_cuda(complex_signal)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Coherence."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                               np.nansum(amp, axis=3)))
        return con

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of Coherence with parallel epoch processing.

        Fuses cross-spectrum and power normalization into a single loop pass.
        Accumulates numerator (cross-spectrum) and denominator (power) in
        CPU registers — zero intermediate tensor allocations.
        """
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        return _coh_numba_kernel(c, s)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of Coherence.

        Uses multiply_conjugate_torch for the cross-spectrum numerator
        and torch.einsum for the power normalization denominator.
        MPS uses float32; CPU/CUDA uses float64.
        """
        device = self._device
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        c, s = sig.real, sig.imag

        # Cross-spectrum: sum_t(X_i * conj(X_j)) — contracts time dim
        dphi = multiply_conjugate_torch(c, s)

        # Power normalization: sqrt(sum|X_i|² * sum|X_j|²)
        amp = torch.abs(sig) ** 2
        power = torch.nansum(amp, dim=3)
        den = torch.sqrt(torch.einsum('efi,efj->efij', power, power))

        con = torch.abs(dphi) / den
        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _coh_numba_kernel(c, s):
        """
        Fused Coherence: cross-spectrum + power normalization in one pass.

        For each (epoch, freq, i, j):
            cross = sum_t (c_i*c_j + s_i*s_j) + i*(s_i*c_j - c_i*s_j)
            pow_i = sum_t (c_i² + s_i²)
            pow_j = sum_t (c_j² + s_j²)
            coh = |cross| / sqrt(pow_i * pow_j)
        """
        n_ep, n_freq, n_ch, n_t = c.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                # Pre-compute power per channel
                power = np.zeros(n_ch)
                for ch in range(n_ch):
                    p = 0.0
                    for t in range(n_t):
                        p += c[e, f, ch, t] ** 2 + s[e, f, ch, t] ** 2
                    power[ch] = p

                # Cross-spectrum for all pairs
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        re_sum = 0.0
                        im_sum = 0.0
                        for t in range(n_t):
                            re_sum += c[e, f, i, t] * c[e, f, j, t] + s[e, f, i, t] * s[e, f, j, t]
                            im_sum += s[e, f, i, t] * c[e, f, j, t] - c[e, f, i, t] * s[e, f, j, t]
                        denom = np.sqrt(power[i] * power[j])
                        if denom > 0:
                            val = np.sqrt(re_sum ** 2 + im_sum ** 2) / denom
                        else:
                            val = 0.0
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val  # symmetry

        return con
