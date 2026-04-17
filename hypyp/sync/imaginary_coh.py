#!/usr/bin/env python
# coding=utf-8

"""
Imaginary Coherence (ImCoh) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from .base import multiply_conjugate_torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class ImCoh(BaseMetric):
    """
    Imaginary Coherence connectivity metric.
    
    Imaginary Coherence uses only the imaginary part of the cross-spectrum,
    making it robust to volume conduction effects.
    
    Mathematical formulation:
        ImCoh = Im(⟨XY*⟩) / √(⟨|X|²⟩⟨|Y|²⟩)
    
    References
    ----------
    Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).
    Identifying true brain interaction from EEG data using the imaginary part
    of coherency. Clinical Neurophysiology, 115(10), 2292-2307.
    """
    
    name = "imcoh"
    
    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Imaginary Coherence.
        
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
            Imaginary Coherence connectivity matrix with shape
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
        """CUDA kernel for Imaginary Coherence."""
        from .kernels.cuda_amplitude import imcoh_cuda
        return imcoh_cuda(complex_signal)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Imaginary Coherence."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                        np.nansum(amp, axis=3)))
        return con

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of Imaginary Coherence.

        Same fused kernel as Coh but returns |Im(cross-spectrum)| instead
        of |cross-spectrum|. This keeps only the non-zero-lag component,
        rejecting volume conduction artifacts.
        """
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        return _imcoh_numba_kernel(c, s)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of Imaginary Coherence.

        Computes Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j directly with
        2 real-valued einsum, instead of building the full complex cross-spectrum
        via multiply_conjugate_torch (4 einsum + complex tensor). This halves
        GPU memory usage and avoids MPS corruption on large signals.

        MPS uses float32; CPU/CUDA uses float64.
        """
        device = self._device
        float_type = torch.float32 if device == 'mps' else torch.float64

        sig = torch.from_numpy(complex_signal).to(device=device,
                                                   dtype=torch.complex64 if device == 'mps'
                                                   else torch.complex128)
        c, s = sig.real, sig.imag

        # Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j — 2 einsum, no complex tensor
        formula = 'efit,efjt->efij'
        im_cross = torch.einsum(formula, s, c) - torch.einsum(formula, c, s)

        # Power normalization: sqrt(sum|X_i|² * sum|X_j|²)
        amp = c ** 2 + s ** 2  # |X|² without creating complex abs
        power = torch.sum(amp, dim=3)
        den = torch.sqrt(torch.einsum('efi,efj->efij', power, power))

        con = torch.abs(im_cross) / den
        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _imcoh_numba_kernel(c, s):
        """
        Fused ImCoh: cross-spectrum imaginary part + power normalization.

        Same as Coh kernel but returns |im_sum| / sqrt(pow_i * pow_j)
        instead of sqrt(re_sum² + im_sum²) / sqrt(pow_i * pow_j).
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

                # Cross-spectrum imaginary part for all pairs
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        im_sum = 0.0
                        for t in range(n_t):
                            # Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j
                            im_sum += s[e, f, i, t] * c[e, f, j, t] - c[e, f, i, t] * s[e, f, j, t]
                        denom = np.sqrt(power[i] * power[j])
                        if denom > 0:
                            val = np.abs(im_sum) / denom
                        else:
                            val = 0.0
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val  # symmetry

        return con
