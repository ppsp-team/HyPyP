#!/usr/bin/env python
# coding=utf-8

"""
Phase Lag Index (PLI) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate_time, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class PLI(BaseMetric):
    """
    Phase Lag Index (PLI) connectivity metric.

    PLI measures the asymmetry of the distribution of instantaneous phase
    differences. It is insensitive to volume conduction as it ignores
    zero-lag interactions.

    Mathematical formulation:
        PLI = |⟨sign(Im(XY*))⟩|

    References
    ----------
    Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index:
    assessment of functional connectivity from multi channel EEG and MEG
    with diminished bias from common sources. Human Brain Mapping, 28(11),
    1178-1193.
    """

    name = "pli"

    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Phase Lag Index.

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
            PLI connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        if self._backend == 'metal':
            return self._compute_metal(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'cuda_kernel':
            return self._compute_cuda(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'torch':
            return self._compute_torch(complex_signal, n_samp, transpose_axes)
        elif self._backend == 'numba':
            return self._compute_numba(complex_signal, n_samp, transpose_axes)
        return self._compute_numpy(complex_signal, n_samp, transpose_axes)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Phase Lag Index."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.mean(np.sign(np.imag(dphi)), axis=4))
        return con

    def _compute_metal(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Metal compute shader implementation of PLI on Apple Silicon GPU.

        Each GPU thread processes one (epoch×freq, channel_pair) combination,
        looping over timepoints. No intermediate tensor — O(1) memory per thread.
        ~3x faster than numba on 256+ channel data.

        Requires: pip install pyobjc-framework-Metal
        """
        from .kernels.metal_phase import pli_metal
        return pli_metal(complex_signal)

    def _compute_cuda(self, complex_signal: np.ndarray, n_samp: int,
                      transpose_axes: tuple) -> np.ndarray:
        """
        CUDA kernel implementation of PLI on NVIDIA GPU.

        Requires: pip install cupy-cuda12x
        """
        from .kernels.cuda_phase import pli_cuda
        return pli_cuda(complex_signal)

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of PLI with fused kernel.

        Computes Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j and sign()
        directly in the inner loop, eliminating the 5D intermediate tensor.
        Memory: O(C²) instead of O(C² × T). Parallelized over epochs.

        Note: PLI uses the raw signal (not phase-normalized). The sign()
        operation makes the result invariant to amplitude anyway.
        """
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        return _pli_numba_kernel(c, s)

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of Phase Lag Index using per-channel broadcast.

        For each channel i, broadcasts s_i and c_i against ALL channels j
        simultaneously using element-wise ops. No einsum, no gather copies.

        s[:,:,i:i+1,:] is a contiguous slice (zero-copy view on GPU).
        The broadcast produces (E, F, C, T) intermediates — the same size
        as the input, not C² × T like the einsum approach.

        MPS uses float32 precision; CPU/CUDA uses float64.
        """
        device = self._device
        float_type = torch.float32 if device == 'mps' else torch.float64
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        n_epochs, n_freq, n_ch, n_times = sig.shape
        c, s = sig.real, sig.imag  # (E, F, C, T)

        con = torch.zeros((n_epochs, n_freq, n_ch, n_ch),
                          device=device, dtype=float_type)

        for i in range(n_ch):
            # s[:,:,i:i+1,:] is a VIEW (contiguous slice), no copy
            # Broadcasting against (E, F, C, T) produces (E, F, C, T)
            im = s[:, :, i:i+1, :] * c - c[:, :, i:i+1, :] * s  # (E, F, C, T)
            con[:, :, i, :] = torch.abs(torch.mean(torch.sign(im), dim=-1))

        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _pli_numba_kernel(c, s):
        """
        Fused PLI: sign(Im(cross-spectrum)) averaged over time.

        Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j
        PLI = |mean_t(sign(Im))|

        No 5D tensor — O(C²) memory instead of O(C² × T).
        """
        n_ep, n_freq, n_ch, n_t = c.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        sign_sum = 0.0
                        for t in range(n_t):
                            im = s[e, f, i, t] * c[e, f, j, t] \
                               - c[e, f, i, t] * s[e, f, j, t]
                            if im > 0:
                                sign_sum += 1.0
                            elif im < 0:
                                sign_sum -= 1.0
                        val = abs(sign_sum) / n_t
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val

        return con
