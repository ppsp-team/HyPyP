#!/usr/bin/env python
# coding=utf-8

"""
Weighted Phase Lag Index (wPLI) connectivity metric.
"""

import numpy as np

from .base import BaseMetric, multiply_conjugate_time, TORCH_AVAILABLE, NUMBA_AVAILABLE

if TORCH_AVAILABLE:
    import torch

if NUMBA_AVAILABLE:
    from numba import njit, prange


class WPLI(BaseMetric):
    """
    Weighted Phase Lag Index (wPLI) connectivity metric.

    wPLI is a modification of PLI that weights the contribution of each
    phase difference by its distance from the real axis. This reduces
    sensitivity to noise-induced perturbations of small phase differences.

    Mathematical formulation:
        wPLI = |⟨|Im(XY*)| sign(Im(XY*))⟩| / ⟨|Im(XY*)|⟩

    References
    ----------
    Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz,
    C. M. (2011). An improved index of phase-synchronization for electro-
    physiological data in the presence of volume-conduction, noise and
    sample-size bias. NeuroImage, 55(4), 1548-1565.
    """

    name = "wpli"

    def compute(self, complex_signal: np.ndarray, n_samp: int,
                transpose_axes: tuple) -> np.ndarray:
        """
        Compute Weighted Phase Lag Index.

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
            wPLI connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
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

    def _compute_metal(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """Metal compute shader for wPLI on Apple Silicon GPU."""
        from .kernels.metal_phase import wpli_metal
        return wpli_metal(complex_signal)

    def _compute_cuda(self, complex_signal: np.ndarray, n_samp: int,
                      transpose_axes: tuple) -> np.ndarray:
        """CUDA kernel for wPLI on NVIDIA GPU."""
        from .kernels.cuda_phase import wpli_cuda
        return wpli_cuda(complex_signal)

    def _compute_numba(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        Numba JIT implementation of wPLI with fused kernel.

        Computes Im(X_i * conj(X_j)) and accumulates |mean(Im)| / mean(|Im|)
        directly in the inner loop. No 5D tensor.
        Uses the simplification |Im|*sign(Im) = Im for the numerator.

        Note: wPLI uses the raw signal (not phase-normalized), unlike PLI.
        """
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        return _wpli_numba_kernel(c, s)

    def _compute_numpy(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """NumPy implementation of Weighted Phase Lag Index."""
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        # |Im(x)| * sign(Im(x)) = Im(x) for all real x
        con_num = np.abs(np.mean(np.imag(dphi), axis=4))
        con_den = np.mean(np.abs(np.imag(dphi)), axis=4)
        con_den = np.where(con_den == 0, 1, con_den)
        con = con_num / con_den
        return con

    def _compute_torch(self, complex_signal: np.ndarray, n_samp: int,
                       transpose_axes: tuple) -> np.ndarray:
        """
        PyTorch implementation of Weighted Phase Lag Index.

        Chunks computation by (epoch, freq) to avoid materializing the full
        5D tensor ``(E, F, C, C, T)`` which exceeds MPS INT_MAX at high
        channel counts. Each chunk ``(C, C, T)`` stays well under the limit.

        Computes Im(X_i * conj(X_j)) = s_i*c_j - c_i*s_j directly with
        2 real einsum instead of 4 complex einsum. Halves GPU memory per chunk.
        Uses simplified numerator: |mean(Im)| instead of |mean(|Im|*sign(Im))|.

        MPS uses float32 precision; CPU/CUDA uses float64.
        """
        device = self._device
        float_type = torch.float32 if device == 'mps' else torch.float64
        complex_type = torch.complex64 if device == 'mps' else torch.complex128

        sig = torch.from_numpy(complex_signal).to(device=device, dtype=complex_type)
        n_epochs, n_freq, n_ch, n_times = sig.shape
        c, s = sig.real, sig.imag

        con = torch.zeros((n_epochs, n_freq, n_ch, n_ch),
                          device=device, dtype=float_type)

        # Chunk by epoch — each chunk is (F, C, C, T), 5x fewer iterations
        # than (epoch, freq) chunking. Falls back to double loop if chunk
        # would exceed MPS INT_MAX.
        chunk_elements = n_freq * n_ch * n_ch * n_times
        if device == 'mps' and chunk_elements > 2_000_000_000:
            # Fallback: (epoch, freq) chunking for very large configs
            formula = 'it,jt->ijt'
            for e in range(n_epochs):
                for f in range(n_freq):
                    c_ef = c[e, f]
                    s_ef = s[e, f]
                    im_dphi = torch.einsum(formula, s_ef, c_ef) - \
                              torch.einsum(formula, c_ef, s_ef)
                    con_num = torch.abs(torch.mean(im_dphi, dim=-1))
                    con_den = torch.mean(torch.abs(im_dphi), dim=-1)
                    con_den = torch.where(con_den == 0, torch.ones_like(con_den), con_den)
                    con[e, f] = con_num / con_den
        else:
            # Fast path: epoch-only chunking
            formula = 'fit,fjt->fijt'
            for e in range(n_epochs):
                c_e = c[e]  # (F, C, T)
                s_e = s[e]
                im_dphi = torch.einsum(formula, s_e, c_e) - \
                          torch.einsum(formula, c_e, s_e)
                # |Im| * sign(Im) = Im
                con_num = torch.abs(torch.mean(im_dphi, dim=-1))
                con_den = torch.mean(torch.abs(im_dphi), dim=-1)
                con_den = torch.where(con_den == 0, torch.ones_like(con_den), con_den)
                con[e] = con_num / con_den

        return con.cpu().numpy()


# Numba JIT kernel (module-level for caching)
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _wpli_numba_kernel(c, s):
        """
        Fused wPLI: weighted sign of Im(cross-spectrum).

        wPLI = |mean_t(Im)| / mean_t(|Im|)
        Uses the simplification |Im|*sign(Im) = Im.

        No 5D tensor — O(C²) memory instead of O(C² × T).
        """
        n_ep, n_freq, n_ch, n_t = c.shape
        con = np.zeros((n_ep, n_freq, n_ch, n_ch))

        for e in prange(n_ep):
            for f in range(n_freq):
                for i in range(n_ch):
                    for j in range(i, n_ch):
                        im_sum = 0.0
                        abs_sum = 0.0
                        for t in range(n_t):
                            im = s[e, f, i, t] * c[e, f, j, t] \
                               - c[e, f, i, t] * s[e, f, j, t]
                            im_sum += im
                            abs_sum += abs(im)
                        if abs_sum > 0:
                            val = abs(im_sum) / abs_sum
                        else:
                            val = 0.0
                        con[e, f, i, j] = val
                        con[e, f, j, i] = val

        return con
