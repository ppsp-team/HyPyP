"""
Shared CUDA dispatch logic for all pairwise sync metric kernels.

Provides ``run_pairwise_kernel``, the common launch + memory-management
routine used by every metric-specific CUDA kernel module
(``cuda_phase``, ``cuda_amplitude``, ``cuda_accorr``).

Uses CuPy ``RawKernel`` for inline CUDA source. All kernels run in
float64 — the NVIDIA A100 reference target has 9.7 TFLOPS of fp64
throughput, so the precision/speed trade-off favours fp64 there.
"""

import numpy as np

from . import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def run_pairwise_kernel(complex_signal, get_kernel_fn):
    """
    Shared dispatch for pairwise CUDA kernels.

    Builds the upper-triangle channel-pair index list, transfers the real
    and imaginary parts to the device as float64, launches one thread per
    ``(epoch*freq, pair)`` tuple, and reads the result back. Computing
    pairwise — rather than materialising the full ``(E, F, C, C, T)``
    cross-spectrum — keeps device memory bounded at high channel counts.

    Parameters
    ----------
    complex_signal : np.ndarray, shape (E, F, C, T)
        Complex analytic signals (epochs, freqs, channels, samples).
    get_kernel_fn : callable -> cupy.RawKernel
        Lazily compiles (and caches) the metric-specific CUDA kernel.

    Returns
    -------
    np.ndarray, shape (E, F, C, C), float64
        Connectivity matrix per (epoch, freq).

    Notes
    -----
    The CuPy default memory pool is explicitly freed before returning, so
    repeated calls in a tight loop don't accumulate device allocations.
    """
    kernel = get_kernel_fn()

    E, F, C, T = complex_signal.shape
    n_ef = E * F

    c_flat = cp.asarray(
        np.ascontiguousarray(np.real(complex_signal).reshape(n_ef, C, T)),
        dtype=cp.float64)
    s_flat = cp.asarray(
        np.ascontiguousarray(np.imag(complex_signal).reshape(n_ef, C, T)),
        dtype=cp.float64)

    # Upper-triangle pair indices
    idx_i, idx_j = [], []
    for i in range(C):
        for j in range(i, C):
            idx_i.append(i)
            idx_j.append(j)
    pairs_i = cp.asarray(np.array(idx_i, dtype=np.int32))
    pairs_j = cp.asarray(np.array(idx_j, dtype=np.int32))
    n_pairs = len(idx_i)

    out = cp.zeros((n_ef, C, C), dtype=cp.float64)

    total_threads = n_ef * n_pairs
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (s_flat, c_flat, out, pairs_i, pairs_j,
         n_ef, C, T, n_pairs)
    )

    result = cp.asnumpy(out).reshape(E, F, C, C)

    # Explicit cleanup: force immediate GPU memory release (not relying on GC)
    cp.get_default_memory_pool().free_all_blocks()

    return result
