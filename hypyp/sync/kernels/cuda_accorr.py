"""
CUDA kernel for ACCorr (Adjusted Circular Correlation).
Float64 for exact precision on NVIDIA GPUs.

ACCorr requires a custom dispatch (not run_pairwise_kernel) because
it needs an extra angle buffer for the sin^2 denominator in pass 2.
"""

import numpy as np

from . import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


_ACCORR_SOURCE = r"""
extern "C" __global__ void accorr_kernel(
    const double* __restrict__ s,
    const double* __restrict__ c,
    const double* __restrict__ angle,
    double* __restrict__ out,
    const int* __restrict__ pairs_i,
    const int* __restrict__ pairs_j,
    int n_ef, int n_ch, int n_t, int n_pairs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_ef * n_pairs) return;
    int ef = gid / n_pairs, p = gid % n_pairs;
    int i = pairs_i[p], j = pairs_j[p];
    int base = ef * n_ch * n_t;

    // Pass 1: cross-products over T
    double cc=0, ss=0, cs=0, sc=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        cc += ci*cj; ss += si*sj;
        cs += ci*sj; sc += si*cj;
    }

    double re_conj = cc + ss;
    double im_conj = -(cs - sc);
    double re_prod = cc - ss;
    double im_prod = cs + sc;

    double r_minus = sqrt(re_conj*re_conj + im_conj*im_conj);
    double r_plus  = sqrt(re_prod*re_prod + im_prod*im_prod);
    double num = r_minus - r_plus;

    double mean_diff = atan2(im_conj, re_conj);
    double mean_sum  = atan2(im_prod, re_prod);
    double n_adj = -0.5 * (mean_diff - mean_sum);
    double m_adj = mean_diff + n_adj;

    // Pass 2: sin^2 adjusted phases over T
    double sum_x2=0, sum_y2=0;
    for (int t = 0; t < n_t; t++) {
        double ai = angle[base+i*n_t+t];
        double aj = angle[base+j*n_t+t];
        double sx = sin(ai - m_adj);
        double sy = sin(aj - n_adj);
        sum_x2 += sx*sx; sum_y2 += sy*sy;
    }

    double den = 2.0 * sqrt(sum_x2 * sum_y2);
    double v = (den > 0.0) ? (num / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_accorr_kernel = None
def _get_accorr():
    global _accorr_kernel
    if _accorr_kernel is None:
        _accorr_kernel = cp.RawKernel(_ACCORR_SOURCE, "accorr_kernel")
    return _accorr_kernel


def accorr_cuda(complex_signal):
    """
    ACCorr via CUDA. Two-pass: cross-products + sin^2 denominator. Float64.

    Custom dispatch (not run_pairwise_kernel) because ACCorr needs an
    extra angle buffer for the sin^2 denominator in pass 2.
    """
    kernel = _get_accorr()

    E, F, C, T = complex_signal.shape
    n_ef = E * F

    z = complex_signal / np.abs(complex_signal)
    c_flat = cp.asarray(
        np.ascontiguousarray(np.real(z).reshape(n_ef, C, T)), dtype=cp.float64)
    s_flat = cp.asarray(
        np.ascontiguousarray(np.imag(z).reshape(n_ef, C, T)), dtype=cp.float64)
    angle_flat = cp.asarray(
        np.ascontiguousarray(np.angle(complex_signal).reshape(n_ef, C, T)),
        dtype=cp.float64)

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
        (s_flat, c_flat, angle_flat, out, pairs_i, pairs_j,
         n_ef, C, T, n_pairs)
    )

    result = cp.asnumpy(out).reshape(E, F, C, C)

    # Explicit cleanup: force immediate GPU memory release (not relying on GC)
    cp.get_default_memory_pool().free_all_blocks()

    return result
