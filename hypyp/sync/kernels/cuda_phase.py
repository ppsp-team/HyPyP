"""
CUDA kernels for phase-based sync metrics: PLI, wPLI, PLV, CCorr.
All float64 for exact precision on NVIDIA GPUs.
"""

import numpy as np

from . import CUPY_AVAILABLE
from ._cuda_dispatch import run_pairwise_kernel

if CUPY_AVAILABLE:
    import cupy as cp


# =========================================================================
# PLI
# =========================================================================

_PLI_SOURCE = r"""
extern "C" __global__ void pli_kernel(
    const double* __restrict__ s, const double* __restrict__ c,
    double* __restrict__ out,
    const int* __restrict__ pairs_i, const int* __restrict__ pairs_j,
    int n_ef, int n_ch, int n_t, int n_pairs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_ef * n_pairs) return;
    int ef = gid / n_pairs, p = gid % n_pairs;
    int i = pairs_i[p], j = pairs_j[p];
    if (i == j) { out[ef*n_ch*n_ch + i*n_ch+j] = 0.0; return; }
    int base = ef * n_ch * n_t;
    double sign_sum = 0.0;
    for (int t = 0; t < n_t; t++) {
        double im = s[base+i*n_t+t]*c[base+j*n_t+t] - c[base+i*n_t+t]*s[base+j*n_t+t];
        if (im > 0.0) sign_sum += 1.0;
        else if (im < 0.0) sign_sum -= 1.0;
    }
    double v = fabs(sign_sum) / (double)n_t;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_pli_kernel = None
def _get_pli():
    global _pli_kernel
    if _pli_kernel is None:
        _pli_kernel = cp.RawKernel(_PLI_SOURCE, "pli_kernel")
    return _pli_kernel

def pli_cuda(complex_signal):
    """PLI via CUDA. Float64."""
    return run_pairwise_kernel(complex_signal, _get_pli)


# =========================================================================
# wPLI
# =========================================================================

_WPLI_SOURCE = r"""
extern "C" __global__ void wpli_kernel(
    const double* __restrict__ s, const double* __restrict__ c,
    double* __restrict__ out,
    const int* __restrict__ pairs_i, const int* __restrict__ pairs_j,
    int n_ef, int n_ch, int n_t, int n_pairs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_ef * n_pairs) return;
    int ef = gid / n_pairs, p = gid % n_pairs;
    int i = pairs_i[p], j = pairs_j[p];
    if (i == j) { out[ef*n_ch*n_ch + i*n_ch+j] = 0.0; return; }
    int base = ef * n_ch * n_t;
    double im_sum = 0.0, abs_sum = 0.0;
    for (int t = 0; t < n_t; t++) {
        double im = s[base+i*n_t+t]*c[base+j*n_t+t] - c[base+i*n_t+t]*s[base+j*n_t+t];
        im_sum += im; abs_sum += fabs(im);
    }
    double v = (abs_sum > 0.0) ? (fabs(im_sum) / abs_sum) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_wpli_kernel = None
def _get_wpli():
    global _wpli_kernel
    if _wpli_kernel is None:
        _wpli_kernel = cp.RawKernel(_WPLI_SOURCE, "wpli_kernel")
    return _wpli_kernel

def wpli_cuda(complex_signal):
    """wPLI via CUDA. Float64."""
    return run_pairwise_kernel(complex_signal, _get_wpli)


# =========================================================================
# PLV
# =========================================================================

_PLV_SOURCE = r"""
extern "C" __global__ void plv_kernel(
    const double* __restrict__ s, const double* __restrict__ c,
    double* __restrict__ out,
    const int* __restrict__ pairs_i, const int* __restrict__ pairs_j,
    int n_ef, int n_ch, int n_t, int n_pairs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_ef * n_pairs) return;
    int ef = gid / n_pairs, p = gid % n_pairs;
    int i = pairs_i[p], j = pairs_j[p];
    int base = ef * n_ch * n_t;
    double re = 0.0, im = 0.0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        re += ci*cj + si*sj;
        im += si*cj - ci*sj;
    }
    double v = sqrt(re*re + im*im) / (double)n_t;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_plv_kernel = None
def _get_plv():
    global _plv_kernel
    if _plv_kernel is None:
        _plv_kernel = cp.RawKernel(_PLV_SOURCE, "plv_kernel")
    return _plv_kernel

def plv_cuda(complex_signal):
    """PLV via CUDA. Phase-normalizes then cross-spectrum. Float64."""
    z = complex_signal / np.abs(complex_signal)
    return run_pairwise_kernel(z, _get_plv)


# =========================================================================
# CCorr
# =========================================================================

_CCORR_SOURCE = r"""
extern "C" __global__ void ccorr_kernel(
    const double* __restrict__ s, const double* __restrict__ c,
    double* __restrict__ out,
    const int* __restrict__ pairs_i, const int* __restrict__ pairs_j,
    int n_ef, int n_ch, int n_t, int n_pairs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_ef * n_pairs) return;
    int ef = gid / n_pairs, p = gid % n_pairs;
    int i = pairs_i[p], j = pairs_j[p];
    int base = ef * n_ch * n_t;
    // Pass 1: C_bar, S_bar
    double cbi=0, sbi=0, cbj=0, sbj=0;
    for (int t = 0; t < n_t; t++) {
        cbi += c[base+i*n_t+t]; sbi += s[base+i*n_t+t];
        cbj += c[base+j*n_t+t]; sbj += s[base+j*n_t+t];
    }
    cbi /= n_t; sbi /= n_t; cbj /= n_t; sbj /= n_t;
    // Pass 2: Pearson
    double num=0, di2=0, dj2=0;
    for (int t = 0; t < n_t; t++) {
        double di = s[base+i*n_t+t]*cbi - c[base+i*n_t+t]*sbi;
        double dj = s[base+j*n_t+t]*cbj - c[base+j*n_t+t]*sbj;
        num += di*dj; di2 += di*di; dj2 += dj*dj;
    }
    double den = sqrt(di2 * dj2);
    double v = (den > 0.0) ? (fabs(num) / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_ccorr_kernel = None
def _get_ccorr():
    global _ccorr_kernel
    if _ccorr_kernel is None:
        _ccorr_kernel = cp.RawKernel(_CCORR_SOURCE, "ccorr_kernel")
    return _ccorr_kernel

def ccorr_cuda(complex_signal):
    """CCorr via CUDA. Phase-normalizes then angle-free Pearson. Float64."""
    z = complex_signal / np.abs(complex_signal)
    return run_pairwise_kernel(z, _get_ccorr)
