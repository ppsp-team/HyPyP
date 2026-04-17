"""
CUDA kernels for amplitude-based sync metrics: Coh, ImCoh, EnvCorr, PowCorr.
All float64 for exact precision on NVIDIA GPUs.
"""

import numpy as np

from . import CUPY_AVAILABLE
from ._cuda_dispatch import run_pairwise_kernel

if CUPY_AVAILABLE:
    import cupy as cp


# =========================================================================
# Coh
# =========================================================================

_COH_SOURCE = r"""
extern "C" __global__ void coh_kernel(
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
    double re=0, im=0, pi=0, pj=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        re += ci*cj + si*sj;
        im += si*cj - ci*sj;
        pi += ci*ci + si*si;
        pj += cj*cj + sj*sj;
    }
    double cross = sqrt(re*re + im*im);
    double den = sqrt(pi * pj);
    double v = (den > 0.0) ? (cross / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_coh_kernel = None
def _get_coh():
    global _coh_kernel
    if _coh_kernel is None:
        _coh_kernel = cp.RawKernel(_COH_SOURCE, "coh_kernel")
    return _coh_kernel

def coh_cuda(complex_signal):
    """Coh via CUDA. Float64."""
    return run_pairwise_kernel(complex_signal, _get_coh)


# =========================================================================
# ImCoh
# =========================================================================

_IMCOH_SOURCE = r"""
extern "C" __global__ void imcoh_kernel(
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
    double im=0, pi=0, pj=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        im += si*cj - ci*sj;
        pi += ci*ci + si*si;
        pj += cj*cj + sj*sj;
    }
    double den = sqrt(pi * pj);
    double v = (den > 0.0) ? (fabs(im) / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_imcoh_kernel = None
def _get_imcoh():
    global _imcoh_kernel
    if _imcoh_kernel is None:
        _imcoh_kernel = cp.RawKernel(_IMCOH_SOURCE, "imcoh_kernel")
    return _imcoh_kernel

def imcoh_cuda(complex_signal):
    """ImCoh via CUDA. Float64."""
    return run_pairwise_kernel(complex_signal, _get_imcoh)


# =========================================================================
# EnvCorr
# =========================================================================

_ENVCORR_SOURCE = r"""
extern "C" __global__ void envcorr_kernel(
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
    // Pass 1: mean envelope
    double si_sum=0, sj_sum=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        si_sum += sqrt(ci*ci + si*si);
        sj_sum += sqrt(cj*cj + sj*sj);
    }
    double mu_i = si_sum / n_t, mu_j = sj_sum / n_t;
    // Pass 2: Pearson
    double num=0, di2=0, dj2=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        double di = sqrt(ci*ci + si*si) - mu_i;
        double dj = sqrt(cj*cj + sj*sj) - mu_j;
        num += di*dj; di2 += di*di; dj2 += dj*dj;
    }
    double den = sqrt(di2 * dj2);
    double v = (den > 0.0) ? (num / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_envcorr_kernel = None
def _get_envcorr():
    global _envcorr_kernel
    if _envcorr_kernel is None:
        _envcorr_kernel = cp.RawKernel(_ENVCORR_SOURCE, "envcorr_kernel")
    return _envcorr_kernel

def envcorr_cuda(complex_signal):
    """EnvCorr via CUDA. Pearson on envelopes. Float64."""
    return run_pairwise_kernel(complex_signal, _get_envcorr)


# =========================================================================
# PowCorr
# =========================================================================

_POWCORR_SOURCE = r"""
extern "C" __global__ void powcorr_kernel(
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
    // Pass 1: mean power
    double si_sum=0, sj_sum=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        si_sum += ci*ci + si*si;
        sj_sum += cj*cj + sj*sj;
    }
    double mu_i = si_sum / n_t, mu_j = sj_sum / n_t;
    // Pass 2: Pearson
    double num=0, di2=0, dj2=0;
    for (int t = 0; t < n_t; t++) {
        double ci=c[base+i*n_t+t], si=s[base+i*n_t+t];
        double cj=c[base+j*n_t+t], sj=s[base+j*n_t+t];
        double di = (ci*ci + si*si) - mu_i;
        double dj = (cj*cj + sj*sj) - mu_j;
        num += di*dj; di2 += di*di; dj2 += dj*dj;
    }
    double den = sqrt(di2 * dj2);
    double v = (den > 0.0) ? (num / den) : 0.0;
    int ob = ef*n_ch*n_ch;
    out[ob+i*n_ch+j] = v; out[ob+j*n_ch+i] = v;
}
"""

_powcorr_kernel = None
def _get_powcorr():
    global _powcorr_kernel
    if _powcorr_kernel is None:
        _powcorr_kernel = cp.RawKernel(_POWCORR_SOURCE, "powcorr_kernel")
    return _powcorr_kernel

def powcorr_cuda(complex_signal):
    """PowCorr via CUDA. Pearson on power. Float64."""
    return run_pairwise_kernel(complex_signal, _get_powcorr)
