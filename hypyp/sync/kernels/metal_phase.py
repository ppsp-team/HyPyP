"""
Metal kernels for sign-based sync metrics: PLI, wPLI.

These metrics work on the imaginary part of the cross-spectrum and
cannot be efficiently expressed as batched einsum/BLAS operations,
making custom kernels faster than torch on Apple Silicon.
"""

from functools import lru_cache

import numpy as np

from . import METAL_AVAILABLE
from ._metal_dispatch import run_pairwise_kernel

if METAL_AVAILABLE:
    import Metal


# =========================================================================
# PLI
# =========================================================================

_PLI_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void pli_kernel(
    device const float* s       [[buffer(0)]],
    device const float* c       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const uint* pairs_i  [[buffer(3)]],
    device const uint* pairs_j  [[buffer(4)]],
    constant uint& n_ef         [[buffer(5)]],
    constant uint& n_ch         [[buffer(6)]],
    constant uint& n_t          [[buffer(7)]],
    constant uint& n_pairs      [[buffer(8)]],
    uint gid                    [[thread_position_in_grid]])
{
    uint total = n_ef * n_pairs;
    if (gid >= total) return;

    uint ef_idx = gid / n_pairs;
    uint pair_idx = gid % n_pairs;
    uint i = pairs_i[pair_idx];
    uint j = pairs_j[pair_idx];

    if (i == j) {
        uint out_base = ef_idx * n_ch * n_ch;
        out[out_base + i * n_ch + j] = 0.0;
        return;
    }

    uint base = ef_idx * n_ch * n_t;
    float sign_sum = 0.0;
    for (uint t = 0; t < n_t; t++) {
        float im = fma(s[base + i * n_t + t], c[base + j * n_t + t],
                       -(c[base + i * n_t + t] * s[base + j * n_t + t]));
        if (im > 0.0) sign_sum += 1.0;
        else if (im < 0.0) sign_sum -= 1.0;
    }

    float pli = abs(sign_sum) / float(n_t);
    uint out_base = ef_idx * n_ch * n_ch;
    out[out_base + i * n_ch + j] = pli;
    out[out_base + j * n_ch + i] = pli;
}
"""


@lru_cache(maxsize=1)
def _compile_pli():
    device = Metal.MTLCreateSystemDefaultDevice()
    options = Metal.MTLCompileOptions.new()
    library, error = device.newLibraryWithSource_options_error_(_PLI_SHADER, options, None)
    if error:
        raise RuntimeError(f"Metal PLI shader failed: {error}")
    fn = library.newFunctionWithName_("pli_kernel")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    if error:
        raise RuntimeError(f"Metal PLI pipeline failed: {error}")
    return device, pipeline


def pli_metal(complex_signal):
    """PLI via Metal. sign(Im(cross-spectrum)) per timepoint."""
    return run_pairwise_kernel(complex_signal, _compile_pli)


# =========================================================================
# wPLI
# =========================================================================

_WPLI_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void wpli_kernel(
    device const float* s       [[buffer(0)]],
    device const float* c       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const uint* pairs_i  [[buffer(3)]],
    device const uint* pairs_j  [[buffer(4)]],
    constant uint& n_ef         [[buffer(5)]],
    constant uint& n_ch         [[buffer(6)]],
    constant uint& n_t          [[buffer(7)]],
    constant uint& n_pairs      [[buffer(8)]],
    uint gid                    [[thread_position_in_grid]])
{
    uint total = n_ef * n_pairs;
    if (gid >= total) return;

    uint ef_idx = gid / n_pairs;
    uint pair_idx = gid % n_pairs;
    uint i = pairs_i[pair_idx];
    uint j = pairs_j[pair_idx];

    if (i == j) {
        uint out_base = ef_idx * n_ch * n_ch;
        out[out_base + i * n_ch + j] = 0.0;
        return;
    }

    uint base = ef_idx * n_ch * n_t;
    float im_sum = 0.0, abs_sum = 0.0;
    for (uint t = 0; t < n_t; t++) {
        float im = fma(s[base + i * n_t + t], c[base + j * n_t + t],
                       -(c[base + i * n_t + t] * s[base + j * n_t + t]));
        im_sum += im;
        abs_sum += fabs(im);
    }

    float wpli = (abs_sum > 0.0) ? (fabs(im_sum) / abs_sum) : 0.0;
    uint out_base = ef_idx * n_ch * n_ch;
    out[out_base + i * n_ch + j] = wpli;
    out[out_base + j * n_ch + i] = wpli;
}
"""


@lru_cache(maxsize=1)
def _compile_wpli():
    device = Metal.MTLCreateSystemDefaultDevice()
    options = Metal.MTLCompileOptions.new()
    library, error = device.newLibraryWithSource_options_error_(_WPLI_SHADER, options, None)
    if error:
        raise RuntimeError(f"Metal wPLI shader failed: {error}")
    fn = library.newFunctionWithName_("wpli_kernel")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    if error:
        raise RuntimeError(f"Metal wPLI pipeline failed: {error}")
    return device, pipeline


def wpli_metal(complex_signal):
    """wPLI via Metal. |sum(Im)| / sum(|Im|) per timepoint."""
    return run_pairwise_kernel(complex_signal, _compile_wpli)
