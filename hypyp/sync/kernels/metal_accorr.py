"""
Metal kernel for Adjusted Circular Correlation (ACCorr).

ACCorr is the most complex metric — it requires a two-pass kernel:
- Pass 1: cross-products (cc, ss, cs, sc) → numerator + phase adjustments
- Pass 2: sin^2 adjusted phases → denominator

This module uses an extended buffer layout (3 input buffers: s, c, angle)
instead of the standard 2-buffer layout used by other metrics.
"""

import struct
from functools import lru_cache

import numpy as np

from . import METAL_AVAILABLE
from ._metal_dispatch import make_const_buffer

if METAL_AVAILABLE:
    import Metal


_ACCORR_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void accorr_kernel(
    device const float* s       [[buffer(0)]],
    device const float* c       [[buffer(1)]],
    device const float* angle   [[buffer(2)]],
    device float* out           [[buffer(3)]],
    device const uint* pairs_i  [[buffer(4)]],
    device const uint* pairs_j  [[buffer(5)]],
    constant uint& n_ef         [[buffer(6)]],
    constant uint& n_ch         [[buffer(7)]],
    constant uint& n_t          [[buffer(8)]],
    constant uint& n_pairs      [[buffer(9)]],
    uint gid                    [[thread_position_in_grid]])
{
    uint total = n_ef * n_pairs;
    if (gid >= total) return;

    uint ef_idx = gid / n_pairs;
    uint pair_idx = gid % n_pairs;
    uint i = pairs_i[pair_idx];
    uint j = pairs_j[pair_idx];
    uint base = ef_idx * n_ch * n_t;

    // Pass 1: cross-products over T
    float cc_sum = 0.0, ss_sum = 0.0, cs_sum = 0.0, sc_sum = 0.0;
    for (uint t = 0; t < n_t; t++) {
        float ci = c[base + i * n_t + t];
        float si = s[base + i * n_t + t];
        float cj = c[base + j * n_t + t];
        float sj = s[base + j * n_t + t];
        cc_sum += ci * cj;
        ss_sum += si * sj;
        cs_sum += ci * sj;
        sc_sum += si * cj;
    }

    float re_conj = cc_sum + ss_sum;
    float im_conj = -(cs_sum - sc_sum);
    float re_prod = cc_sum - ss_sum;
    float im_prod = cs_sum + sc_sum;

    float r_minus = sqrt(re_conj * re_conj + im_conj * im_conj);
    float r_plus  = sqrt(re_prod * re_prod + im_prod * im_prod);
    float num = r_minus - r_plus;

    float mean_diff = atan2(im_conj, re_conj);
    float mean_sum  = atan2(im_prod, re_prod);
    float n_adj = -0.5 * (mean_diff - mean_sum);
    float m_adj = mean_diff + n_adj;

    // Pass 2: sin^2 adjusted phases over T
    float sum_x2 = 0.0, sum_y2 = 0.0;
    for (uint t = 0; t < n_t; t++) {
        float ai = angle[base + i * n_t + t];
        float aj = angle[base + j * n_t + t];
        float sx = sin(ai - m_adj);
        float sy = sin(aj - n_adj);
        sum_x2 += sx * sx;
        sum_y2 += sy * sy;
    }

    float den = 2.0 * sqrt(sum_x2 * sum_y2);
    float accorr = (den > 0.0) ? (num / den) : 0.0;

    uint out_base = ef_idx * n_ch * n_ch;
    out[out_base + i * n_ch + j] = accorr;
    out[out_base + j * n_ch + i] = accorr;
}
"""


@lru_cache(maxsize=1)
def _compile_accorr():
    device = Metal.MTLCreateSystemDefaultDevice()
    options = Metal.MTLCompileOptions.new()
    library, error = device.newLibraryWithSource_options_error_(_ACCORR_SHADER, options, None)
    if error:
        raise RuntimeError(f"Metal ACCorr shader failed: {error}")
    fn = library.newFunctionWithName_("accorr_kernel")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    if error:
        raise RuntimeError(f"Metal ACCorr pipeline failed: {error}")
    return device, pipeline


def accorr_metal(complex_signal: np.ndarray) -> np.ndarray:
    """
    ACCorr via Metal. Two-pass kernel: cross-products + sin^2 denominator.

    Uses 3 input buffers (s, c, angle) instead of standard 2.
    """
    device, pipeline = _compile_accorr()

    E, F, C, T = complex_signal.shape
    n_ef = E * F

    z = complex_signal / np.abs(complex_signal)
    c_flat = np.ascontiguousarray(np.real(z).reshape(n_ef, C, T), dtype=np.float32)
    s_flat = np.ascontiguousarray(np.imag(z).reshape(n_ef, C, T), dtype=np.float32)
    angle_flat = np.ascontiguousarray(
        np.angle(complex_signal).reshape(n_ef, C, T), dtype=np.float32)

    idx_i, idx_j = [], []
    for i in range(C):
        for j in range(i, C):
            idx_i.append(i)
            idx_j.append(j)
    idx_i = np.array(idx_i, dtype=np.uint32)
    idx_j = np.array(idx_j, dtype=np.uint32)
    n_pairs = len(idx_i)

    # Metal buffers — extended layout for ACCorr
    buf_s = device.newBufferWithBytes_length_options_(
        s_flat.tobytes(), s_flat.nbytes, Metal.MTLResourceStorageModeShared)
    buf_c = device.newBufferWithBytes_length_options_(
        c_flat.tobytes(), c_flat.nbytes, Metal.MTLResourceStorageModeShared)
    buf_angle = device.newBufferWithBytes_length_options_(
        angle_flat.tobytes(), angle_flat.nbytes, Metal.MTLResourceStorageModeShared)
    out_nbytes = n_ef * C * C * 4
    buf_out = device.newBufferWithLength_options_(
        out_nbytes, Metal.MTLResourceStorageModeShared)
    buf_pi = device.newBufferWithBytes_length_options_(
        idx_i.tobytes(), idx_i.nbytes, Metal.MTLResourceStorageModeShared)
    buf_pj = device.newBufferWithBytes_length_options_(
        idx_j.tobytes(), idx_j.nbytes, Metal.MTLResourceStorageModeShared)

    # Dispatch
    try:
        queue = device.newCommandQueue()
        cmd_buffer = queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_s, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_c, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_angle, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_out, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_pi, 0, 4)
        encoder.setBuffer_offset_atIndex_(buf_pj, 0, 5)
        encoder.setBuffer_offset_atIndex_(make_const_buffer(device, n_ef), 0, 6)
        encoder.setBuffer_offset_atIndex_(make_const_buffer(device, C), 0, 7)
        encoder.setBuffer_offset_atIndex_(make_const_buffer(device, T), 0, 8)
        encoder.setBuffer_offset_atIndex_(make_const_buffer(device, n_pairs), 0, 9)

        total_threads = n_ef * n_pairs
        threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())

        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSize(total_threads, 1, 1),
            Metal.MTLSize(threads_per_group, 1, 1))
        encoder.endEncoding()

        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        out_ptr = buf_out.contents()
        membuf = out_ptr.as_buffer(out_nbytes)
        result = np.frombuffer(membuf, dtype=np.float32).copy().reshape(n_ef, C, C)

        return result.reshape(E, F, C, C)
    finally:
        # Critical: Release all Metal buffers to prevent GPU memory leak
        buf_s.release()
        buf_c.release()
        buf_angle.release()
        buf_out.release()
        buf_pi.release()
        buf_pj.release()
