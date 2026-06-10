"""
Shared Metal dispatch logic for all pairwise sync metric kernels.

All kernels share the same buffer layout:
- buffer(0): s (imaginary parts), float32
- buffer(1): c (real parts), float32
- buffer(2): output, float32
- buffer(3): pair indices i, uint32
- buffer(4): pair indices j, uint32
- buffer(5-8): constants (n_ef, n_ch, n_t, n_pairs)

ACCorr uses an extended layout with buffer(2) = angle and buffer(3) = output,
so it has its own dispatch function.

Memory
------
Three measures keep GPU memory flat across a long surrogate loop:

1. A single command queue per device, cached by ``_get_command_queue``.
   Command queues are device-limited; creating one per dispatch exhausts the
   ceiling and ``newCommandQueue()`` then returns ``None`` (the historical
   crash).
2. The dispatch runs inside an ``objc.autorelease_pool()`` so the autoreleased
   command buffer and encoder are reclaimed on each call.
3. Every owned Metal buffer (data buffers and the four ``uint32`` constant
   buffers) is released in a ``finally`` block.
"""

import contextlib
import struct
from functools import lru_cache

import numpy as np

from . import METAL_AVAILABLE

if METAL_AVAILABLE:
    import Metal

    try:
        import objc

        # Drains the autoreleased Metal temporaries (command buffer and
        # encoder) at the end of each call. Without a pool draining in a
        # tight Python loop they accumulate on the top-level autorelease
        # pool until the device refuses a new command queue
        # (``newCommandQueue()`` returns None) — the long-run leak.
        _autorelease_pool = objc.autorelease_pool
    except ImportError:  # pragma: no cover - objc ships with pyobjc Metal
        _autorelease_pool = contextlib.nullcontext
else:  # pragma: no cover - only exercised without PyObjC Metal
    _autorelease_pool = contextlib.nullcontext


def make_const_buffer(device, value):
    """Create a Metal buffer containing a single uint32 constant."""
    return device.newBufferWithBytes_length_options_(
        struct.pack('I', value), 4, Metal.MTLResourceStorageModeShared)


@lru_cache(maxsize=8)
def _get_command_queue(device):
    """
    Return a command queue for ``device``, cached one-per-device.

    Command queues are a heavyweight, device-limited resource (the device
    refuses a new one past a small ceiling). Creating one per dispatch — as
    the original code did — exhausts that ceiling in a long surrogate loop
    and ``newCommandQueue()`` then returns ``None``. A single queue is the
    idiomatic Metal pattern: many command buffers are enqueued onto it.
    """
    return device.newCommandQueue()


def run_pairwise_kernel(complex_signal, compile_fn):
    """
    Shared dispatch for pairwise Metal kernels with standard buffer layout.

    Extracts real/imag as float32, builds upper-triangle pair indices,
    dispatches the kernel, and reads back the result.

    Parameters
    ----------
    complex_signal : np.ndarray, shape (E, F, C, T)
    compile_fn : callable -> (device, pipeline)

    Returns
    -------
    np.ndarray, shape (E, F, C, C), float32
    """
    device, pipeline = compile_fn()

    E, F, C, T = complex_signal.shape
    n_ef = E * F

    c_flat = np.ascontiguousarray(np.real(complex_signal).reshape(n_ef, C, T),
                                  dtype=np.float32)
    s_flat = np.ascontiguousarray(np.imag(complex_signal).reshape(n_ef, C, T),
                                  dtype=np.float32)

    # Upper-triangle pair indices
    idx_i, idx_j = [], []
    for i in range(C):
        for j in range(i, C):
            idx_i.append(i)
            idx_j.append(j)
    idx_i = np.array(idx_i, dtype=np.uint32)
    idx_j = np.array(idx_j, dtype=np.uint32)
    n_pairs = len(idx_i)

    # Metal buffers
    buf_s = device.newBufferWithBytes_length_options_(
        s_flat.tobytes(), s_flat.nbytes, Metal.MTLResourceStorageModeShared)
    buf_c = device.newBufferWithBytes_length_options_(
        c_flat.tobytes(), c_flat.nbytes, Metal.MTLResourceStorageModeShared)
    out_nbytes = n_ef * C * C * 4
    buf_out = device.newBufferWithLength_options_(
        out_nbytes, Metal.MTLResourceStorageModeShared)
    buf_pi = device.newBufferWithBytes_length_options_(
        idx_i.tobytes(), idx_i.nbytes, Metal.MTLResourceStorageModeShared)
    buf_pj = device.newBufferWithBytes_length_options_(
        idx_j.tobytes(), idx_j.nbytes, Metal.MTLResourceStorageModeShared)

    # Constant buffers held in named locals so they can be released in the
    # finally block (passing them inline to setBuffer would leak them).
    buf_n_ef = make_const_buffer(device, n_ef)
    buf_n_ch = make_const_buffer(device, C)
    buf_n_t = make_const_buffer(device, T)
    buf_n_pairs = make_const_buffer(device, n_pairs)

    # Dispatch — wrapped in an autorelease pool so the autoreleased command
    # buffer / encoder are reclaimed each call (see the module docstring).
    try:
        with _autorelease_pool():
            queue = _get_command_queue(device)
            cmd_buffer = queue.commandBuffer()
            encoder = cmd_buffer.computeCommandEncoder()

            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(buf_s, 0, 0)
            encoder.setBuffer_offset_atIndex_(buf_c, 0, 1)
            encoder.setBuffer_offset_atIndex_(buf_out, 0, 2)
            encoder.setBuffer_offset_atIndex_(buf_pi, 0, 3)
            encoder.setBuffer_offset_atIndex_(buf_pj, 0, 4)
            encoder.setBuffer_offset_atIndex_(buf_n_ef, 0, 5)
            encoder.setBuffer_offset_atIndex_(buf_n_ch, 0, 6)
            encoder.setBuffer_offset_atIndex_(buf_n_t, 0, 7)
            encoder.setBuffer_offset_atIndex_(buf_n_pairs, 0, 8)

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
        # Release every owned Metal buffer to prevent GPU memory growth —
        # the data buffers and the four const buffers alike.
        buf_s.release()
        buf_c.release()
        buf_out.release()
        buf_pi.release()
        buf_pj.release()
        buf_n_ef.release()
        buf_n_ch.release()
        buf_n_t.release()
        buf_n_pairs.release()
