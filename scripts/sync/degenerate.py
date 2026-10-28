"""Degenerate inputs: zero channel, constant envelope, single zero sample,
uniform phase.  Shows what the numpy path returns, and what the numba /
CUDA / Metal / torch guards (read from source) would return instead."""
import warnings

import numpy as np

from hypyp.sync import get_metric
from hypyp.analyses import compute_sync

warnings.simplefilter("ignore")
AX = (0, 1, 3, 2)
rng = np.random.default_rng(1)
T = 500
MODES = ["plv", "ccorr", "accorr", "coh", "imcoh", "pli", "wpli", "envcorr", "powcorr"]


def run(mode, sig):
    m = get_metric(mode)
    if mode == "accorr":
        m.show_progress = False
    return m.compute(sig, sig.shape[-1], AX)


good = rng.standard_normal(T) + 1j * rng.standard_normal(T)
good2 = rng.standard_normal(T) + 1j * rng.standard_normal(T)

print("=== 1. channel 1 is all zeros (a zeroed bad channel) ===")
sig = np.stack([good, np.zeros(T, complex), good2])[None, None]
for mode in MODES:
    r = run(mode, sig)[0, 0]
    print(f"{mode:8s} row of zero ch: {np.array2string(r[1], precision=3)}   good pair (0,2)={r[0,2]:.4f}")

print("\n=== 2. channel 1 has a constant envelope (pure unit-amplitude oscillation) ===")
sig = np.stack([good, np.exp(1j * np.linspace(0, 40, T)), good2])[None, None]
for mode in ["envcorr", "powcorr", "coh", "plv"]:
    r = run(mode, sig)[0, 0]
    print(f"{mode:8s} row of const-env ch: {np.array2string(r[1], precision=3)}")

print("\n=== 3. channel 1 has ONE zero sample out of 500 ===")
z = good2.copy()
z[123] = 0
sig = np.stack([good, z, good2])[None, None]
for mode in MODES:
    r = run(mode, sig)[0, 0]
    print(f"{mode:8s} row of ch with 1 zero sample: {np.array2string(r[1], precision=3)}")

print("\n=== 4. CCorr: channel with a phase distribution of resultant length R == 0 exactly ===")
# alpha_k = k*pi/2 for k=0..3 repeated: cos = 1,0,-1,0 ; sin = 0,1,0,-1  -> exact zero mean
a = np.tile([0, 1, -1, 0], T // 4) + 1j * np.tile([0, 0, 0, 0], T // 4)
a = np.tile(np.array([1, 1j, -1, -1j]), T // 4)
print("R =", abs(np.mean(a)))
sig = np.stack([good, a])[None, None]
print("numpy ccorr:", run("ccorr", sig)[0, 0, 0, 1])
# the angle-free form used by numba / torch / cuda:
c, s = a.real, a.imag
Cb, Sb = c.mean(), s.mean()
d_a = s * Cb - c * Sb
zb = good / abs(good)
d_b = zb.imag * zb.real.mean() - zb.real * zb.imag.mean()
num = np.sum(d_a * d_b)
den = np.sqrt(np.sum(d_a**2) * np.sum(d_b**2))
print(f"angle-free form (numba/cuda/torch): num={num:.3e} den={den:.3e} -> "
      f"numba/cuda return 0.0, torch returns {abs(num/den) if den else float('nan')}")

print("\n=== 5. compute_sync: nanmean silently drops the NaN epoch for that pair ===")
raw = rng.standard_normal((2, 3, 2, 1, T)) + 1j * rng.standard_normal((2, 3, 2, 1, T))
raw[0, 1, 0, 0, 7] = 0  # one zero sample in epoch 1, subject 0, channel 0
res = compute_sync(raw, "plv", epochs_average=False)
print("per-epoch plv for pair (0, 2):", res[0, :, 0, 2])
print("averaged plv for pair (0, 2):", compute_sync(raw, "plv")[0, 0, 2],
      "(= mean of the 2 finite epochs, no warning)")
