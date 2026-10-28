"""compute_nmPLV: axis-slicing bug and squeeze."""
import warnings
import numpy as np
from hypyp import analyses

warnings.simplefilter("ignore")
fs = 256
t = np.arange(fs * 4) / fs
n_ep, n_ch = 3, 2
rng = np.random.default_rng(2)

# p1 oscillates at 10 Hz, p2 at 20 Hz with locked phase (2*phi1 = phi2 + const): true 1:2 PLV = 1
data = np.zeros((2, n_ep, n_ch, t.size))
for e in range(n_ep):
    ph = rng.uniform(0, 2 * np.pi)
    for c in range(n_ch):
        data[0, e, c] = np.cos(2 * np.pi * 10 * t + ph) + 0.05 * rng.standard_normal(t.size)
        data[1, e, c] = np.cos(2 * np.pi * 20 * t + 2 * ph + 0.3) + 0.05 * rng.standard_normal(t.size)

res = analyses.compute_nmPLV(data, fs, [8, 12], [16, 24])
print("result shape:", res.shape, "(n_freq = 24-8 = 16 bins, same-bin comparison)")
print("inter-brain block, max over freq:", np.round(res[:, :n_ch, n_ch:].max(axis=0), 3))
print("intra p1 block (should be ~1 at 10 Hz bin):", np.round(res[2, :n_ch, :n_ch], 3))

# Show the slicing bug explicitly on the internal array
cs = np.mean(analyses.compute_single_freq(data, fs, [8, 24]), 3).squeeze()
n_epoch, n_c, n_freq, n_samp = cs.shape[1], cs.shape[2], cs.shape[3], cs.shape[4]
internal = cs.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_c, n_samp)
print("internal layout (epoch, freq, 2*n_ch, time):", internal.shape)
print("phase[:, :, :, :n_ch] slices axis 3 = TIME, selecting", n_c, "of", n_samp, "samples")

# Correct 1:2 PLV computed by hand: phase of p1 at 10 Hz bin ^2 vs p2 at 20 Hz bin ^1
tf = np.mean(analyses.compute_single_freq(data, fs, [8, 24]), 3)  # (2, ep, ch, freq, t)
freqs = np.arange(8, 24)
i10, i20 = list(freqs).index(10), list(freqs).index(20)
p1 = tf[0, :, :, i10, :]
p2 = tf[1, :, :, i20, :]
z1 = (p1 / np.abs(p1)) ** 2
z2 = (p2 / np.abs(p2)) ** 1
true_nmplv = np.abs(np.mean(z1[:, 0] * np.conj(z2[:, 0]), axis=-1)).mean()
print(f"correct 1:2 PLV (p1@10Hz^2 vs p2@20Hz) = {true_nmplv:.3f}")
print(f"hypyp nmPLV inter-brain p1ch0-p2ch0 at 10 Hz bin = {res[i10, 0, n_ch]:.3f}, at 20 Hz bin = {res[i20, 0, n_ch]:.3f}")

# Now what the code's operation actually does: raise first n_ch time points to n, rest to m
print("\n--- what the code computes with the time-axis slice (n=2, m=1):")
phase = internal / np.abs(internal)
phase_bug = phase.copy()
phase_bug[:, :, :, :n_c] = phase_bug[:, :, :, :n_c] ** 2
phase_bug[:, :, :, n_c:] = phase_bug[:, :, :, n_c:] ** 1
phase_ok = phase.copy()
phase_ok[:, :, :n_c, :] = phase_ok[:, :, :n_c, :] ** 2
phase_ok[:, :, n_c:, :] = phase_ok[:, :, n_c:, :] ** 1
con_bug = np.abs(np.einsum("efit,efjt->feij", phase_bug, phase_bug.conj())) / n_samp
con_ok = np.abs(np.einsum("efit,efjt->feij", phase_ok, phase_ok.conj())) / n_samp
print("time-slice version  p1ch0-p2ch0 @10Hz:", np.round(con_bug.mean(1)[i10, 0, n_c], 3), "(== hypyp result:", np.round(res[i10, 0, n_c], 3), ")")
print("channel-slice version p1ch0-p2ch0 @10Hz:", np.round(con_ok.mean(1)[i10, 0, n_c], 3), "(same-bin, still not cross-frequency)")

# squeeze crash: single epoch
try:
    analyses.compute_nmPLV(data[:, :1], fs, [8, 12], [16, 24])
    print("single epoch: OK")
except Exception as exc:
    print("single epoch:", type(exc).__name__, str(exc)[:120])
try:
    analyses.compute_nmPLV(data[:, :, :1], fs, [8, 12], [16, 24])
    print("single channel: OK")
except Exception as exc:
    print("single channel:", type(exc).__name__, str(exc)[:120])
