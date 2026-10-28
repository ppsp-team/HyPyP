import warnings, numpy as np
from hypyp import analyses
warnings.simplefilter("ignore")
fs = 256; t = np.arange(fs*4)/fs; n_ep, n_ch = 3, 2; rng = np.random.default_rng(2)
data = np.zeros((2, n_ep, n_ch, t.size))
for e in range(n_ep):
    ph = rng.uniform(0, 2*np.pi)
    for c in range(n_ch):
        data[0, e, c] = np.cos(2*np.pi*10*t + ph) + 0.05*rng.standard_normal(t.size)
        data[1, e, c] = np.cos(2*np.pi*20*t + 2*ph + 0.3) + 0.05*rng.standard_normal(t.size)
res = analyses.compute_nmPLV(data, fs, [8, 12], [16, 24])
print("hypyp result shape", res.shape, "-> inter-brain p1ch0-p2ch0 per 'row':", np.round(res[:, 0, n_ch], 3))
# fix only the slicing + axis bugs (channel-axis powers, epoch average) and keep same-bin comparison
tf = np.mean(analyses.compute_single_freq(data, fs, [8, 24]), 3)
ph_ = tf/np.abs(tf); ph_ = ph_.at if False else ph_
z1 = ph_[0]**2; z2 = ph_[1]**1                     # (ep, ch, f, t)
same_bin = np.abs(np.einsum("ecft,edft->fecd", z1, z2.conj()))/t.size
print("channel-axis fix, same-bin, epoch-avg, p1ch0-p2ch0 at bins 8..23:", np.round(same_bin.mean(1)[:, 0, 0], 3))
i10, i20 = 2, 12
cross = np.abs(np.mean(z1[:, 0, i10]*np.conj(z2[:, 0, i20]), -1)).mean()
print(f"true cross-frequency 1:2 PLV (p1@10Hz^2 vs p2@20Hz): {cross:.3f}")
