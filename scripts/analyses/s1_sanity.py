"""End-to-end sanity of compute_sync on analytic signals with known coupling."""
import warnings
import numpy as np
from scipy.signal import hilbert
from hypyp import analyses

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
fs, T = 256, 4.0
n_ep, n_ch = 10, 2
t = np.arange(int(fs * T)) / fs
MODES = ["plv", "ccorr", "accorr", "coh", "imcoh", "pli", "wpli", "envelope_corr", "pow_corr"]


def narrowband(n):
    """n independent narrowband (8-12 Hz) signals, shape (n, n_times)."""
    x = rng.standard_normal((n, t.size))
    import mne
    return mne.filter.filter_data(x, fs, 8, 12, verbose=False)


def build(case):
    """Return data shaped (2, n_ep, n_ch, n_times)."""
    data = np.zeros((2, n_ep, n_ch, t.size))
    for e in range(n_ep):
        base = narrowband(n_ch)
        if case == "identical":
            data[0, e], data[1, e] = base, base
        elif case == "shift90":
            data[0, e] = base
            data[1, e] = np.imag(hilbert(base))  # Hilbert = -90 deg shift
        elif case == "independent":
            data[0, e], data[1, e] = base, narrowband(n_ch)
        elif case == "quadrant":
            # p1 ch0 identical to p2 ch1, everything else independent
            other = narrowband(n_ch)
            data[0, e] = base
            data[1, e] = other
            data[1, e, 1] = base[0]
    return data


for case in ["identical", "shift90", "independent", "quadrant"]:
    data = build(case)
    # analytic signal by compute_freq_bands with filtering OFF (data already narrowband)
    cs = analyses.compute_freq_bands(data, fs, {"alpha": [8, 12]}, filter_signal=False)
    print(f"\n=== {case}: complex_signal shape {cs.shape}")
    for mode in MODES:
        con = analyses.compute_sync(cs, mode)
        inter = con[0, :n_ch, n_ch:]
        if case == "quadrant":
            print(f"{mode:14s} inter block=\n{np.round(inter, 3)}")
        else:
            print(f"{mode:14s} p1ch0-p2ch0={inter[0,0]: .4f}  p1ch1-p2ch1={inter[1,1]: .4f}  "
                  f"p1ch0-p2ch1={inter[0,1]: .4f}  shape={con.shape}")

# Coherence: squared or not? Build x, y = x + noise with known MSC.
print("\n=== coherence squared-or-not check")
x = narrowband(1)[0]
y = x + 0.5 * narrowband(1)[0]
ax, ay = hilbert(x), hilbert(y)
num = np.abs(np.sum(ax * np.conj(ay)))
den = np.sqrt(np.sum(np.abs(ax) ** 2) * np.sum(np.abs(ay) ** 2))
cs = np.stack([ax, ay])[:, None, None, None, :]  # (2, 1 ep, 1 ch, 1 freq, t)
coh = analyses.compute_sync(cs, "coh")[0, 0, 1]
print(f"hypyp coh={coh:.5f}   |C|={num/den:.5f}   |C|^2={(num/den)**2:.5f}")

# Per-epoch PLV bias: independent noise, same total duration, different epoch length
print("\n=== PLV bias for independent noise vs epoch length (total 80 s)")
for ep_len in [1.0, 4.0, 16.0]:
    n = int(80 / ep_len)
    tt = np.arange(int(fs * ep_len)) / fs
    import mne
    d = np.zeros((2, n, 1, tt.size))
    for e in range(n):
        d[:, e, 0] = mne.filter.filter_data(rng.standard_normal((2, tt.size)), fs, 8, 12, verbose=False)
    cs = analyses.compute_freq_bands(d, fs, {"a": [8, 12]}, filter_signal=False)
    plv = analyses.compute_sync(cs, "plv")[0, 0, 1]
    print(f"epoch {ep_len:5.1f}s  n_epochs={n:3d}  PLV={plv:.3f}")

# NaN handling: a flat channel in one epoch
print("\n=== flat channel in epoch 0 -> nanmean silently drops that epoch")
data = build("independent")
data[1, 0, 0] = 0.0
cs = analyses.compute_freq_bands(data, fs, {"a": [8, 12]}, filter_signal=False)
con_e = analyses.compute_sync(cs, "plv", epochs_average=False)
con = analyses.compute_sync(cs, "plv")
print("per-epoch value epoch0 p1ch0-p2ch0:", con_e[0, 0, 0, n_ch], " averaged:", con[0, 0, n_ch],
      " any warning?  (none printed above)")
