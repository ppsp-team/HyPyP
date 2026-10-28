"""Complex taper averaging (pair_connectivity list path) vs proper multitaper cross-spectra;
compute_sync on real input; compute_nmPLV output axes."""
import warnings
import numpy as np
import mne
from scipy.signal import hilbert
from hypyp import analyses

warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
fs = 256
t = np.arange(fs * 4) / fs
n_ep = 20

# amplitude-modulated narrowband signals, p2 = 90-degree lagged copy of p1
data = np.zeros((2, n_ep, 1, t.size))
for e in range(n_ep):
    x = mne.filter.filter_data(rng.standard_normal(t.size), fs, 8, 12, verbose=False)
    data[0, e, 0] = x
    data[1, e, 0] = np.imag(hilbert(x))

tf = analyses.compute_single_freq(data, fs, [10, 11])  # (2, ep, 1, tapers, 1, t)
tf = tf[:, :, 0, :, 0, fs:-fs]  # drop edges -> (2, ep, tapers, t)
x1, x2 = tf[0], tf[1]

# per-taper estimators, then averaged over tapers (standard multitaper)
def per_taper(fn):
    return np.mean([fn(x1[:, k], x2[:, k]) for k in range(x1.shape[1])])

def plv(a, b):
    z = (a / np.abs(a)) * np.conj(b / np.abs(b))
    return np.abs(z.mean(-1)).mean()

def pli(a, b):
    return np.abs(np.sign(np.imag(a * np.conj(b))).mean(-1)).mean()

def imcoh(a, b):
    s12 = (a * np.conj(b)).sum(-1)
    return (np.imag(s12) / np.sqrt((np.abs(a) ** 2).sum(-1) * (np.abs(b) ** 2).sum(-1))).mean()

def coh(a, b):
    s12 = (a * np.conj(b)).sum(-1)
    return (np.abs(s12) / np.sqrt((np.abs(a) ** 2).sum(-1) * (np.abs(b) ** 2).sum(-1))).mean()

m1, m2 = x1.mean(1), x2.mean(1)  # hypyp: complex mean over tapers
print(f"{'measure':8s} {'per-taper (k=0,1,2)':32s} {'taper-avg':>10s} {'hypyp complex mean':>20s}")
for name, fn in [("plv", plv), ("pli", pli), ("imcoh", imcoh), ("coh", coh)]:
    vals = [fn(x1[:, k], x2[:, k]) for k in range(3)]
    print(f"{name:8s} {str(np.round(vals, 3)):32s} {np.mean(vals):10.3f} {fn(m1, m2):20.3f}")

# Same through the public API
con = analyses.pair_connectivity(data, fs, [10, 11], "pli", epochs_average=True)
print("pair_connectivity(list) pli p1-p2:", np.round(con[0, 0, 1], 3))
con = analyses.pair_connectivity(data, fs, [10, 11], "imcoh", epochs_average=True)
print("pair_connectivity(list) imcoh p1-p2:", np.round(con[0, 0, 1], 3))
con = analyses.pair_connectivity(data, fs, {"a": [8, 12]}, "pli", epochs_average=True)
print("pair_connectivity(dict) pli p1-p2:", np.round(con[0, 0, 1], 3))

print("\n=== compute_sync with real-valued (non-analytic) input")
real = rng.standard_normal((2, 3, 2, 1, 512))
for mode in ["plv", "coh", "pli"]:
    con = analyses.compute_sync(real, mode)
    print(mode, "inter block:", np.round(con[0, :2, 2:], 3))

print("\n=== compute_nmPLV output axes")
d = rng.standard_normal((2, 5, 2, 1024))
res = analyses.compute_nmPLV(d, fs, [8, 12], [16, 24])
print("n_epochs=5, n_ch=2, freq bins 8..23 (16) -> result.shape =", res.shape,
      " docstring says (n_freq, 2*n_channels, 2*n_channels)")
