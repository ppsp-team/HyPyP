import warnings, numpy as np, mne
from scipy.signal import hilbert
from hypyp import analyses
warnings.simplefilter("ignore")
rng = np.random.default_rng(10); fs = 256; t = np.arange(fs*4)/fs
x = mne.filter.filter_data(rng.standard_normal((3, t.size)), fs, 8, 12, verbose=False)
ax = hilbert(x)
for shift_deg in [0, 30, 90, 180]:
    ay = ax * np.exp(-1j*np.deg2rad(shift_deg))   # exact constant phase offset, same envelope
    cs = np.stack([ax, ay])[:, None, :, None, :]   # (2, 1 ep, 3 ch, 1 f, t)
    out = {m: analyses.compute_sync(cs, m)[0, 0, 3] for m in ["plv", "ccorr", "accorr"]}
    print(f"exact shift {shift_deg:3d} deg: " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
