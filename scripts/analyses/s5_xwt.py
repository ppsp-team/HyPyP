"""xwt: single-epoch crash, WTC sanity, smoothing window, redundant CWTs."""
import warnings
import time
import numpy as np
import mne
from hypyp import analyses

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
rng = np.random.default_rng(4)
fs = 128
n_ep, n_ch, n_t = 3, 3, 512


def make_epochs(data):
    info = mne.create_info([f"C{i}" for i in range(data.shape[1])], fs, "eeg")
    return mne.EpochsArray(data, info, verbose=False)


base = rng.standard_normal((n_ep, n_ch, n_t))
e1 = make_epochs(base)
e2 = make_epochs(base.copy())
freqs = np.arange(4, 20, 2.0)

t0 = time.time()
wtc = analyses.xwt(e1, e2, freqs, n_cycles=5, mode="wtc")
print("wtc shape", wtc.shape, f"time {time.time()-t0:.2f}s")
print("identical signals: WTC min/max in the central region:",
      np.round(np.nanmin(wtc[0, 0, :, :, 100:-100]), 4), np.round(np.nanmax(wtc[0, 0, :, :, 100:-100]), 4))

# independent signals: WTC bias
e3 = make_epochs(rng.standard_normal((n_ep, n_ch, n_t)))
wtc_ind = analyses.xwt(e1, e3, freqs, n_cycles=5, mode="wtc")
print("independent signals: mean WTC central region:", np.round(np.nanmean(wtc_ind[0, 0, :, :, 100:-100]), 3),
      " (Grinsted 2004 with time+scale smoothing gives a much lower null)")

# smoothing window: how much of the Gaussian is kept?
scales = 5 * fs / (2 * np.pi * freqs)
for s in scales[[0, -1]]:
    win_size = max(3, int(2 * s))
    print(f"scale {s:.1f} samples -> window length {win_size} = +/-{win_size/2/s:.2f} sigma (full Gaussian would need ~+/-3 sigma)")

# single epoch
try:
    out = analyses.xwt(make_epochs(base[:1]), make_epochs(base[:1]), freqs, mode="wtc")
    print("single epoch OK", out.shape)
except Exception as exc:
    print("single epoch:", type(exc).__name__, str(exc)[:150])

# invalid mode returns a string
out = analyses.xwt(e1, e2, freqs, mode="bogus")
print("invalid mode returns:", type(out).__name__)

# count CWT calls
import hypyp.analyses as A
calls = {"n": 0}
orig = mne.time_frequency.tfr.cwt
def counting(*a, **k):
    calls["n"] += 1
    return orig(*a, **k)
mne.time_frequency.tfr.cwt = counting
analyses.xwt(e1, e2, freqs, mode="xwt")
mne.time_frequency.tfr.cwt = orig
print(f"cwt calls for n_ch={n_ch}: {calls['n']} (minimum needed: {2*n_ch})")
print("output memory for 64 ch, 100 ep, 30 freqs, 1000 samples, complex128:",
      64 * 64 * 100 * 30 * 1000 * 16 / 1e9, "GB")
