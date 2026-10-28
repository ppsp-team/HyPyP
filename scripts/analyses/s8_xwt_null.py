"""xwt WTC null level: hypyp smoothing (time-only, +/-1 sigma) vs Grinsted 2004 smoothing
(full Gaussian in time, boxcar of 0.6 scale across scales)."""
import warnings
import numpy as np
import mne
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
from hypyp import analyses

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
rng = np.random.default_rng(8)
fs, n_t, n_ep = 128, 2048, 10
n_cycles = 6.0
freqs = 2 ** np.arange(np.log2(4), np.log2(32), 1 / 8)  # log-spaced like Grinsted (8 per octave)


def ep(data):
    return mne.EpochsArray(data, mne.create_info(["C0"], fs, "eeg"), verbose=False)

a = rng.standard_normal((n_ep, 1, n_t))
b = rng.standard_normal((n_ep, 1, n_t))
wtc_h = analyses.xwt(ep(a), ep(b), freqs, n_cycles=n_cycles, mode="wtc")[0, 0]  # (ep, f, t)
core = slice(400, -400)
print(f"hypyp WTC (independent white noise), mean over core = {np.nanmean(wtc_h[:, :, core]):.3f}, "
      f"95th pct = {np.nanpercentile(wtc_h[:, :, core], 95):.3f}")

# Reference smoothing
Ws = mne.time_frequency.tfr.morlet(fs, freqs, n_cycles=n_cycles, zero_mean=True)
W1 = mne.time_frequency.tfr.cwt(a[:, 0], Ws, use_fft=True, mode="same")
W2 = mne.time_frequency.tfr.cwt(b[:, 0], Ws, use_fft=True, mode="same")
scales = n_cycles * fs / (2 * np.pi * freqs)


def smooth_time(x, trunc_sigma):
    out = np.empty_like(x)
    for fi, s in enumerate(scales):
        n = int(2 * trunc_sigma * s) | 1
        w = gaussian(n, std=s); w /= w.sum()
        out[:, fi] = convolve1d(x[:, fi], w, axis=-1, mode="constant")
    return out


def smooth_scale(x, dj=1 / 8, width=0.6):
    n = max(1, int(round(width / dj)))
    w = np.ones(n) / n
    return convolve1d(x, w, axis=1, mode="nearest")


def wtc_ref(trunc_sigma, with_scale):
    sinv = (1 / scales)[None, :, None]
    S12, S11, S22 = W1 * W2.conj() * sinv, np.abs(W1) ** 2 * sinv, np.abs(W2) ** 2 * sinv
    S12, S11, S22 = smooth_time(S12, trunc_sigma), smooth_time(S11, trunc_sigma), smooth_time(S22, trunc_sigma)
    if with_scale:
        S12, S11, S22 = smooth_scale(S12), smooth_scale(S11), smooth_scale(S22)
    return np.abs(S12) ** 2 / (S11 * S22)

for ts, sc in [(1, False), (3, False), (3, True)]:
    w = wtc_ref(ts, sc)[:, :, core]
    print(f"reference  time +/-{ts} sigma, scale smoothing={sc!s:5s}: mean = {np.nanmean(w):.3f}, 95th pct = {np.nanpercentile(w, 95):.3f}")

# integer freqs argument (docstring says Union[int, np.ndarray])
try:
    analyses.xwt(ep(a), ep(b), 10, mode="wtc")
except Exception as exc:
    print("freqs=int ->", type(exc).__name__, exc)
