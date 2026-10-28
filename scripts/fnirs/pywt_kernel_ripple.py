"""Quantify pywt's staircase kernel and its effect on the final WTC (HyPyP vs independent reference, same smoothing)."""

import logging
import warnings

import numpy as np
import pywt
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve, butter, sosfiltfilt

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet

logging.disable()
warnings.filterwarnings("ignore")


def brute_cwt(y, scales_samples, omega0):
    n = len(y)
    W = np.empty((len(scales_samples), n), dtype=complex)
    for i, s in enumerate(scales_samples):
        half = int(np.ceil(8 * s))
        eta = np.arange(-half, half + 1) / s
        psi = np.pi**-0.25 * np.exp(1j * omega0 * eta) * np.exp(-0.5 * eta**2)
        W[i] = fftconvolve(y, np.conj(psi[::-1]), mode="same") / np.sqrt(s)
    return W


def smooth_time(W, s_samp, nfft):
    n = W.shape[1]
    k = 2 * np.pi * np.fft.fftfreq(nfft)
    F = np.exp(-0.5 * (s_samp[:, None] ** 2) * k[None, :] ** 2)
    return np.fft.ifft(F * np.fft.fft(W, n=nfft, axis=1), axis=1)[:, :n]


def wtc_ref(y1, y2, s_samp, kernel, nfft, omega0):
    y1 = (y1 - y1.mean()) / y1.std()
    y2 = (y2 - y2.mean()) / y2.std()
    W1 = brute_cwt(y1, s_samp, omega0)
    W2 = brute_cwt(y2, s_samp, omega0)
    s = s_samp[:, None]
    S1 = convolve1d(smooth_time(np.abs(W1) ** 2 / s, s_samp, nfft).real, kernel, axis=0, mode="nearest")
    S2 = convolve1d(smooth_time(np.abs(W2) ** 2 / s, s_samp, nfft).real, kernel, axis=0, mode="nearest")
    S12 = np.abs(convolve1d(smooth_time(W1 * W2.conj() / s, s_samp, nfft), kernel, axis=0, mode="nearest"))
    return S12**2 / (S1 * S2)


w = ComplexMorletWavelet(disable_caching=True)
omega0 = 2 * np.pi * w.center_frequency
kernel = BaseWavelet._get_smoothing_window(w.wtc_smoothing_win_size, w.dj)

print("=== 1. pywt impulse response (kernel) ripple vs smooth Morlet, precision=12 ===")
for s in [20, 100, 200, 500, 1000]:
    n = int(20 * s) | 1
    d = np.zeros(n)
    d[n // 2] = 1.0
    Wp, _ = pywt.cwt(d, [s], w._wavelet, method="fft")
    kp = Wp[0]
    eta = (np.arange(n) - n // 2) / s
    ks = np.exp(-0.5 * eta**2)  # envelope
    env_err = np.abs(np.abs(kp) / np.abs(kp).max() - ks).max()
    # ripple: high-frequency content of |kernel| relative to its peak
    hf = np.abs(np.diff(np.abs(kp) / np.abs(kp).max())).max()
    step = 16 / (2**12 - 1)
    print(f"scale {s:5d} samples (scale*step={s*step:.2f}): max |envelope err| {env_err:.3f}, max sample-to-sample jump of |kernel| {hf:.3f}")


def compare(fs, N, label, make_pair, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1 / fs
    x = np.arange(N) * dt
    y1, y2 = make_pair(rng, x)
    res = w.wtc(PairSignals(x, y1, y2))
    nfft = int(2 ** np.ceil(np.log2(N)))
    R = wtc_ref(y1, y2, res.scales, kernel, nfft, omega0)
    inside = ~res.wtc_masked.mask
    d = np.abs(R - res.W)
    print(
        f"{label:48s} fs={fs:4.1f} N={N:5d}: mean|diff| {d[inside].mean():.4f} max|diff| {d[inside].max():.3f} | mean WTC in COI: HyPyP {res.W[inside].mean():.4f} ref {R[inside].mean():.4f}"
    )


def white(rng, x):
    return rng.standard_normal(len(x)), rng.standard_normal(len(x))


def common_sine(rng, x):
    s = np.sin(2 * np.pi * x / 10)
    return s + rng.standard_normal(len(x)), s + rng.standard_normal(len(x))


def bandlimited(fs):
    sos = butter(4, [0.02, 0.7], btype="band", fs=fs, output="sos")

    def f(rng, x):
        s = np.sin(2 * np.pi * x / 10)
        return sosfiltfilt(sos, s + rng.standard_normal(len(x))), sosfiltfilt(sos, s + rng.standard_normal(len(x)))

    return f


print("\n=== 2. end-to-end WTC: HyPyP (pywt CWT) vs independent CWT, identical smoothing ===")
compare(10, 4096, "white noise", white)
compare(10, 4096, "common 10 s sine + white noise", common_sine)
compare(10, 4096, "band-limited 0.02-0.7 Hz (fNIRS-like)", bandlimited(10))
compare(50, 16384, "white noise", white)
compare(50, 16384, "common 10 s sine + white noise", common_sine)
compare(50, 16384, "band-limited 0.02-0.7 Hz (fNIRS-like)", bandlimited(50))

print("\n=== 3. bundled fNIRS sample (7.8125 Hz), S1_D1: raw intensity (AsIs) and HbO (RawToHaemo) ===")
from hypyp import datasets
from hypyp.fnirs.recording import Recording
from hypyp.fnirs.dyad import Dyad
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_raw_to_haemo import MnePreprocessorRawToHaemo

f1, f2 = datasets.fnirs_dcare(subject=1), datasets.fnirs_dcare(subject=2)
for label, pre, ch in [("raw 760 nm intensity", None, "S1_D1 760"), ("HbO (0.02-0.7 Hz filtered)", MnePreprocessorRawToHaemo(), "S1_D1 hbo")]:
    s1 = Recording(subject_label="a").load_file(f1, pre)
    s2 = Recording(subject_label="b").load_file(f2, pre)
    pair = Dyad(s1, s2).get_pairs(s1, s2, ch_match=ch)[0]
    res = w.wtc(pair)
    N = len(pair.y1)
    R = wtc_ref(pair.y1, pair.y2, res.scales, kernel, int(2 ** np.ceil(np.log2(N))), omega0)
    inside = ~res.wtc_masked.mask
    d = np.abs(R - res.W)
    print(f"{label:30s} N={N}: mean|diff| {d[inside].mean():.4f} max|diff| {d[inside].max():.3f} | mean WTC in COI: HyPyP {res.W[inside].mean():.4f} ref {R[inside].mean():.4f}")

print("\n=== 4. does cwt_params=dict(precision=16) fix it? (fs=50 white noise) ===")
w16 = ComplexMorletWavelet(disable_caching=True, cwt_params=dict(precision=16))
rng = np.random.default_rng(0)
dt = 1 / 50
N = 16384
x = np.arange(N) * dt
y1, y2 = rng.standard_normal(N), rng.standard_normal(N)
res = w16.wtc(PairSignals(x, y1, y2))
R = wtc_ref(y1, y2, res.scales, kernel, N, omega0)
inside = ~res.wtc_masked.mask
d = np.abs(R - res.W)
print(f"precision=16: mean|diff| {d[inside].mean():.4f} max|diff| {d[inside].max():.3f}")
