"""Diagnose pywt CWT vs T&C reference per scale; redo WTC comparison with a zero-padded reference."""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet


def tc_cwt(y, dt, scales_sec, omega0, pad_to=None):
    n = len(y)
    nfft = n if pad_to is None else pad_to
    k = 2 * np.pi * np.fft.fftfreq(nfft, d=dt)
    yhat = np.fft.fft(y, n=nfft)
    W = np.empty((len(scales_sec), n), dtype=complex)
    for i, s in enumerate(scales_sec):
        psi_hat = np.sqrt(2 * np.pi * s / dt) * np.pi**-0.25 * np.exp(-0.5 * (s * k - omega0) ** 2) * (k > 0)
        W[i] = np.fft.ifft(yhat * psi_hat)[:n]
    return W


def brute_cwt(y, scales_samples, omega0):
    """Direct L2-normalised convolution with the sampled Morlet, zero padded (conj convention of T&C)."""
    n = len(y)
    W = np.empty((len(scales_samples), n), dtype=complex)
    for i, s in enumerate(scales_samples):
        half = int(np.ceil(8 * s))
        eta = np.arange(-half, half + 1) / s
        psi = np.pi**-0.25 * np.exp(1j * omega0 * eta) * np.exp(-0.5 * eta**2)
        # W(b) = s^-1/2 sum_t y(t) psi*((t-b)/s)
        W[i] = fftconvolve(y, np.conj(psi[::-1]), mode="same") / np.sqrt(s)
    return W


rng = np.random.default_rng(1)
dt = 0.1
N = 2048
x = np.arange(N) * dt
y = rng.standard_normal(N)
y = (y - y.mean()) / y.std()

w = ComplexMorletWavelet(disable_caching=True)
cw = w.cwt(y, dt)
omega0 = 2 * np.pi * w.center_frequency
s_samp = cw.scales
mid = slice(N // 4, 3 * N // 4)

Wb = brute_cwt(y, s_samp, omega0)
Wt = tc_cwt(y, dt, cw.periods, omega0)
Wtp = tc_cwt(y, dt, cw.periods, omega0, pad_to=4 * N)

const = (2 * np.pi) ** -0.5 * np.pi**0.25  # (pi*B)^-1/2 / pi^-1/4 for B=2
print("expected constant ratio pywt/T&C:", const)
for name, Wr in [("brute (zero pad)", Wb), ("T&C fft no pad", Wt), ("T&C fft pad 4N", Wtp)]:
    ratio = np.abs(cw.W[:, mid]) / np.abs(Wr[:, mid])
    per_scale = np.median(ratio, axis=1)
    print(f"{name:18s} per-scale median ratio: first {per_scale[0]:.4f} mid {per_scale[len(per_scale)//2]:.4f} last {per_scale[-1]:.4f}; p1/p99 {np.percentile(ratio,[1,99])}")
    # relative error after removing the constant
    rel = np.abs(cw.W[:, mid] / const - Wr[:, mid]) / np.abs(Wr[:, mid]).max()
    print(f"{'':18s} max rel error (vs max |W|) per scale: first {rel[0].max():.2e} mid {rel[len(rel)//2].max():.2e} last {rel[-1].max():.2e}")

# WTC comparison with padded reference, same kernel as HyPyP
def smooth_time(W, s_samp, nfft):
    n = W.shape[1]
    k = 2 * np.pi * np.fft.fftfreq(nfft)
    F = np.exp(-0.5 * (s_samp[:, None] ** 2) * k[None, :] ** 2)
    return np.fft.ifft(F * np.fft.fft(W, n=nfft, axis=1), axis=1)[:, :n]


def wtc_ref(W1, W2, s_samp, kernel, mode, nfft):
    s = s_samp[:, None]
    S1 = convolve1d(smooth_time(np.abs(W1) ** 2 / s, s_samp, nfft).real, kernel, axis=0, mode=mode)
    S2 = convolve1d(smooth_time(np.abs(W2) ** 2 / s, s_samp, nfft).real, kernel, axis=0, mode=mode)
    S12 = np.abs(convolve1d(smooth_time(W1 * W2.conj() / s, s_samp, nfft), kernel, axis=0, mode=mode))
    return S12**2 / (S1 * S2)


P = 10.0
y1 = np.sin(2 * np.pi * x / P) + rng.standard_normal(N)
y2 = np.sin(2 * np.pi * x / P + 0.7) + rng.standard_normal(N)
res = w.wtc(PairSignals(x, y1, y2))
inside = ~res.wtc_masked.mask
z1 = (y1 - y1.mean()) / y1.std()
z2 = (y2 - y2.mean()) / y2.std()
W1 = brute_cwt(z1, s_samp, omega0)
W2 = brute_cwt(z2, s_samp, omega0)
hy_kernel = BaseWavelet._get_smoothing_window(w.wtc_smoothing_win_size, w.dj)
R = wtc_ref(W1, W2, s_samp, hy_kernel, "nearest", N)
d = np.abs(R - res.W)
print("\n[brute CWT + HyPyP kernel + same edge handling] max|diff| inside COI:", d[inside].max(), " mean:", d[inside].mean())

dj0steps = 0.6 / (2 * w.dj)
gk = np.r_[dj0steps % 1, np.ones(2 * round(dj0steps) - 1), dj0steps % 1]
gk /= gk.sum()
Rg = wtc_ref(W1, W2, s_samp, gk, "constant", N)
d = np.abs(Rg - res.W)
print("[brute CWT + Grinsted kernel (8.2 steps), zero edge] max|diff| inside COI:", d[inside].max(), " mean:", d[inside].mean())
print("mean WTC inside COI: HyPyP", res.W[inside].mean(), " Grinsted-kernel ref", Rg[inside].mean())
