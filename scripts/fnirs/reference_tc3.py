"""Is pywt's CWT (as called by HyPyP) an accurate L2 Morlet CWT at every scale of the default grid?"""

import inspect

import numpy as np
import pywt
from scipy.signal import fftconvolve

from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet

print("pywt.cwt signature:", inspect.signature(pywt.cwt))


def brute_cwt(y, scales_samples, omega0):
    n = len(y)
    W = np.empty((len(scales_samples), n), dtype=complex)
    for i, s in enumerate(scales_samples):
        half = int(np.ceil(8 * s))
        eta = np.arange(-half, half + 1) / s
        psi = np.pi**-0.25 * np.exp(1j * omega0 * eta) * np.exp(-0.5 * eta**2)
        W[i] = fftconvolve(y, np.conj(psi[::-1]), mode="same") / np.sqrt(s)
    return W


dt = 0.1
N = 2048
x = np.arange(N) * dt
w = ComplexMorletWavelet(disable_caching=True)
omega0 = 2 * np.pi * w.center_frequency
const = (2 * np.pi) ** -0.5 * np.pi**0.25
mid = slice(N // 4, 3 * N // 4)

# 1. white noise: conjugation-aware complex error and magnitude error, per scale
rng = np.random.default_rng(3)
y = rng.standard_normal(N)
cw = w.cwt(y, dt)
Wr = brute_cwt(y, cw.scales, omega0)
Wp = cw.W / const
for label, Wc in [("as is", Wr), ("conj", np.conj(Wr))]:
    err = np.abs(Wp[:, mid] - Wc[:, mid]).max(axis=1) / np.abs(Wc[:, mid]).max(axis=1)
    print(f"complex error vs {label:5s}: per-scale max rel err: min {err.min():.2e} max {err.max():.2e}")
magerr = np.abs(np.abs(Wp[:, mid]) - np.abs(Wr[:, mid])).max(axis=1) / np.abs(Wr[:, mid]).max(axis=1)
print("magnitude error per scale (max over time / max|W|): worst", magerr.max(), "at period", cw.periods[np.argmax(magerr)], " best", magerr.min())
print("period of each of the 3 worst scales:", cw.periods[np.argsort(magerr)[-3:]])

# 2. pure sinusoid at each scale's period: closed form |W| = 0.5*sqrt(2*pi*s)*pi^-1/4 (T&C discrete, L2)
print("\nsinusoid amplitude check (pywt |W| / const vs closed form), middle of record:")
for P in [2.0, 4.0, 8.0, 16.0, 20.0]:
    y = np.cos(2 * np.pi * x / P)
    cw = w.cwt(y, dt)
    i = np.argmin(np.abs(cw.periods - P))
    s = cw.scales[i]
    closed = 0.5 * np.sqrt(2 * np.pi * s) * np.pi**-0.25 * np.exp(-0.5 * (s * 2 * np.pi * dt / P - omega0) ** 2)
    got = np.abs(cw.W[i, mid]).mean() / const
    print(f"  P={P:5.1f} s (grid period {cw.periods[i]:.3f}, scale {s:.1f} samples): pywt {got:.4f}  closed form {closed:.4f}  ratio {got/closed:.4f}")

# 3. method='conv' vs 'fft' and precision
y = rng.standard_normal(N)
Wf, _ = pywt.cwt(y, w.get_scales(dt), w._wavelet, sampling_period=dt, method="fft")
Wc, _ = pywt.cwt(y, w.get_scales(dt), w._wavelet, sampling_period=dt, method="conv")
print("\nfft vs conv method max|diff|/max|W|:", np.abs(Wf - Wc).max() / np.abs(Wf).max())
Wr = brute_cwt(y, w.get_scales(dt), omega0)
for prec in [10, 12, 14]:
    Wp, _ = pywt.cwt(y, w.get_scales(dt), w._wavelet, sampling_period=dt, method="fft", precision=prec)
    e = np.abs(np.abs(Wp[:, mid]) / const - np.abs(Wr[:, mid])).max(axis=1) / np.abs(Wr[:, mid]).max(axis=1)
    print(f"precision={prec}: worst per-scale magnitude error {e.max():.3e}")
