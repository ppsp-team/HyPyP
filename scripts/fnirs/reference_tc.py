"""Independent reference: Torrence & Compo (1998) Morlet CWT + Grinsted et al. (2004) smoothing.

Compares with HyPyP ComplexMorletWavelet on the same signals and the same period grid.
"""

import numpy as np

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet


def tc_cwt(y, dt, scales, omega0):
    """T&C 1998 eq. 4 + 6 (Fourier-domain Morlet, L2 normalised), no padding."""
    n = len(y)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    yhat = np.fft.fft(y)
    W = np.empty((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        psi_hat = (
            np.sqrt(2 * np.pi * s / dt) * np.pi**-0.25 * np.exp(-0.5 * (s * k - omega0) ** 2) * (k > 0)
        )
        W[i] = np.fft.ifft(yhat * psi_hat)
    return W


def smooth_time(W, scales_samples, pad=False):
    """Gaussian in time with std = scale (in samples), via FFT (Grinsted smoothwavelet.m)."""
    n = W.shape[1]
    nfft = int(2 ** np.ceil(np.log2(n))) if pad else n
    k = 2 * np.pi * np.fft.fftfreq(nfft)
    F = np.exp(-0.5 * (scales_samples[:, None] ** 2) * k[None, :] ** 2)
    return np.fft.ifft(F * np.fft.fft(W, n=nfft, axis=1), axis=1)[:, :n]


def smooth_scale(T, kernel, mode):
    from scipy.ndimage import convolve1d

    return convolve1d(T, kernel, axis=0, mode=mode)


def grinsted_kernel(dj, dj0=0.6):
    dj0steps = dj0 / (dj * 2)
    k = np.r_[dj0steps % 1, np.ones(2 * round(dj0steps) - 1), dj0steps % 1]
    return k / k.sum()


def wtc_ref(y1, y2, dt, periods_sec, omega0, kernel, mode, pad):
    y1 = (y1 - y1.mean()) / y1.std()
    y2 = (y2 - y2.mean()) / y2.std()
    # scale in seconds such that pywt-style period = scale (fc = omega0/2pi, period = scale/fc*... )
    # HyPyP uses period = scale*dt/fc with fc = omega0/(2*pi); here scale_sec = periods*fc
    fc = omega0 / (2 * np.pi)
    scales_sec = periods_sec * fc
    W1 = tc_cwt(y1, dt, scales_sec, omega0)
    W2 = tc_cwt(y2, dt, scales_sec, omega0)
    s = (scales_sec / dt)[:, None]
    S1 = smooth_scale(smooth_time(np.abs(W1) ** 2 / s, s[:, 0], pad).real, kernel, mode)
    S2 = smooth_scale(smooth_time(np.abs(W2) ** 2 / s, s[:, 0], pad).real, kernel, mode)
    S12 = np.abs(smooth_scale(smooth_time(W1 * W2.conj() / s, s[:, 0], pad), kernel, mode))
    return S12**2 / (S1 * S2), W1


rng = np.random.default_rng(1)
dt = 0.1
N = 2048  # power of two so that padding is a no-op in HyPyP
x = np.arange(N) * dt
P = 10.0
y1 = np.sin(2 * np.pi * x / P) + rng.standard_normal(N)
y2 = np.sin(2 * np.pi * x / P + 0.7) + rng.standard_normal(N)

w = ComplexMorletWavelet(disable_caching=True)
res = w.wtc(PairSignals(x, y1, y2))
omega0 = 2 * np.pi * w.center_frequency
periods = res.periods

inside = ~res.wtc_masked.mask

print("HyPyP scales (samples) == periods/dt ?", np.allclose(res.scales, periods / dt))

# 1. same smoothing window as HyPyP, same edge mode -> isolates the CWT + normalisation
hy_kernel = BaseWavelet._get_smoothing_window(w.wtc_smoothing_win_size, w.dj)
R, W1ref = wtc_ref(y1, y2, dt, periods, omega0, hy_kernel, "nearest", pad=True)
d = np.abs(R - res.W)
print("[same kernel as HyPyP] max|diff| all:", d.max(), " inside COI:", d[inside].max())

# CWT magnitude ratio pywt vs T&C (should be a constant across scales and times)
cw = w.cwt((y1 - y1.mean()) / y1.std(), dt)
ratio = np.abs(cw.W) / np.abs(W1ref)
mid = slice(N // 4, 3 * N // 4)
print("|W| ratio pywt/T&C: median", np.median(ratio[:, mid]), " spread (p1,p99)", np.percentile(ratio[:, mid], [1, 99]))

# 2. Grinsted kernel (dj0=0.6, width dj0/dj + 1 steps) -> effect of the 2x window
gk = grinsted_kernel(w.dj)
Rg, _ = wtc_ref(y1, y2, dt, periods, omega0, gk, "constant", pad=False)
d = np.abs(Rg - res.W)
print("[Grinsted kernel, no pad, zero-edge] max|diff| inside COI:", d[inside].max(), " mean|diff|:", d[inside].mean())
print("HyPyP kernel len", len(hy_kernel), " Grinsted kernel len", len(gk))

# 3. independent noise mean WTC in COI, both kernels
vals_h, vals_g = [], []
for _ in range(10):
    a = rng.standard_normal(N)
    b = rng.standard_normal(N)
    rh = w.wtc(PairSignals(x, a, b))
    rg, _ = wtc_ref(a, b, dt, periods, omega0, gk, "constant", pad=False)
    vals_h.append(rh.coherence_metric)
    vals_g.append(np.mean(rg[~rh.wtc_masked.mask]))
print("white noise mean WTC in COI: HyPyP kernel", np.mean(vals_h), " Grinsted kernel", np.mean(vals_g))
