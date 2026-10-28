"""Synthetic checks of the HyPyP WTC: identity, independent noise, sinusoid peak period."""

import numpy as np

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.base_wavelet import BaseWavelet
from hypyp.wavelet.implementations.pywavelets_wavelet import (
    ComplexMorletWavelet,
    ComplexGaussianWavelet,
)

rng = np.random.default_rng(0)
dt = 0.1
N = 4000
x = np.arange(N) * dt

print("=== smoothing window actually used (win=0.6, dj=1/12) ===")
win = BaseWavelet._get_smoothing_window(0.6, 1 / 12)
print("len", len(win), "sum", win.sum(), "effective width in steps", 1 / win.max())
print("Grinsted kernel for dj0=0.6, dj=1/12: dj0steps=", 0.6 / (2 / 12))
dj0steps = 0.6 / (2 / 12)
gk = np.r_[dj0steps % 1, np.ones(2 * round(dj0steps) - 1), dj0steps % 1]
gk /= gk.sum()
print("Grinsted len", len(gk), "effective width in steps", 1 / gk.max())

print("\n=== period grid ===")
w = ComplexMorletWavelet(disable_caching=True)
p = w.get_periods()
print("n periods", len(p), "first/last", p[0], p[-1])
print("requested dj", w.dj, "actual dj (log2 step)", np.diff(np.log2(p))[0])

print("\n=== (a) identical signals ===")
y = rng.standard_normal(N)
res = w.wtc(PairSignals(x, y, y.copy()))
print("W min/max over everything:", res.W.min(), res.W.max())
print("W min/max inside COI:", res.wtc_masked.min(), res.wtc_masked.max())
print("coherence_metric", res.coherence_metric)

print("\n=== (b) independent white noise, mean WTC inside COI ===")
for label, kw in [
    ("default (win 0.6 -> 14.4 steps)", {}),
    ("Grinsted-equivalent (8.2 steps)", dict(wtc_smoothing_win_size=8.2 * (1 / 12) / 2)),
    ("cgau2 default", None),
]:
    vals = []
    for _ in range(10):
        y1 = rng.standard_normal(N)
        y2 = rng.standard_normal(N)
        if kw is None:
            wv = ComplexGaussianWavelet(disable_caching=True)
        else:
            wv = ComplexMorletWavelet(disable_caching=True, **kw)
        r = wv.wtc(PairSignals(x, y1, y2))
        vals.append(r.coherence_metric)
    print(f"{label:40s} mean WTC in COI = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

print("\n=== (c) two sinusoids of period 10 s + noise, dj=1/100 ===")
P = 10.0
w_fine = ComplexMorletWavelet(disable_caching=True, dj=1 / 100, period_range=(5, 20))
y1 = np.sin(2 * np.pi * x / P) + 0.5 * rng.standard_normal(N)
y2 = np.sin(2 * np.pi * x / P + 1.0) + 0.5 * rng.standard_normal(N)
r = w_fine.wtc(PairSignals(x, y1, y2))
prof = r.wtc_masked.mean(axis=1)
print("WTC peak period:", r.periods[np.argmax(prof)], " (peak value", prof.max(), ")")
cw = w_fine.cwt(np.sin(2 * np.pi * x / P), dt)
pw = (np.abs(cw.W) ** 2)[:, N // 4 : 3 * N // 4].mean(axis=1)
print("CWT power peak period (pure sinusoid):", cw.periods[np.argmax(pw)])
w0 = 2 * np.pi * w_fine.center_frequency
print("T&C prediction for pywt period at CWT power peak:", P * (w0 + np.sqrt(2 + w0**2)) / (2 * w0))
print("T&C Fourier factor for omega0=2pi:", 4 * np.pi / (w0 + np.sqrt(2 + w0**2)), " code flambda:", w_fine.flambda)

print("\n=== (c') same with cgau2 ===")
g_fine = ComplexGaussianWavelet(disable_caching=True, dj=1 / 100, period_range=(5, 20))
cg = g_fine.cwt(np.sin(2 * np.pi * x / P), dt)
pg = (np.abs(cg.W) ** 2)[:, N // 4 : 3 * N // 4].mean(axis=1)
print("cgau2 CWT power peak period:", cg.periods[np.argmax(pg)])
print("cgau2 flambda used for COI:", g_fine.flambda, " vs 1/fc(pywt)=2.5")

print("\n=== COI value at centre vs signal half-length ===")
print("coi max (s):", res.coi.max(), " N*dt/2 =", N * dt / 2)
