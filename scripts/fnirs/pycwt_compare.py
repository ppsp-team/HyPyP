"""Cross-implementation: HyPyP PycwtWavelet (pycwt.wct) vs ComplexMorletWavelet (pywt)."""

import numpy as np

from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.wavelet.implementations.pycwt_wavelet import PycwtWavelet
import pycwt

print("pycwt", pycwt.__version__, " PycwtWavelet importable:", PycwtWavelet is not None)

rng = np.random.default_rng(2)
dt = 0.1
N = 2048
x = np.arange(N) * dt
P = 10.0
y1 = np.sin(2 * np.pi * x / P) + rng.standard_normal(N)
y2 = np.sin(2 * np.pi * x / P + 0.7) + rng.standard_normal(N)
pair = PairSignals(x, y1, y2)

pc = PycwtWavelet(disable_caching=True)
rp = pc.wtc(pair)
print("pycwt wct periods:", rp.periods[0], "...", rp.periods[-1], " n", len(rp.periods), " W shape", rp.W.shape)

# HyPyP pywt with same omega0 = 6, on the pycwt period grid
w = ComplexMorletWavelet(center_frequency=6 / (2 * np.pi), disable_caching=True, period_range=(rp.periods[0], rp.periods[-1]), dj=1 / 12)
rw = w.wtc(pair)
print("pywt periods:", rw.periods[0], "...", rw.periods[-1], " n", len(rw.periods))

# interpolate pywt WTC onto pycwt periods (log axis) and compare inside the pywt COI
from scipy.interpolate import interp1d

f = interp1d(np.log(rw.periods), rw.W, axis=0, bounds_error=False)
Wi = f(np.log(rp.periods))
ok = np.isfinite(Wi).all(axis=1)
mask = rp.periods[:, None] > rp.coi[None, :]
inside = (~mask) & ok[:, None]
d = np.abs(Wi - rp.W)
print("max|diff| inside COI (omega0 matched, grids interpolated):", np.nanmax(d[inside]), " mean|diff|:", np.nanmean(d[inside]))

# same but default HyPyP (omega0 = 2pi)
w2 = ComplexMorletWavelet(disable_caching=True, period_range=(rp.periods[0], rp.periods[-1]))
rw2 = w2.wtc(pair)
Wi2 = interp1d(np.log(rw2.periods), rw2.W, axis=0, bounds_error=False)(np.log(rp.periods))
d2 = np.abs(Wi2 - rp.W)
print("max|diff| inside COI (default omega0=2pi vs pycwt 6):", np.nanmax(d2[inside]), " mean|diff|:", np.nanmean(d2[inside]))

# white noise means
vals_p, vals_w = [], []
for _ in range(5):
    a = rng.standard_normal(N)
    b = rng.standard_normal(N)
    pp = PycwtWavelet(disable_caching=True).wtc(PairSignals(x, a, b))
    ww = ComplexMorletWavelet(disable_caching=True).wtc(PairSignals(x, a, b))
    vals_p.append(pp.coherence_metric)
    vals_w.append(ww.coherence_metric)
print("white noise mean WTC in COI: pycwt", np.mean(vals_p), " pywt default", np.mean(vals_w))
