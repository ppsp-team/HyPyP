"""mvarica.py checks."""

import warnings

import numpy as np

from hypyp.mvarica import MVAR, connectivity_mvarica

warnings.filterwarnings("ignore")
rng = np.random.RandomState(0)

# Reference VAR(2), 3 channels, chain c -> b -> a (B[0,1], B[1,2] nonzero)
B1 = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.4], [0.0, 0.0, 0.5]])
B2 = np.array([[-0.2, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, -0.2]])
n_epo, n_ch, n_t = 5, 3, 4000
x = np.zeros((n_epo, n_ch, n_t))
for e in range(n_epo):
    for t in range(2, n_t):
        x[e, :, t] = B1 @ x[e, :, t - 1] + B2 @ x[e, :, t - 2] + rng.randn(n_ch)

print("===== 1) MVAR fit recovers coefficients (order 2, delta=0)")
m = MVAR(model_order=2).fit(x)
Bhat1, Bhat2 = m.coeff[:, 0::2], m.coeff[:, 1::2]
print(f"  max|Bhat1-B1|={np.abs(Bhat1-B1).max():.3f}  max|Bhat2-B2|={np.abs(Bhat2-B2).max():.3f}")
print(f"  stability() = {m.stability()}")

print("\n===== 2) residuals: first p samples per epoch are raw signal, not innovations")
print(f"  residuals[:, :, :2] == signal[:, :, :2]: {np.array_equal(m.residuals[:, :, :2], x[:, :, :2])}")

print("\n===== 3) delta != 0 (documented ridge option)")
try:
    MVAR(model_order=2, delta=0.1).fit(x)
    print("  OK")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== 4) stability() with model_order=1")
try:
    print(f"  {MVAR(model_order=1).fit(x).stability()}")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== 5) connectivity_mvarica default var_model=MVAR (class, not instance)")
try:
    connectivity_mvarica(x, {"method": "infomax", "random_state": 0}, "pdc", n_fft=16)
    print("  OK")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== 6) custom fitting_method object (documented)")


class LS:
    def fit(self, X, y):
        self.coef = np.linalg.lstsq(X, y, rcond=None)[0].T
        return self


try:
    MVAR(model_order=2, fitting_method=LS()).fit(x)
    print("  OK")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== 7) PDC formula + direction vs hand-written Baccala PDC on the TRUE coefficients")
n_fft = 16
p = 2
A_true = np.zeros((n_ch, n_ch, n_fft), complex)
for k in range(n_fft):
    f = k / (2 * n_fft - 1)  # same frequency grid as fft(..., 2*n_fft-1)[:n_fft]
    A_true[:, :, k] = np.eye(n_ch) - B1 * np.exp(-2j * np.pi * f * 1) - B2 * np.exp(-2j * np.pi * f * 2)
pdc_ref = np.abs(A_true) / np.sqrt(np.sum(np.abs(A_true) ** 2, axis=0, keepdims=True))
# the library computes PDC via connectivity_mvarica after ICA, so emulate its spectral step on the fitted coeffs
from scipy.fftpack import fft

coeffs = m.coeff
re = np.reshape(coeffs, (n_ch, n_ch, p), "c")
a = fft(np.dstack([np.eye(n_ch), -re]), n_fft * 2 - 1)[:, :, :n_fft]
pdc_lib = np.abs(a / np.sqrt(np.sum(a.conj() * a, axis=0, keepdims=True)))
print(f"  max|pdc_lib(fitted) - pdc_ref(true)| = {np.abs(pdc_lib - pdc_ref).max():.3f}")
print(f"  mean over f: PDC[0,1] (b->a) = {pdc_lib[0,1].mean():.3f}  PDC[1,0] (a->b) = {pdc_lib[1,0].mean():.3f}  PDC[2,0] = {pdc_lib[2,0].mean():.3f}")

print("\n===== 8) full pipeline with ICA on Gaussian innovations: does the ICA rotation destroy the known structure?")
mv = MVAR(model_order=2)
out = connectivity_mvarica(x, {"method": "infomax_extended", "random_state": 0}, "pdc", n_fft=n_fft, var_model=mv)
print(f"  output shape {out.shape}; mean-over-f PDC matrix:\n{np.round(out.mean(-1), 2)}")
print(f"  reference (true, mean-over-f):\n{np.round(pdc_ref.mean(-1), 2)}")

print("\n===== 9) stability on an explosive VAR(2)")
y = np.zeros((1, 2, 500))
Bu = np.array([[1.2, 0.0], [0.0, 0.3]])
for t in range(2, 500):
    y[0, :, t] = Bu @ y[0, :, t - 1] + 1e-3 * rng.randn(2)
print(f"  stability() on explosive model = {MVAR(model_order=2).fit(y).stability()}")
