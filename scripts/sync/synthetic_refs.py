"""Independent scalar references for the 9 metrics on synthetic 2-channel data.

The references below use np.angle / np.corrcoef / scipy.stats.circmean and
plain loops; none of them calls hypyp.sync.base.multiply_*.
"""
import warnings

import numpy as np
from scipy.stats import circmean

from hypyp.sync import get_metric

warnings.simplefilter("ignore")
AX = (0, 1, 3, 2)
rng = np.random.default_rng(0)
T = 4000


def run(mode, x, y):
    sig = np.stack([x, y])[None, None]  # (1,1,2,T)
    m = get_metric(mode) if mode != "accorr" else get_metric(mode)
    if mode == "accorr":
        m.show_progress = False
    return m.compute(sig, T, AX)[0, 0]


# ---- independent references --------------------------------------------
def ref_plv(x, y):
    return abs(np.mean(np.exp(1j * (np.angle(x) - np.angle(y)))))


def ref_ccorr_signed(x, y):  # Jammalamadaka & SenGupta 2001, eq. 8.2.2
    a, b = np.angle(x), np.angle(y)
    am, bm = circmean(a, high=np.pi, low=-np.pi), circmean(b, high=np.pi, low=-np.pi)
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    return num / np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))


def ref_coherency(x, y):  # complex coherency, Nolte 2004 eq. 1
    return np.mean(x * np.conj(y)) / np.sqrt(np.mean(abs(x) ** 2) * np.mean(abs(y) ** 2))


def ref_pli(x, y):  # Stam 2007
    return abs(np.mean(np.sign(np.imag(x * np.conj(y)))))


def ref_wpli(x, y):  # Vinck 2011 eq. 8 (not debiased)
    im = np.imag(x * np.conj(y))
    return abs(np.mean(im)) / np.mean(abs(im))


def ref_wpli_debiased(x, y):  # Vinck 2011 debiased estimator
    im = np.imag(x * np.conj(y))
    n = len(im)
    return (np.sum(im) ** 2 - np.sum(im**2)) / (np.sum(abs(im)) ** 2 - np.sum(im**2))


def ref_env(x, y):
    return np.corrcoef(abs(x), abs(y))[0, 1]


def ref_pow(x, y):
    return np.corrcoef(abs(x) ** 2, abs(y) ** 2)[0, 1]


def ref_accorr(x, y):
    """Centred sine product with the pair-specific centring (m, n) of
    Zimmermann et al. 2024, written from the phase arrays only."""
    a, b = np.angle(x), np.angle(y)
    md = np.angle(np.mean(np.exp(1j * (a - b))))
    ms = np.angle(np.mean(np.exp(1j * (a + b))))
    m, n = (md + ms) / 2, (ms - md) / 2
    num = np.sum(np.sin(a - m) * np.sin(b - n))
    return num / np.sqrt(np.sum(np.sin(a - m) ** 2) * np.sum(np.sin(b - n) ** 2))


# ---- case A: random complex Gaussian pair, y = x + noise ---------------
x = rng.standard_normal(T) + 1j * rng.standard_normal(T)
y = x + (rng.standard_normal(T) + 1j * rng.standard_normal(T)) * 1j * 0.7 + 0.3 * x * 1j
print("=== case A: random pair ===")
C = ref_coherency(x, y)
rows = [
    ("plv", run("plv", x, y)[0, 1], ref_plv(x, y)),
    ("ccorr", run("ccorr", x, y)[0, 1], abs(ref_ccorr_signed(x, y))),
    ("ccorr signed ref", None, ref_ccorr_signed(x, y)),
    ("accorr", run("accorr", x, y)[0, 1], ref_accorr(x, y)),
    ("coh  (|coherency|)", run("coh", x, y)[0, 1], abs(C)),
    ("coh  (MSC=|C|^2)", None, abs(C) ** 2),
    ("imcoh (|Im C|)", run("imcoh", x, y)[0, 1], abs(C.imag)),
    ("imcoh signed ref", None, C.imag),
    ("pli", run("pli", x, y)[0, 1], ref_pli(x, y)),
    ("wpli", run("wpli", x, y)[0, 1], ref_wpli(x, y)),
    ("wpli debiased ref", None, ref_wpli_debiased(x, y)),
    ("envcorr", run("envcorr", x, y)[0, 1], ref_env(x, y)),
    ("powcorr", run("powcorr", x, y)[0, 1], ref_pow(x, y)),
]
for name, got, exp in rows:
    print(f"{name:22s} hypyp={'' if got is None else f'{got:.10f}':>14s} ref={exp:.10f}")

# ---- case B: constant lag pi/4, unit amplitude -------------------------
print("\n=== case B: y = x * exp(-i pi/4), |x|=1, phase uniform on circle ===")
phi = rng.uniform(-np.pi, np.pi, T)
x = np.exp(1j * phi)
y = np.exp(1j * (phi - np.pi / 4))
exp_B = dict(plv=1, ccorr=1, accorr=1, coh=1, imcoh=np.sin(np.pi / 4), pli=1, wpli=1)
for mode, e in exp_B.items():
    print(f"{mode:8s} hypyp={run(mode, x, y)[0, 1]:.10f} expected={e:.10f}")
print(
    f"ccorr circ-mean resultant length R of x: "
    f"{abs(np.mean(np.exp(1j*phi))):.3e} (centring ill-defined when ~0)"
)

# ---- case C: amplitude relation A2 = 2*A1 + 1 ----------------------------
print("\n=== case C: envelope relation |y| = 2|x| + 1, same phase ===")
A1 = rng.uniform(0.5, 2.0, T)
x = A1 * np.exp(1j * phi)
y = (2 * A1 + 1) * np.exp(1j * phi)
print(f"envcorr hypyp={run('envcorr', x, y)[0,1]:.10f} expected=1")
print(f"powcorr hypyp={run('powcorr', x, y)[0,1]:.10f} ref(np.corrcoef)={ref_pow(x, y):.10f}")

# ---- case D: y = x (identical) ------------------------------------------
print("\n=== case D: y = x (identical channel) ===")
x = rng.standard_normal(T) + 1j * rng.standard_normal(T)
for mode in ["plv", "coh", "imcoh", "pli", "wpli", "ccorr", "accorr", "envcorr", "powcorr"]:
    r = run(mode, x, x)
    print(f"{mode:8s} offdiag={r[0,1]:.10f} diag={r[0,0]:.10f}")

# ---- case E: Coh not squared: y = x + n, |C| = 1/sqrt(2), MSC = 0.5 -----
print("\n=== case E: y = x + n, independent unit-variance complex Gaussians ===")
Tbig = 200_000
x = (rng.standard_normal(Tbig) + 1j * rng.standard_normal(Tbig)) / np.sqrt(2)
n = (rng.standard_normal(Tbig) + 1j * rng.standard_normal(Tbig)) / np.sqrt(2)
y = x + n
print(f"coh hypyp={run('coh', x, y)[0,1]:.5f}  |C| expected={1/np.sqrt(2):.5f}  MSC expected=0.5")

# ---- identity check for the ACCorr numerator ------------------------------
print("\n=== ACCorr numerator identity: sum sin(a-m) sin(b-n) == (r_minus - r_plus)/2 ===")
a, b = rng.uniform(-np.pi, np.pi, T), rng.uniform(-np.pi, np.pi, T)
md = np.angle(np.mean(np.exp(1j * (a - b))))
ms = np.angle(np.mean(np.exp(1j * (a + b))))
m, nn = (md + ms) / 2, (ms - md) / 2
lhs = np.sum(np.sin(a - m) * np.sin(b - nn))
rhs = (abs(np.sum(np.exp(1j * (a - b)))) - abs(np.sum(np.exp(1j * (a + b))))) / 2
print(f"lhs={lhs:.10f} rhs={rhs:.10f} diff={lhs-rhs:.2e}")
