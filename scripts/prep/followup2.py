"""Kuramoto solver tolerance; MVARICA ordering under a non-identity mixing."""

import warnings

import numpy as np
from scipy.integrate import solve_ivp

from hypyp.mvarica import MVAR, connectivity_mvarica

warnings.filterwarnings("ignore")

print("===== Kuramoto RHS copied from utils.generate_virtual_epoch, W[2,0]=20, default vs tight solve_ivp tolerances")
n_chan, n_samp, n_epo, sfreq = 4, 500, 4, 250.0
Nt = n_samp * n_epo
tv = np.linspace(0.0, n_samp / sfreq * n_epo, Nt)
W = np.zeros((n_chan, n_chan))
W[2, 0] = 20.0
np.random.seed(0)
omega = 2.0 * np.pi * (10 + 0.5 * np.random.randn(n_chan))
p0 = 2 * np.pi * np.block([np.zeros(2), np.random.rand(2) + 0.5])
print(f"  natural freqs (Hz): {np.round(omega / (2 * np.pi), 3).tolist()}  |w0-w2|={abs(omega[0]-omega[2]):.2f} rad/s < K=20 -> must lock")


def fp(t, p):
    p = np.atleast_2d(p)
    coupling = np.squeeze((np.sin(p) * np.matmul(W, np.cos(p).T).T) - (np.cos(p) * np.matmul(W, np.sin(p).T).T))
    return omega - coupling  # noise term omitted (0.005/n_samp, negligible)


for label, kw in [("library defaults (rtol=1e-3, atol=1e-6)", {}), ("rtol=1e-9, atol=1e-12", dict(rtol=1e-9, atol=1e-12)), ("defaults + max_step=1/sfreq", dict(max_step=1 / sfreq))]:
    ans = solve_ivp(fp, (tv[0], tv[-1]), p0, t_eval=tv, **kw)
    dphi = ans.y[0] - ans.y[2]
    last = slice(-n_samp, None)
    plv = np.abs(np.mean(np.exp(1j * dphi[last])))
    print(f"  {label}: n_steps={ans.t.size if False else len(ans.t)} nfev={ans.nfev}; last-epoch PLV(0,2) from solver phases = {plv:.3f}; std(dphi) = {np.std(dphi[last]):.3f} rad; |p| at end ~ {abs(ans.y[0,-1]):.0f} rad")

print("\n===== MVARICA under a fixed random mixing M (volume conduction), two independent data sets")
B1 = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.4], [0.0, 0.0, 0.5]])
B2 = -0.2 * np.eye(3)
M = np.random.RandomState(42).randn(3, 3)


def gen(seed):
    rng = np.random.RandomState(seed)
    s = np.zeros((5, 3, 4000))
    for e in range(5):
        for t in range(2, 4000):
            s[e, :, t] = B1 @ s[e, :, t - 1] + B2 @ s[e, :, t - 2] + rng.laplace(size=3)
    return np.einsum("ij,ejt->eit", M, s)


for seed in (0, 1, 2):
    out = connectivity_mvarica(gen(seed), {"method": "infomax_extended", "random_state": 0}, "pdc", n_fft=16, var_model=MVAR(2))
    m = out.mean(-1)
    np.fill_diagonal(m, 0)
    links = [tuple(int(v) for v in idx) for idx in np.argwhere(m > 0.2)]
    print(f"  seed {seed}: links (i<-j) with mean PDC>0.2: {links}   (source-space truth: (0<-1),(1<-2); returned indices are ICA sources, not channels)")
