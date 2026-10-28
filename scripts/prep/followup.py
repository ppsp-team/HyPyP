"""Follow-up: Kuramoto accuracy/locking, MVARICA source permutation."""

import warnings

import mne
import numpy as np
from scipy.signal import hilbert

from hypyp import utils
from hypyp.mvarica import MVAR, connectivity_mvarica

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

n_ch, n_epo, n_t, sfreq = 4, 4, 500, 250.0
info = mne.create_info([f"C{i}" for i in range(n_ch)], sfreq, "eeg")
tmpl = mne.EpochsArray(np.zeros((n_epo, n_ch, n_t)), info)


def plv(x, y):
    px, py = np.angle(hilbert(x)), np.angle(hilbert(y))
    return np.abs(np.mean(np.exp(1j * (px - py))))


print("===== Kuramoto: integration accuracy, W=0, no noise, frequency_std=0 (analytic: sin(2*pi*10*t + p0))")
np.random.seed(3)
sim = utils.generate_virtual_epoch(tmpl, np.zeros((n_ch, n_ch)), frequency_std=0.0, noise_phase_level=0.0, noise_amplitude_level=0.0)
d = sim.get_data(copy=True)
x = np.concatenate([d[e, 0] for e in range(n_epo)])  # channel 0 starts at phase 0
Nt = n_t * n_epo
t_lib = np.linspace(0.0, n_t / sfreq * n_epo, Nt)  # the grid the function integrates on
t_nom = np.arange(Nt) / sfreq  # the grid implied by sfreq
err_lib = np.abs(x - np.sin(2 * np.pi * 10 * t_lib)).max()
err_nom = np.abs(x - np.sin(2 * np.pi * 10 * t_nom)).max()
print(f"  max|sim - analytic| on solver grid = {err_lib:.3e}; on nominal 1/sfreq grid = {err_nom:.3e}")
psd = np.abs(np.fft.rfft(x))
freqs = np.fft.rfftfreq(Nt, 1 / sfreq)
print(f"  peak frequency on nominal grid = {freqs[np.argmax(psd)]:.3f} Hz (expected 10.000, bias 10*Nt/(Nt-1)={10*Nt/(Nt-1):.4f})")

print("\n===== Kuramoto: strong unidirectional coupling W[2,0]=20 (locking expected)")
W = np.zeros((n_ch, n_ch))
W[2, 0] = 20.0
np.random.seed(0)
sim = utils.generate_virtual_epoch(tmpl, W, frequency_std=0.5, noise_amplitude_level=0.0).get_data(copy=True)
print(f"  PLV(ch0,ch2) last epoch = {plv(sim[-1,0], sim[-1,2]):.3f}; uncoupled PLV(ch0,ch3) = {plv(sim[-1,0], sim[-1,3]):.3f}")
# does the driven oscillator (2) adopt the driver's (0) frequency, i.e. does W[i,j] mean j->i?
f0 = freqs_e = np.fft.rfftfreq(n_t * n_epo, 1 / sfreq)
for ch in (0, 2):
    xx = np.concatenate([sim[e, ch] for e in range(n_epo)])
    print(f"  ch{ch} peak freq = {f0[np.argmax(np.abs(np.fft.rfft(xx)))]:.2f} Hz")

print("\n===== MVARICA: source ordering with identifiable (Laplacian) innovations, two independent data sets")
B1 = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.4], [0.0, 0.0, 0.5]])
B2 = -0.2 * np.eye(3)


def gen(seed):
    rng = np.random.RandomState(seed)
    x = np.zeros((5, 3, 4000))
    for e in range(5):
        for t in range(2, 4000):
            x[e, :, t] = B1 @ x[e, :, t - 1] + B2 @ x[e, :, t - 2] + rng.laplace(size=3)
    return x


for seed in (0, 1):
    out = connectivity_mvarica(gen(seed), {"method": "infomax_extended", "random_state": 0}, "pdc", n_fft=16, var_model=MVAR(2))
    m = out.mean(-1)
    np.fill_diagonal(m, 0)
    print(f"  seed {seed}: off-diagonal PDC (mean over f), rounded:\n{np.round(m, 2)}")
    print(f"    strongest links (i<-j): {[tuple(int(v) for v in idx) for idx in np.argwhere(m > 0.2)]}  (truth: (0<-1), (1<-2))")
