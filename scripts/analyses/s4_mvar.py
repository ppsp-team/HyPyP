"""compute_conn_mvar: merge-reshape scrambling, direction convention, return types."""
import warnings
import numpy as np
from hypyp import analyses
from hypyp.mvarica import MVAR, connectivity_mvarica

warnings.simplefilter("ignore")

# 1. The merge reshape at analyses.py:766
n_epoch, n_freq, n_ch2, n_samp = 2, 1, 2, 3
real_signal = np.arange(n_epoch * n_freq * n_ch2 * n_samp, dtype=float).reshape(n_epoch, n_freq, n_ch2, n_samp)
aug_signal = real_signal[np.newaxis, ...]
merged = aug_signal.reshape(1, real_signal.shape[1], real_signal.shape[2], real_signal.shape[3] * real_signal.shape[0])
correct = np.concatenate([real_signal[e] for e in range(n_epoch)], axis=-1)[np.newaxis]
print("epoch 0 ch0:", real_signal[0, 0, 0], " epoch 1 ch0:", real_signal[1, 0, 0])
print("code's merged ch0:", merged[0, 0, 0])
print("code's merged ch1:", merged[0, 0, 1])
print("correct merged ch0:", correct[0, 0, 0])
print("identical?", np.array_equal(merged, correct))

# 2. Direction convention: VAR(1) where x1 drives x2 (x2[t] = 0.8*x1[t-1] + noise)
rng = np.random.default_rng(3)
N = 4000
x1 = rng.standard_normal(N)
x2 = np.zeros(N)
for i in range(1, N):
    x2[i] = 0.8 * x1[i - 1] + 0.3 * rng.standard_normal()
sig = np.stack([x1, x2])[np.newaxis]  # (1 epoch, 2 ch, N)
mvar = MVAR(2, "default", 0)
ica_params = {"method": "infomax_extended", "random_state": 0}
for name in ["pdc", "dtf"]:
    out = connectivity_mvarica(sig, ica_params, name, n_fft=16, var_model=mvar)
    m = out.mean(-1)
    print(f"{name}: [i,j] mean over freq =\n{np.round(m, 3)}  -> [1,0] (0->1) = {m[1,0]:.3f}, [0,1] (1->0) = {m[0,1]:.3f}")

# 3. compute_conn_mvar with check_stability=False: dtype and shape, direction preserved?
cs = np.zeros((2, 1, 1, 1, N), dtype=complex)
cs[0, 0, 0, 0] = x1
cs[1, 0, 0, 0] = x2
out = analyses.compute_conn_mvar(cs, {"mvar_order": 2, "fitting_method": "default", "delta": 0},
                                 ica_params, {"name": "pdc", "n_fft": 16}, check_stability=False)
print("compute_conn_mvar shape", out.shape, "dtype", out.dtype, "mean[1,0]", np.round(out[0, 0, 1, 0].mean(), 3), "mean[0,1]", np.round(out[0, 0, 0, 1].mean(), 3))
print("docstring return shape says (n_epochs, n_freq, n_channels, n_channels, n_fft); n_channels here is 2*n_ch =", out.shape[2])
