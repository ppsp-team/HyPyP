"""Input mutation (pow drops EOG from caller's Epochs), indices with EOG, behav_corr edge cases."""
import warnings
import numpy as np
import mne
from hypyp import analyses

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
rng = np.random.default_rng(5)
fs = 128
info = mne.create_info(["C1", "EOG1", "EOG2", "C2"], fs, ["eeg", "eog", "eog", "eeg"])
ep = mne.EpochsArray(rng.standard_normal((4, 4, 256)), info, verbose=False)
print("before pow:", ep.ch_names)
res = analyses.pow(ep, 1, 40, 128, 64, True)
print("after pow: ", ep.ch_names, "  psd shape", res.psd.shape)

# indices with EOG present in the merged epochs
info_h = mne.create_info(["C1_S1", "C2_S1", "EOG_S1", "C1_S2", "C2_S2", "EOG_S2"], fs,
                         ["eeg", "eeg", "eog", "eeg", "eeg", "eog"])
eph = mne.EpochsArray(rng.standard_normal((2, 6, 256)), info_h, verbose=False)
idx = analyses.indices_connectivity_interbrain(eph)
print("interbrain indices with EOG present:", idx, " -> names:", [(eph.ch_names[i], eph.ch_names[j]) for i, j in idx])

# behav_corr: 2-D data whose shape differs from behav -> returns?
out = analyses.behav_corr(rng.standard_normal((5, 3)), rng.standard_normal(5), "d", "b", 0.05)
print("behav_corr with 2-D data returns:", out)
# n < 8
try:
    analyses.behav_corr(rng.standard_normal(6), rng.standard_normal(6), "d", "b", 0.05)
    print("n=6 ok")
except Exception as exc:
    print("behav_corr n=6:", type(exc).__name__, str(exc)[:120])

# compute_freq_bands / compute_sync: do they mutate input?
d = rng.standard_normal((2, 2, 2, 512))
d0 = d.copy()
cs = analyses.compute_freq_bands(d, fs, {"a": [8, 12]})
cs0 = cs.copy()
analyses.compute_sync(cs, "accorr")
analyses.compute_sync(cs, "plv")
print("compute_freq_bands mutates data?", not np.array_equal(d, d0), " compute_sync mutates complex_signal?", not np.array_equal(cs, cs0))

# DeprecationWarning from compute_nmPLV using _multiply_conjugate
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    analyses.compute_nmPLV(d, fs, [8, 12], [16, 24])
    print("nmPLV warnings:", [f"{x.category.__name__}: {str(x.message)[:60]}" for x in w if x.category is DeprecationWarning])
