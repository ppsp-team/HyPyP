"""prep.py: filt in-place, get_rejection_threshold determinism, corrmap no-match, ICA_choice_comp non-interactive."""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np
from autoreject import get_rejection_threshold

from hypyp import prep

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz"]
rng = np.random.RandomState(0)

print("===== filt: in place? zero-phase? preload required?")
info = mne.create_info(CH, 250, "eeg")
raw = mne.io.RawArray(rng.randn(len(CH), 5000) * 1e-5, info)
before = raw.get_data().copy()
(out,) = prep.filt([raw])
print(f"  returned object is the input: {out is raw}; input data modified: {not np.array_equal(before, raw.get_data())}")
print(f"  info highpass/lowpass after: {raw.info['highpass']}/{raw.info['lowpass']}")
# phase: filter an impulse, check symmetry
imp = np.zeros((1, 5001))
imp[0, 2500] = 1.0
r = mne.io.RawArray(imp, mne.create_info(["a"], 250, "eeg"))
(rf,) = prep.filt([r], (2.0, 40.0))
h = rf.get_data()[0]
print(f"  impulse response symmetric about impulse (zero-phase): {np.allclose(h[2500-200:2500], h[2500+1:2500+201][::-1], atol=1e-9)}")

print("\n===== get_rejection_threshold (used by ICA_fit) deterministic without random_state?")
data = rng.randn(60, len(CH), 250) * 1e-5
ep = mne.EpochsArray(data, info)
ep.set_montage("standard_1020")
t1 = get_rejection_threshold(ep, ch_types="eeg", verbose=False)
t2 = get_rejection_threshold(ep, ch_types="eeg", verbose=False)
print(f"  two unseeded calls equal: {t1 == t2}  ({t1}, {t2})")

print("\n===== ICA_fit + ICA_apply: alignment and corrmap with no match")
ep1 = mne.EpochsArray(rng.randn(60, len(CH), 250) * 1e-5, info)
ep2 = mne.EpochsArray(rng.randn(60, len(CH), 250) * 1e-5, info)
for e in (ep1, ep2):
    e.set_montage("standard_1020")
icas = prep.ICA_fit([ep1, ep2], n_components=6, method="fastica", fit_params=dict(tol=0.05, max_iter=500), random_state=97)
print(f"  n epochs after ICA_fit (input untouched): {len(ep1)}, {len(ep2)}")
# use a template that is unlikely to match anything at 0.9 in the second subject: random noise ICA
try:
    cleaned = prep.ICA_apply(icas, 0, 0, [ep1, ep2], plot=False)
    print(f"  ICA_apply OK: lengths {[len(c) for c in cleaned]}; excludes {[i.exclude for i in icas]}")
except Exception as e:  # noqa: BLE001
    print(f"  ICA_apply raised {type(e).__name__}: {e}")

print("\n===== ICA_choice_comp with stdin closed (non-interactive run)")
try:
    prep.ICA_choice_comp(icas, [ep1, ep2])
    print("  returned")
except BaseException as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e!r}")
sys.stdout.flush()
