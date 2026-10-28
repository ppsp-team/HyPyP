"""AR_local dyad alignment with planted bad epochs."""

import warnings

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np

from hypyp import prep

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz"]
SFREQ = 250
N_EPO = 40
N_T = 250


def make_epochs(seed, bad_epochs):
    rng = np.random.RandomState(seed)
    data = rng.randn(N_EPO, len(CH), N_T) * 1e-5
    for k in bad_epochs:
        data[k] += 5e-4 * rng.randn(len(CH), N_T)  # 50x noise on every channel
    info = mne.create_info(CH, SFREQ, "eeg")
    events = np.c_[np.arange(N_EPO) * N_T, np.zeros(N_EPO, int), np.ones(N_EPO, int)]
    ep = mne.EpochsArray(data, info, events=events, tmin=0.0)
    ep.set_montage("standard_1020")
    return ep


def report(label, cleaned, dic):
    ev = [e.events[:, 0] for e in cleaned]
    print(f"[{label}] lengths={[len(e) for e in cleaned]} dic={dic}")
    print(f"  S1 kept event samples / N_T: {(ev[0] // N_T).tolist()}")
    print(f"  S2 kept event samples / N_T: {(ev[1] // N_T).tolist()}")
    if len(cleaned) > 2:
        print(f"  S3 kept event samples / N_T: {(ev[2] // N_T).tolist()}")
    aligned = all(np.array_equal(ev[0], e) for e in ev[1:])
    print(f"  ALIGNED (identical kept-epoch sets): {aligned}")


for strategy in ("union", "intersection"):
    print(f"\n===== case A: bad epoch 5 in S1 only, strategy={strategy}")
    s1, s2 = make_epochs(0, [5]), make_epochs(1, [])
    cleaned, dic = prep.AR_local([s1, s2], strategy=strategy, threshold=50.0)
    report(strategy, cleaned, dic)

for strategy in ("union", "intersection"):
    print(f"\n===== case B: bad 5 in S1 only, bad 9 in S2 only, strategy={strategy}")
    s1, s2 = make_epochs(0, [5]), make_epochs(1, [9])
    cleaned, dic = prep.AR_local([s1, s2], strategy=strategy, threshold=50.0)
    report(strategy, cleaned, dic)

print("\n===== case C: three participants, bad 7 in S3 only, strategy=union")
s1, s2, s3 = make_epochs(0, []), make_epochs(1, []), make_epochs(2, [7])
cleaned, dic = prep.AR_local([s1, s2, s3], strategy="union", threshold=50.0)
report("union-3p", cleaned, dic)

print("\n===== case D: input mutation check")
s1, s2 = make_epochs(0, [5]), make_epochs(1, [])
n_before = (len(s1), len(s2))
d_before = s1.get_data(copy=True).copy()
prep.AR_local([s1, s2], strategy="union", threshold=50.0)
print(f"  input lengths before/after: {n_before} -> {(len(s1), len(s2))}")
print(f"  input data unchanged: {np.array_equal(d_before, s1.get_data(copy=True))}")

print("\n===== case E: verbose=True plotting path")
s1, s2 = make_epochs(0, [5]), make_epochs(1, [])
try:
    prep.AR_local([s1, s2], strategy="union", threshold=50.0, verbose=True)
    print("  verbose=True OK")
except Exception as e:  # noqa: BLE001 - diagnostic script
    print(f"  verbose=True raised {type(e).__name__}: {e}")
