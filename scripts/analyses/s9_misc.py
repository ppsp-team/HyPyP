import warnings, numpy as np, mne
from hypyp import analyses
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")
rng = np.random.default_rng(9)
info = mne.create_info(["C1", "EOG", "C2", "C3"], 128, ["eeg", "eog", "eeg", "eeg"])
ep = mne.EpochsArray(rng.standard_normal((2, 4, 128)), info, verbose=False)
idx = analyses.indices_connectivity_intrabrain(ep)
print("intrabrain with EOG at index 1:", idx, "->", [(ep.ch_names[i], ep.ch_names[j]) for i, j in idx])
# compute_conn_mvar default check_stability=True under non-interactive stdin
cs = rng.standard_normal((2, 1, 1, 1, 400)) + 0j
try:
    analyses.compute_conn_mvar(cs, {"mvar_order": 2, "fitting_method": "default", "delta": 0},
                               {"method": "infomax_extended", "random_state": 0}, {"name": "pdc", "n_fft": 8})
except BaseException as exc:
    print("check_stability=True, stdin=/dev/null ->", type(exc).__name__, exc)
