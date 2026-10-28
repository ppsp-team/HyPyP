"""statsCond checks: double correction, positive-only data, unused epochs."""

import mne
import numpy as np
from mne.stats import fdr_correction, permutation_t_test

from hypyp import stats

names = ["A", "B", "C", "D"]
pos = {"A": [0, 0, 0], "B": [1, 0, 0], "C": [0, 1, 0], "D": [2.5, 2.5, 0]}
info = mne.create_info(names, 100.0, "eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"))
epochs = mne.EpochsArray(np.zeros((2, 4, 10)), info, verbose=False)

rng = np.random.default_rng(0)
n_subj, n_ch, n_freq = 16, 4, 3
data = rng.standard_normal((n_subj, n_ch, n_freq))
data[:, 0, :] += 0.9  # one channel with a true (moderate) effect

print("=== 1. p_values returned by statsCond are already max-T (FWER) corrected, then FDR'd again")
out = stats.statsCond(data, epochs, 5000, 0.05)
power = data.mean(axis=2)
# independent per-variable sign-flip p (no multiple-comparison adjustment)
t_obs = power.mean(0) / (power.std(0, ddof=1) / np.sqrt(n_subj))
signs = rng.choice([-1, 1], size=(5000, n_subj))
t_perm = np.array([(s[:, None] * power).mean(0) / ((s[:, None] * power).std(0, ddof=1) / np.sqrt(n_subj)) for s in signs])
p_uncorr = (np.abs(t_perm) >= np.abs(t_obs)).mean(0)
p_maxT = (np.abs(t_perm).max(1)[:, None] >= np.abs(t_obs)[None, :]).mean(0)
print("T_obs            :", np.round(out.T_obs, 2))
print("own uncorrected p:", np.round(p_uncorr, 4))
print("own max-T p      :", np.round(p_maxT, 4))
print("statsCond p_values:", np.round(out.p_values, 4), "  <- matches max-T, not uncorrected")
print("statsCond adj_p[1] (FDR on top):", np.round(out.adj_p[1], 4))
print("MNE docstring:", [l.strip() for l in permutation_t_test.__doc__.splitlines() if "tmax" in l or "family" in l.lower()])

print("\n=== 2. raw PSD (strictly positive) makes the one-sample test trivially significant everywhere")
psd = rng.uniform(1.0, 2.0, size=(n_subj, n_ch, n_freq))
out = stats.statsCond(psd, epochs, 2000, 0.05)
print("T_obs:", np.round(out.T_obs, 1), " adj_p[0] (all 'significant'):", out.adj_p[0])

print("\n=== 3. epochs argument: only used for a dead 'pos' computation")
import inspect
src = inspect.getsource(stats.statsCond)
print("'pos' referenced after computation?", src.count("pos") , "occurrences;", "returned?", "pos" in src.split("return")[-1])

print("\n=== 4. H0 length vs docstring (n_permutations)")
print("len(H0) =", len(out.H0), "for n_permutations=2000")
