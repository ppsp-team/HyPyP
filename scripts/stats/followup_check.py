"""Follow-ups: negative-t plot, metaconn diagonal, freq-order mismatch, extra diagonal links."""

import warnings

import mne
import numpy as np
import scipy.sparse
from mne.stats import combine_adjacency, permutation_cluster_test

from hypyp import stats

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
np.set_printoptions(linewidth=200)

n_feat = 12
adj = scipy.sparse.csr_matrix(np.eye(n_feat) + np.eye(n_feat, k=1) + np.eye(n_feat, k=-1))

print("=== B'. negative t cluster (group 2 larger) vanishes from Stat_obs_plot")
rng = np.random.default_rng(1)
g0 = rng.standard_normal((20, n_feat))
g1 = rng.standard_normal((20, n_feat))
g1[:, 3:6] += 3.0
r = stats.statscluster([g0, g1], "ind ttest", None, adj, 0, 500, 0.05)
print("cluster p:", np.round(r.cluster_p_values, 3), "Stat_obs[3:6]:", np.round(r.Stat_obs[3:6], 2))
print("Stat_obs_plot nonzero count:", int(np.count_nonzero(r.Stat_obs_plot)), "(expected 3)")

print("\n=== metaconn diagonal debug")
names = ["A", "B", "C", "D"]
pos = {"A": [0, 0, 0], "B": [1, 0, 0], "C": [0, 1, 0], "D": [2.5, 2.5, 0]}
info = mne.create_info(names, 100.0, "eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"))
epochs = mne.EpochsArray(np.zeros((2, 4, 10)), info, verbose=False)
cm = stats.con_matrix(epochs, [10, 11, 12])
ch_con = cm.ch_con
print("ch_con[0,0], ch_con[3,3], ch_con[0,3]:", ch_con[0, 0], ch_con[3, 3], ch_con[0, 3], type(ch_con[0, 0]))
e = (0, 3)
print("rule1 ch_con[0,0] and ch_con[3,3] ->", (ch_con[0, 0]) and (ch_con[3, 3]))
mc = stats.metaconn_matrix([(0, 3)], ch_con, [10], plot=False)
print("metaconn for the single pair (A,D):", mc.metaconn)
mc = stats.metaconn_matrix([(0, 1)], ch_con, [10], plot=False)
print("metaconn for the single pair (A,B):", mc.metaconn)

print("\n=== ch_con_freq vs MNE combine_adjacency(n_freq, ch_con): extra links")
ref = combine_adjacency(3, ch_con).toarray().astype(int)
hy = cm.ch_con_freq.astype(int)
extra = np.argwhere((hy == 1) & (ref == 0))
print("n links hypyp:", hy.sum(), " n links combine_adjacency:", ref.sum(), " missing in hypyp:", int(((hy == 0) & (ref == 1)).sum()))
print("first extra links as (f,ch)-(f,ch):", [((i // 4, names[i % 4]), (j // 4, names[j % 4])) for i, j in extra[:6]])

print("\n=== feature-order mismatch, clean case: effect on channel A at f0 and f2 only (freqs 2 apart)")
# data layout (n_obs, n_ch, n_freq) as in tests/test_stats.py and the tutorial
n_ch, n_freq = 4, 3
rng = np.random.default_rng(3)
g0 = rng.standard_normal((20, n_ch, n_freq))
g1 = rng.standard_normal((20, n_ch, n_freq))
g1[:, 0, 0] += 4.0
g1[:, 0, 2] += 4.0
hy_adj = scipy.sparse.csr_matrix(cm.ch_con_freq)
ok_adj = combine_adjacency(ch_con, n_freq)  # (ch, freq) C-order
for label, A in [("hypyp ch_con_freq (freq-major)", hy_adj), ("combine_adjacency(ch_con, n_freq)", ok_adj)]:
    F, cl, pv, H0 = permutation_cluster_test([g0, g1], adjacency=A, n_permutations=300, tail=1,
                                             seed=0, out_type="mask", threshold=8.0)
    members = [[(names[c], f"f{f}") for c, f in np.argwhere(m)] for m in cl]
    print(f"{label}: {len(cl)} cluster(s) p={np.round(pv, 3)} members={members}")
# what the hypyp adjacency thinks the neighbours of feature (A, f1) [flat index 1] are
k = 1
nb = np.flatnonzero(cm.ch_con_freq[k])
print("data feature 1 = (A, f1); hypyp row 1 = (f0, B); neighbours it gets, decoded in data layout (ch, f):",
      [(names[j // n_freq], f"f{j % n_freq}") for j in nb])
