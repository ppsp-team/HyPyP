"""Matrix-shaped data with a channel adjacency (tutorial usage), H0 length, dense adjacency."""

import warnings

import mne
import numpy as np
import scipy.sparse

from hypyp import stats

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

print("=== (n_obs, n_ch, n_ch) connectivity matrices + (n_ch x n_ch) ch_con, as in the tutorial")
# chain-free adjacency: 0~2, 1~3 only (plus self); 0 and 1 are NOT neighbours
A = np.eye(4)
A[0, 2] = A[2, 0] = A[1, 3] = A[3, 1] = 1
ch_con = scipy.sparse.csr_matrix(A)
rng = np.random.default_rng(0)
g0 = rng.standard_normal((20, 4, 4))
g1 = rng.standard_normal((20, 4, 4))
g1[:, 0, 0] += 4.0  # pair (0,0)
g1[:, 1, 0] += 4.0  # pair (1,0): shares channel 0 only through row index 0->1, which are NOT neighbours
r = stats.statscondCluster([g0, g1], [10], ch_con, 1, 200, 0.05)
print("clusters:", len(r.clusters), "members:", [list(map(tuple, np.argwhere(c))) for c in r.clusters],
      "p:", np.round(r.cluster_p_values, 3))
print("-> MNE treated the first channel axis as a 'time' lattice (row i ~ row i+1); ch 0 and 1 are not neighbours")

print("\n=== H0 length from statscondCluster")
print("len(H0):", len(r.H0), "for n_permutations=200")

print("\n=== dense metaconn_freq passed straight to statscluster")
mc = stats.metaconn_matrix([(0, 1), (1, 2), (2, 3)], ch_con, [10], plot=False)
print("metaconn_freq type:", type(mc.metaconn_freq).__name__)
try:
    stats.statscluster([rng.standard_normal((10, 3)), rng.standard_normal((10, 3))], "ind ttest", None,
                       mc.metaconn_freq, 0, 50)
except Exception as e:
    print("ERROR", type(e).__name__, e)

print("\n=== statsCond n_permutations semantics: exhaustive sign flips = 2**n")
print("docstring says 'at least 2*n_samples'; MNE uses all 2**n flips when n_permutations >= 2**n")
