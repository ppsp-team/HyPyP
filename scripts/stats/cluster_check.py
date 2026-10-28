"""statscondCluster / statscluster checks with synthetic data."""

import warnings

import numpy as np
import scipy.sparse
import scipy.stats
from mne.stats import f_oneway, permutation_cluster_1samp_test, permutation_cluster_test

from hypyp import stats

warnings.simplefilter("ignore")
np.set_printoptions(linewidth=200)

n_feat = 12
adj = scipy.sparse.csr_matrix(np.eye(n_feat) + np.eye(n_feat, k=1) + np.eye(n_feat, k=-1))


def groups(n, lift, feats, seed, n_groups=2):
    rng = np.random.default_rng(seed)
    gs = [rng.standard_normal((n, n_feat)) for _ in range(n_groups)]
    gs[-1][:, feats] += lift
    return gs


print("=== A. f oneway with 3 groups: stat_fun only uses the first two groups")
g = groups(15, 2.0, [3, 4, 5], 0, n_groups=3)  # only group 3 differs
r = stats.statscluster(g, "f oneway", None, adj, 0, 500, 0.05)
print("hypyp 'f oneway' : clusters", len(r.clusters), "p", np.round(r.cluster_p_values, 3),
      "| Stat_obs[3:6]", np.round(r.Stat_obs[3:6], 2))
print("scipy f_oneway on all 3 groups, features 3:6:",
      np.round(scipy.stats.f_oneway(*g)[0][3:6], 2))
F, cl, pv, H0 = permutation_cluster_test(g, adjacency=adj, n_permutations=500, tail=1, seed=0,
                                         verbose=False, out_type="mask")
print("MNE default f_oneway  : clusters", len(cl), "p", np.round(pv, 3))

print("\n=== B. negative t cluster vanishes from Stat_obs_plot (np.maximum with zeros)")
g = groups(20, -3.0, [3, 4, 5], 1)
r = stats.statscluster(g, "ind ttest", None, adj, 0, 500, 0.05)
print("cluster p:", np.round(r.cluster_p_values, 3), "Stat_obs[3:6]:", np.round(r.Stat_obs[3:6], 2))
print("Stat_obs_plot:", np.round(r.Stat_obs_plot, 2), " nonzero:", int(np.count_nonzero(r.Stat_obs_plot)))

print("\n=== C. two significant clusters with identical p -> second one dropped from F_obs_plot")
g = groups(20, 6.0, [1, 2, 8, 9], 2)
r = stats.statscondCluster(g, [10], adj, 1, 500, 0.05)
print("cluster p:", np.round(r.cluster_p_values, 4), "cluster sizes:", [int(c.sum()) for c in r.clusters])
print("F_obs_plot nonzero idx:", np.flatnonzero(r.F_obs_plot), " (expected [1 2 8 9])")

print("\n=== D. tail=-1 with t-tests: threshold is positive -> MNE raises")
g = groups(20, -3.0, [3, 4, 5], 3)
for test in ("ind ttest", "rel ttest"):
    try:
        stats.statscluster(g, test, None, adj, -1, 100, 0.05)
        print(test, ": ran")
    except Exception as e:
        print(test, ": ERROR", type(e).__name__, e)

print("\n=== E. rel ttest null destroys pairing (known); quick confirmation, 10 datasets")
hits_hypyp = hits_1samp = 0
for seed in range(10):
    rng = np.random.default_rng(100 + seed)
    subj = rng.standard_normal((20, 1)) * 3.0  # big between-subject variance
    a = subj + rng.standard_normal((20, n_feat)) * 0.5
    b = a.copy()
    b[:, 3:6] += 0.6
    r = stats.statscluster([b, a], "rel ttest", None, adj, 0, 500, 0.05)
    hits_hypyp += int(np.any(r.cluster_p_values <= 0.05))
    T, cl, pv, H0 = permutation_cluster_1samp_test(b - a, adjacency=adj, n_permutations=500,
                                                   tail=0, seed=seed, verbose=False, out_type="mask")
    hits_1samp += int(np.any(pv <= 0.05))
print(f"detections hypyp rel ttest: {hits_hypyp}/10 ; sign-flip 1samp on differences: {hits_1samp}/10")

print("\n=== F. f multipleway")
rng = np.random.default_rng(5)
X = rng.standard_normal((4, 12, n_feat))  # 2x2 design, 12 subjects
try:
    stats.statscluster(X, "f multipleway", [2, 2], adj, 1, 100, 0.05)
    print("2x2: ran")
except Exception as e:
    print("2x2: ERROR", type(e).__name__, str(e)[:120])
try:
    r = stats.statscluster(X[:3], "f multipleway", [3], adj, 1, 100, 0.05)
    print("single factor [3]: ran; Stat_obs shape", r.Stat_obs.shape)
except Exception as e:
    print("single factor [3]: ERROR", type(e).__name__, str(e)[:120])
try:
    stats.statscluster(list(X), "f multipleway", [2, 2], adj, 1, 100, 0.05)
except Exception as e:
    print("list input: ERROR", type(e).__name__, str(e)[:120])

print("\n=== G. ind ttest: Welch statistic but pooled df for the threshold")
g = groups(8, 0.0, [], 6)
g[1] = np.random.default_rng(7).standard_normal((30, n_feat)) * 4
# welch df per feature
v1, v2 = g[0].var(0, ddof=1), g[1].var(0, ddof=1)
n1, n2 = 8, 30
df_w = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
print("pooled df used:", n1 + n2 - 2, "-> t threshold", round(scipy.stats.t.ppf(0.975, n1 + n2 - 2), 3))
print("Welch df range:", np.round(df_w.min(), 1), "-", np.round(df_w.max(), 1),
      "-> t threshold", round(scipy.stats.t.ppf(0.975, df_w.min()), 3), "-", round(scipy.stats.t.ppf(0.975, df_w.max()), 3))

print("\n=== H. statscondCluster tail handling: tail=0 vs tail=1 give identical results for F")
g = groups(20, 2.0, [3, 4, 5], 8)
r0 = stats.statscondCluster(g, [10], adj, 0, 300, 0.05)
r1 = stats.statscondCluster(g, [10], adj, 1, 300, 0.05)
print("tail=0 p:", np.round(r0.cluster_p_values, 3), " tail=1 p:", np.round(r1.cluster_p_values, 3))
print("threshold used tail=0: F.ppf(0.95,1,38)=", round(scipy.stats.f.ppf(0.95, 1, 38), 3),
      "| tail!=0: MNE default F.ppf(0.95) too (alpha ignored) ->", "same")
r2 = stats.statscondCluster(g, [10], adj, 1, 300, 0.001)
print("alpha=0.001, tail=1: threshold unchanged? H0 identical to alpha=0.05 run:",
      np.allclose(np.sort(r1.H0), np.sort(r2.H0)))
rm1 = stats.statscondCluster(g, [10], adj, -1, 300, 0.05)
print("tail=-1 ran, p:", np.round(rm1.cluster_p_values, 3))

print("\n=== I. H0 length and n_permutations")
print("len(H0):", len(r1.H0), "n_permutations=300")

print("\n=== J. full-matrix flattening double counts pairs (docstring line 604 advice)")
n = 5
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
print("upper-triangle features:", len(pairs), "| full-matrix features:", n * n, "| diagonal included:", True)

print("\n=== K. dense adjacency from metaconn_matrix straight into statscluster")
A = scipy.sparse.csr_matrix(np.eye(4) + np.eye(4, k=1) + np.eye(4, k=-1))
mc = stats.metaconn_matrix([(0, 1), (1, 2), (2, 3)], A, [10], plot=False)
try:
    stats.statscluster(groups(10, 0, [], 9)[:2], "ind ttest", None, mc.metaconn_freq[:3, :3], 0, 50)
except Exception as e:
    print("ERROR", type(e).__name__, e)
