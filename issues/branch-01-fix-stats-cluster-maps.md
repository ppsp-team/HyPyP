# fix(stats): cluster maps, f oneway groups, tail and alpha

**Fix branch:** [`fix/stats-cluster-maps`](https://github.com/ppsp-team/HyPyP/tree/fix/stats-cluster-maps), two commits: `d9505a6` (the fix with its tests) and `38510a0` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/master...fix/stats-cluster-maps. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

Four repairs to the two cluster-permutation wrappers in `hypyp/stats.py`, found by the 2026-10-13 review (F2, F3, F4 and the `tail`/`alpha` items), plus a `seed` argument so a run can be reproduced.

**hypyp/stats.py**

- `statscluster(test="f oneway")` passed only the first two groups to `scipy.stats.f_oneway` while the degrees of freedom and the permutations used every group; it now computes the one-way F on all of them. With three groups where only the third differs, the old statistic on the planted features was 0.27, 0.02, 3.14 (cluster p = 0.49) where scipy on all three gives 15.5, 26.6, 14.0 and MNE's own one-way F finds the cluster at p = 0.002
- `F_obs_plot` and `Stat_obs_plot` looked each significant cluster up by its p-value with `np.where(cluster_p_values == p)[0][0]`, which always resolves to the first cluster sharing that p-value, and built the map with `np.maximum(zeros, ...)`, which discards a negative t cluster. Both arrays are now built by walking `zip(clusters, cluster_p_values)` and copying the statistic, with its sign, on every significant cluster (`_significant_values`)
- `tail=-1` raised `ValueError: incompatible tail and threshold signs` for the t-tests because the cluster-forming threshold was always positive; `_t_threshold` negates it for a lower-tail test. For the F-tests, `tail=-1` is rejected with a message, since the statistic is non-negative
- `statscondCluster` set the F threshold from `alpha` only when `tail == 0` and let MNE use its fixed p = 0.05 threshold otherwise; the threshold is now computed from `alpha` for every tail
- `seed` on both functions, forwarded to `permutation_cluster_test`

Docstrings of both functions say what `alpha` does (cluster-forming threshold and cluster-level selection) and which tails each statistic accepts.

**Changes numerical results: yes.** `statscluster(test="f oneway")` with more than two groups; `F_obs_plot` and `Stat_obs_plot` whenever several significant clusters tie on p or a cluster has a negative statistic; `statscondCluster` with `tail=1` and `alpha != 0.05`. `clusters`, `cluster_p_values` and `H0` are unchanged for every call that ran before, apart from the alpha case, and `tail=-1` calls that used to raise now return a result.

## Measured

- `tests/test_stats_cluster.py` (new, 9 tests, synthetic 12-feature chain, seeded): written before the fix. On `master` all nine fail (eight on the missing `seed` argument; with `seed` removed, the statistic-vs-scipy, negative-cluster, tied-cluster, `tail=-1` and `alpha` tests fail on the substance). After the fix: 9 passed in 1.3 s
- The review's reproduction `scripts/stats/cluster_check.py` before and after the fix is in `evidence/stats-cluster-maps.{before,after}.txt`. Before: `f oneway` 1 cluster p = 0.49; `F_obs_plot` non-zero at [1 2] where [1 2 8 9] is expected; `tail=-1` raises for both t-tests. After: see the file
- Full suite on this branch: 367 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 58 s, Python 3.13.7, Linux x86-64, mne 1.11.0, scipy 1.17.1
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays computed on the sample data by `scripts/compare/outputs.py`, every stochastic call seeded): 51 bit-identical. Changed: `statscluster.f_oneway.Stat_obs`, `.Stat_obs_plot`, `.cluster_p_values` (three groups: the statistic now uses all of them and one more cluster is found) and `statscluster.ind_ttest.Stat_obs_plot` (the negative cluster is now in the map). `statscondCluster` with `tail=1` at `alpha=0.05`, `rel ttest`, and everything outside `stats` are bit-identical

## Left out on purpose

- `rel ttest` still builds its null with `permutation_cluster_test`, which permutes observations between the two groups and so breaks the pairing (review F5); that is the subject of `fix/stats-paired-test`, which builds on this branch
- `f multipleway` is untouched here; it cannot run with two factors and runs a between-subject null for a repeated-measures F (F7), same follow-up branch
- The frequency-major adjacency built by `con_matrix` and the two `metaconn_*` functions (F1) is the subject of `fix/stats-adjacency-layout`, which builds on this branch
- The Welch statistic in `ind ttest` is thresholded with the pooled degrees of freedom; the difference is in the third decimal of the threshold and is left as is

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
