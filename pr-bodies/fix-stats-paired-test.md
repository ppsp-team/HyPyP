# fix(stats): build the paired cluster test's null by sign flipping

Branch: `fix/stats-paired-test`, **built on `fix/stats-adjacency-layout`** (which is built on `fix/stats-cluster-maps`; merge in that order). Open with:
`gh pr create --draft --base fix/stats-adjacency-layout --head fix/stats-paired-test --title "fix(stats): build the paired cluster test's null by sign flipping" --body-file pr-bodies/fix-stats-paired-test.md`

(Opened against its base branch so that the diff shows only this branch's commits; GitHub retargets it once the base is merged.)

## What this does

Repairs review findings F5 and F7.

**hypyp/stats.py**

- `statscluster(test="rel ttest")` computed `scipy.stats.ttest_rel` but built its null with `permutation_cluster_test`, which concatenates the two groups and permutes rows (`cluster_level.py:1042-1046`): after a permutation, observation k of one group and observation k of the other are different pairs, so the null described var(A) + var(B) instead of var(A − B). When the pairs share a large component across features, as every EEG participant does through its topography, the permuted differences move together across all features and the null's cluster masses have nothing to do with the observed ones. On twenty pairs with a per-pair offset of 3 and noise of 0.5, the old test gave p = 0.03 and 0.046 where the sign-flip null on the same data and seed gives 0.002 and 0.278: wrong in both directions. The size of the error depends on how much the shared component dominates; it reached a total loss of power on EEG-like planted data (UCLA-R21, 2026-10-03) and was small in a regime where the within-pair noise dominates
- The paired test now runs `mne.stats.permutation_cluster_1samp_test` on the within-pair differences, whose null flips the sign of each difference, the exchangeability a paired design has, with the same cluster-forming threshold, adjacency, `t_power` and `seed`; the returned tuple is unchanged. Groups of unequal size are rejected with a `ValueError`
- `test="f multipleway"` is removed. With two factors it never ran (`f_threshold_mway_rm(effects="all")` returns one threshold per effect and MNE rejects the list: `TypeError: float() argument must be ... not 'list'`), a list input hit `AttributeError`, and with one factor it permuted observations between subjects under a repeated-measures F, the same defect as above. The option raises a `ValueError` that says so; `factor_levels` stays in the signature, documented as ignored, so that existing calls keep working

**Changes numerical results: yes, for every `rel ttest` call** (`Stat_obs` is the same paired t; `clusters`, `cluster_p_values` and `H0` change). `ind ttest` and `f oneway` return exactly what they returned on the base branch.

## Measured

- `tests/test_stats_paired.py` (new, 6 tests): for three seeds the result equals MNE's sign-flip test on the differences, statistic for statistic, p-value for p-value, `H0` for `H0`, cluster for cluster; the statistic equals `scipy.stats.ttest_rel`; a lift of 0.5 under a shared topography of 3 is found as one cluster on the planted features; unequal groups and `f multipleway` raise. On the base branch: the three equivalence tests fail with the p-values quoted above (`pr-bodies/evidence/stats-paired-test.before.txt`). After the fix: 6 passed, plus the 36 of the other stats test files
- Full suite on this branch: 384 passed, 42 skipped (no numba, torch, Metal or CuPy installed), Python 3.13.7, Linux x86-64, mne 1.11.0
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `HyPyP-review/scripts/compare/`): 46 bit-identical. Changed beyond the six of its base branches: `statscluster.rel_ttest.cluster_p_values` (by 0.05 on twenty pairs sharing a per-pair offset of 3: the planted cluster becomes significant), `.Stat_obs_plot` (that cluster now in the map), `.Stat_obs` (4e-16: MNE's one-sample t against scipy's paired t). `ind ttest` and `f oneway` are identical to the base branch

## Downstream note

UCLA-R21's `tests/test_contrast.py:164` asserts that HyPyP's `rel ttest` detects *less* than its own sign-flip implementation (`flipped > broken`); after this change the two are the same test and that assertion no longer holds. UCLA-R21 pins HyPyP 0.6.0, so nothing changes until it upgrades.

## Left out on purpose

- `statsCond`'s double correction (F6) is the subject of `fix/stats-statscond`, built on this branch
- A within-subject permutation scheme for a repeated-measures ANOVA (what `f multipleway` was meant to be) is a new feature, not a repair; a mixed two-by-two decomposes into three cluster-testable contrasts, which the existing tests cover

🤖 Generated with [Claude Code](https://claude.com/claude-code)
