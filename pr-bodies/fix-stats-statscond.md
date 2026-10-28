# fix(stats): return statsCond's max-T p-values without a second correction

Branch: `fix/stats-statscond`, **built on `fix/stats-paired-test`** (chain: cluster-maps, adjacency-layout, paired-test, this one; merge in that order). Open with:
`gh pr create --draft --base fix/stats-paired-test --head fix/stats-statscond --title "fix(stats): return statsCond's max-T p-values without a second correction" --body-file pr-bodies/fix-stats-statscond.md`

(Opened against its base branch so that the diff shows only this branch's commits; GitHub retargets it once the base is merged.)

## What this does

Repairs review finding F6 and the two items next to it (positive input, dead `pos` block).

**hypyp/stats.py**

- `statsCond` ran `mne.stats.permutation_t_test`, whose p-values are already corrected for the number of channels by the max-T method ("the tmax method is used for adjusting the p-values ... controls the family-wise error rate", MNE docstring), and then applied `fdr_correction` to them: corrected twice, doubly conservative, while the docstring described a single FDR correction. On the review's example the returned `p_values` equalled an independent max-T computation (`[6e-4, 1, 0.527, 0.737]`) and `adj_p[1]` was the FDR of those (`[0.0024, 1, 0.983, 0.983]`). The max-T p-values are now returned as they are; `adj_p` is `(p_values <= alpha, p_values)`, the tuple shape callers read; `T_obs_plot` follows
- The docstring says what the test is (a one-sample sign-flip test of the mean against zero, family-wise error controlled by max-T) and that the input must be a contrast: on raw power, positive everywhere, every channel comes out "significant" (t of 32 to 36 on the sample data), which is what `tutorial/getting_started.ipynb` cell 34 does. An input whose values all share one sign now warns
- The block that built `pos` from `epochs` and never used it is gone; `epochs` is documented as unused and kept so that calls keep working. The Examples section, a copy of `statscluster`'s, shows a contrast. `seed` is forwarded to MNE and `n_permutations="all"` runs the exact test

**Changes numerical results: yes, for `adj_p` and `T_obs_plot`**: `adj_p[1]` was the FDR of the max-T p-values and is now the max-T p-values; `adj_p[0]` and `T_obs_plot` are less conservative accordingly. `T_obs`, `p_values` and `H0` are unchanged.

## Measured

- `tests/test_statscond.py` (new, 4 tests): the sign flips of ten participants enumerated by hand with MNE's exact convention match the returned `p_values` to machine precision and `T_obs` matches `scipy.stats.ttest_1samp`; no second correction is applied; a same-sign input warns; a seed reproduces the run. On the base branch: the no-second-correction and warning tests fail, the seed argument is missing (`pr-bodies/evidence/stats-statscond.before.txt`); the exact p-value test passes before and after, which is the finding that `p_values` themselves were right. After the fix: 4 passed, plus the 16 of `tests/test_stats.py` (its `p_values[i] <= adj_p[1][i]` assertion holds with equality)
- Full suite on this branch: 388 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 45 s, Python 3.13.7, Linux x86-64, mne 1.11.0
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `HyPyP-review/scripts/compare/`): 44 bit-identical. Changed beyond the nine of its base branches: `statsCond.adj_p1` (the max-T p-values themselves instead of their FDR) and `statsCond.T_obs_plot`; `statsCond.T_obs` and `statsCond.p_values` are bit-identical, as declared

## Left out on purpose

- Whether the tutorial's `statsCond(np.array([psd1, psd2]), ...)` call (two participants as the observations, raw power as the data) should be rewritten around a contrast is a tutorial change for the conventions pull request
- `n_permutations` with `n_samples > 20` still uses MNE's hash-table draw; nothing changed there

🤖 Generated with [Claude Code](https://claude.com/claude-code)
