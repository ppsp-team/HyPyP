# Hand-over: eleven branches for the HyPyP 0.6.2 review findings

Everything lives in `~/Github/HyPyP` as local branches (nothing pushed, nothing merged, `master` untouched at `dbeeb37`; upstream `master` has not moved since 0.6.2 as of 2026-10-26). The work was done in the worktree `~/Github/HyPyP/.claude/worktrees/hypyp-fixes`, which has `docs/conventions` checked out; the branches are ordinary refs of the clone. The worktree can be removed once you no longer need it (`git worktree remove .claude/worktrees/hypyp-fixes`); the branches stay. The branch `worktree-hypyp-fixes` is an empty leftover and can be deleted.

Each branch has two commits: the change with its tests, then its changelog entry under `## [Unreleased]`. Commit messages follow the house style and end with a "Changes numerical results: yes/no" statement. Every PR body has a `## Measured` section with four kinds of evidence: the regression test seen failing before the fix, the review's reproduction script before and after, the full suite count on the branch, and the **output comparison** against `master` (below). Raw outputs are in `evidence/`.

## The branches, in merge order

| # | Branch | Base | Subject | Values change | Suite | Output comparison (55 arrays) | PR body |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fix/stats-cluster-maps` | master | F2 `f oneway` groups, F3/F4 cluster maps, `tail=-1`, `alpha` for every tail, `seed` | yes | 367 / 42 | 51 identical; 3-group F arrays, negative-cluster map | `fix-stats-cluster-maps.md` |
| 2 | `fix/xdf-markers-scale` | master | F16 marker merge, F19 `scale="auto"`, annotations attached, sorted onsets | yes | 363 / 42 | 53; annotations 0 → 12 on both Raws | `fix-xdf-markers-scale.md` |
| 3 | `fix/analyses-nmplv` | master | F8 n:m PLV axes (#310) | yes | 362 / 42 | 54; `compute_nmPLV` | `fix-analyses-nmplv.md` |
| 4 | `fix/mvarica-arity-merge` | master | F9 epoch merge, ridge arity, order-1 stability | merge path only | 363 / 42 | 55 identical | `fix-mvarica-arity-merge.md` |
| 5 | `fix/prep-ar-local-alignment` | master | F11 `AR_local` alignment, N participants | intersection only | 361 / 42 | 55 identical (no bad epoch in the sample data) | `fix-prep-ar-local-alignment.md` |
| 6 | `fix/utils-virtual-epoch` | master | F12 simulator integration, grid, `rng` | yes | 362 / 42 | 53; the two simulators | `fix-utils-virtual-epoch.md` |
| 7 | `fix/stats-adjacency-layout` | **1** | F1 adjacency layout, sparse return, `n` inference, layout check | yes (>1 frequency) | 378 / 42 | 49; + two multi-frequency adjacencies | `fix-stats-adjacency-layout.md` |
| 8 | `fix/stats-paired-test` | **7** | F5 paired null by sign flipping, F7 `f multipleway` removed | yes | 384 / 42 | 46; + paired p-values and map | `fix-stats-paired-test.md` |
| 9 | `fix/stats-statscond` | **8** | F6 double correction, contrast input, `seed` | `adj_p` only | 388 / 42 | 44; + `statsCond.adj_p1`, `T_obs_plot` | `fix-stats-statscond.md` |
| 10 | `test/reference-suite` | **6** | 37 reference tests, #308, seeded signals, lazy XDF fixtures | no | 400 / 42 | 52; `add_noise` + base's simulators | `test-reference-suite.md` |
| 11 | `docs/conventions` | master | conventions as computed (#293, #301, #302), averaging convention, tutorial layouts | no | 358 / 42 | 55 identical | `docs-conventions.md` |

"Suite" is `passed / skipped` of `uv run pytest tests/` on that branch alone (`master`: 358 / 42; the 42 skips are the GPU backends). Branches 1 to 6 and 11 are independent; 7 → 8 → 9 chain on 1; 10 chains on 6. Merging in this order never conflicts except in `CHANGELOG.md`, where every branch adds under `## [Unreleased]` (keep both sides).

## The output comparison

`scripts/compare/outputs.py` computes 55 arrays on the sample data in `tests/data` plus the cached XDF markers file: the nine `compute_sync` metrics, `pair_connectivity`, `pow`, the three adjacency builders at one and several frequencies, `statscondCluster`, the three `statscluster` tests, `statsCond`, `compute_nmPLV`, both simulators, `AR_local` under both strategies, the XDF import, `MVAR`, `SyntheticSignal`. Every stochastic call is preceded by `np.random.seed`, and only 0.6.2 call signatures are used, so the script runs unchanged on `master` and on every branch. `compare.py` reports each array as bit-identical or changed; `run_all.sh` and `compare_all.sh` do all twelve checkouts. The tables are in `evidence/output-comparison.md`, and every changed array is one its PR body declares. This is the same kind of check Rémy's #315 and #316 reported ("112 arrays bit-identical between master and this branch"), so he can rerun it rather than take the statement on trust.

## Hand-off as issues linked to branches (chosen 2026-10-28)

Rémy's own findings live as issues (#297 to #311) that he fixes in his own pull requests, so the fixes are handed over the same way: one issue per branch, the branch linked to it in GitHub's "Development" panel, nothing proposed for merge. `issues/branch-NN-*.md` are the issue bodies (generated from the PR bodies: the branch, its two commits and a compare link on top, then the same What / Measured / Left out sections). `issues/create_issues.sh` does the whole hand-off: for each branch it pushes the commits under a temporary ref, opens the issue, creates the branch on GitHub at the tip through `createLinkedBranch` (which is what links it to the issue), and drops the temporary ref; then the six deferred issues; then the umbrella with every number filled in. `DRY_RUN=1` prints the steps. The compare links of the stacked branches (7, 8, 9, 10) are against their base branch, so the diff shows only their own commits, and a pull request opened from such a link targets the base and is retargeted to `master` when the base is merged.

The pull-request route is still available: the PR bodies in this directory are unchanged, and each one's second line is its `gh pr create --draft` command.

## For Rémy

- `issues/00-review-and-fix-plan.md`: the umbrella issue, with the summary of the review, the checklist of the eleven PRs and the deferred issues, and a review guide (what each PR body contains, which four PRs need his judgement rather than a read, a suggested order with time estimates of about two and a half hours in all, how to rerun the evidence). Attach `REVIEW-2026-10-13.md` to it, or paste it in a gist, so that the file:line references are at hand
- `message-remy.fr.md`: a short message to send him with the links
- `issues/F13-F14-F15-fnirs-wavelet.md` is for Patrice, after #237; the other issue drafts are for Rémy's backlog

## Downstream

UCLA-R21 pins HyPyP 0.6.0 and is not exposed to any finding (one frequency block per band, own pair vector, own paired test, reads `clusters` not `F_obs_plot`). Its `tests/test_contrast.py:164` asserts that HyPyP's `rel ttest` loses to sign flipping; after branch 8 the two are the same test, so that assertion will need to go when UCLA-R21 upgrades.
