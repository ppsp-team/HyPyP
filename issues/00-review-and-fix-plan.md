# Review of 0.6.2: twenty silent-result defects, eleven fix branches

*Umbrella issue. The full review (6,500 words, every finding with file:line and a reproduction script) is [`REVIEW-2026-10-13.md` on the branch `review/2026-10-13`](https://github.com/ppsp-team/HyPyP/blob/review/2026-10-13/REVIEW-2026-10-13.md), next to the evidence files, the reproduction scripts and the output-comparison harness; this issue carries the summary, one issue per ready fix branch, the deferred items and a guide to reviewing the branches.*

## In one paragraph

A full read of 0.6.2 on 2026-10-13, eight areas in parallel, with the rule that a claim about a wrong value had to be reproduced on synthetic data before it was reported. The core holds: all nine `hypyp.sync` metrics reproduce independent references to ten digits, the inter-brain block is sliced from the right quadrant, the viz geometry is a proper rotation with the right index mapping, and XDF clock synchronisation is correctly delegated to pyxdf. Around that core, twenty defects change results silently, and the test suite is green through all of them because only 20 of its 400 tests (5 %) assert a value derived independently of HyPyP.

## The findings that change results (F-numbers from the review)

| | Where | What |
| --- | --- | --- |
| F1 | `stats.con_matrix`, `metaconn_*` | across-frequency adjacency is frequency-major, the data MNE flattens is channel-major: wrong clusters whenever more than one frequency is used (the tutorial does) |
| F2, F3, F4 | `stats.statscluster`, `statscondCluster` | `f oneway` ignores groups beyond two; the significant-cluster maps drop tied and negative clusters |
| F5, F7 | `stats.statscluster` | `rel ttest` builds its null between groups, breaking the pairing; `f multipleway` cannot run with two factors and runs a between-subject null for one |
| F6 | `stats.statsCond` | FDR applied on top of max-T corrected p-values; raw power is "significant" everywhere |
| F8 | `analyses.compute_nmPLV` | the n:m exponent lands on the time axis, the average on the frequency axis (#310) |
| F9 | `analyses.compute_conn_mvar` | the epoch merge reshapes without moving the epoch axis |
| F10 | `analyses.xwt` | smoothing window ±1σ and no scale smoothing: null WTC 0.66 instead of 0.33 |
| F11 | `prep.AR_local` | `strategy="intersection"` drops different epochs per participant: inter-brain measures on non-simultaneous data |
| F12 | `utils.generate_virtual_epoch` | solver tolerances never lock the coupled oscillators (coupled PLV 0.24, uncoupled 0.66) |
| F13, F14, F15 | `wavelet`, `fnirs` | CWT cache keyed on `his_id` collides on the bundled files (inter rows become intra); pywt kernel degrades at large scales; `dt` off by n/(n−1) |
| F16 to F19 | `xdf` | `Markers.__add__` duplicates its own timestamps; Raw time axis ignores gaps and drift; no cross-stream alignment; `scale="auto"` multiplies by 1e-5 |
| F20 | `viz` | an unsliced (2n, 2n) matrix is drawn as if it were the inter-brain block |

## Fix branches, one issue each (adopt in this order)

Each branch is pushed to this repository and linked to its issue; the issue carries the defect, the evidence, and the branch's two commits (the fix with its tests, then the changelog entry). A branch is adopted by opening a pull request from its compare link (one click, the commits are kept) or by cherry-picking the two commits; nothing is merged by anyone but the maintainer. Branches 1 to 6 and 11 start from `master`; 7, 8 and 9 are stacked on 1; 10 is stacked on 6. Every one carries the regression test that reproduces its finding, seen failing before the fix, and states whether it changes computed values.

- [ ] 1. `fix/stats-cluster-maps`: F2, F3, F4, `tail=-1`, `alpha` for every tail, `seed`
- [ ] 2. `fix/xdf-markers-scale`: F16, F19, annotations attached, sorted onsets
- [ ] 3. `fix/analyses-nmplv`: F8 (#310)
- [ ] 4. `fix/mvarica-arity-merge`: F9, ridge arity, order-1 stability
- [ ] 5. `fix/prep-ar-local-alignment`: F11, any number of participants
- [ ] 6. `fix/utils-virtual-epoch`: F12, `rng`
- [ ] 7. `fix/stats-adjacency-layout` (on 1): F1, sparse return, channel count, layout check
- [ ] 8. `fix/stats-paired-test` (on 7): F5 by sign flipping, F7 removed
- [ ] 9. `fix/stats-statscond` (on 8): F6, contrast input
- [ ] 10. `test/reference-suite` (on 6): 37 reference tests, #308, seeded signals, lazy XDF fixtures
- [ ] 11. `docs/conventions`: formulas as computed (#293, #301, #302), averaging convention, tutorial layouts

(The issue numbers go here once they exist.)

## Deferred, with their own issues

- [ ] F13, F14, F15 and the fNIRS MAJOR items: for the fNIRS maintainer after #237
- [ ] F17, F18: regular Raw time grid from the dejittered timestamps, `aligned_raws()`
- [ ] F10 and `plot_xwt`
- [ ] F20 and the viz MAJOR items
- [ ] CI with an accelerated extra, wheel smoke test, unused dependencies, version floors, hygiene, HypypData licence
- [ ] the nine method decisions (conventions kept as they are and documented for now)

## How to review the branches

**Decisions already taken** (by Guillaume, 2026-10-13): no convention changes in this round, every computed value stays as it is unless a bug made it wrong, and those changes are declared in the changelog with a **Changes results** clause; fNIRS and Shiny wait for #237.

**Five minutes per issue.** Each issue body has the same four sections.

1. *What this does*: the defect in one paragraph with the file:line, what the code now does.
2. *Measured*: the new test file, with the output of running it on the base branch (the failure) and after the fix; the review's reproduction script before and after; the full suite count on the branch; and the **output comparison**, a fixed set of 50 arrays computed on the sample data on `master` and on the branch (`HyPyP-review/scripts/compare/`), listing every array that changed. If the "Changes numerical results" statement and the comparison table disagree, that is the thing to look at.
3. *Left out on purpose*: what the same files still have wrong and why it is not in this diff.
4. The changelog entry, in the second commit.

**What needs your judgement, not just a read** (about an hour of the two and a half):

- PR 1: the F threshold now comes from `alpha` for every tail, and `tail=-1` is rejected for F tests. Is that the behaviour you want, and is the `seed` name right?
- PR 7: the adjacency follows MNE's `combine_adjacency` convention (no links between spatial neighbours at *neighbouring* frequencies). The old expansion had them. The `(n_observations, n_channels, n_frequencies)` layout is now enforced with a `ValueError`; tutorial cell 38 needed rewriting (PR 11).
- PR 8: `rel ttest` becomes `permutation_cluster_1samp_test` on the differences; `f multipleway` is removed rather than repaired. A within-subject permutation scheme for repeated-measures ANOVA would be a new feature.
- PR 9: the FDR step is removed rather than the max-T correction; `adj_p` keeps its tuple shape.

**Mechanical, verify and adopt:** branches 2, 3, 4, 5, 6, 10, 11.

**Suggested order and time:** 1 (20 min), 2 (10), 3 (10), 4 (10), 5 (15), 6 (10), 7 (25), 8 (15), 9 (10), 10 (15), 11 (15).

**Stacked branches:** the compare links of 7, 8, 9 and 10 are against their base branch, so the diff shows only their own commits; a pull request opened from such a link targets the base branch and GitHub retargets it to `master` when the base is merged. The only merge conflict to expect is `CHANGELOG.md`, where every branch adds under `## [Unreleased]`: keep both sides.

**Rerunning the evidence yourself:** `uv run pytest tests/` on each branch; `python HyPyP-review/scripts/compare/outputs.py out.npz` from a checkout of `master` and of the branch, then `compare.py master.npz branch.npz`.
