# docs: state the metric conventions as the code computes them

**Fix branch:** [`docs/conventions`](https://github.com/ppsp-team/HyPyP/tree/docs/conventions), two commits: `497e8fc` (the fix with its tests) and `da92af1` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/master...docs/conventions. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

The 2026-10-13 review found that the documentation states formulas the code does not compute, and that nine method choices are nowhere stated. Following the decision to keep every computed value as it is and fix the documentation, this branch writes the conventions down where users read them.

**hypyp/sync/README.md, hypyp/sync/{coh,imaginary_coh,ccorr,wpli,envelope_corr}.py, hypyp/analyses.py (`compute_sync`, `pair_connectivity`)**

- Coherence was written as the magnitude-squared coherence `|⟨S₁₂⟩|² / (⟨|S₁|²⟩⟨|S₂|²⟩)`; the code returns the modulus of the coherency `|⟨S₁₂⟩| / √(⟨|S₁|²⟩⟨|S₂|²⟩)`, as mne-connectivity's `coh` does (0.8361 where the magnitude squared is 0.6991 on the review's pair; `y = x + n` gives 1/√2, not 0.5). Imaginary coherence and circular correlation were written with their sign (Nolte et al. 2004, Jammalamadaka & SenGupta 2001); the code returns absolute values. The formulas now read as computed, with the consequences stated: the lead/lag sign is discarded, and a modulus has a positive bias under the null (#301, #302)
- wPLI is documented as the estimator of Vinck et al. 2011 eq. 8 without the debiasing term of eq. 11 (0.674 against 0.454 debiased on the same data); envelope correlation as not orthogonalised, so within-brain values include volume conduction although Hipp et al. 2012 is cited
- The averaging convention, stated once in the README and in both docstrings: every metric is computed within each epoch, then the per-epoch values are averaged, so the bias of independent signals depends on the epoch length and not on the amount of data (80 s of independent 8-12 Hz noise cut into 1 s, 4 s or 16 s epochs gives a PLV of 0.32, 0.15 or 0.08; Zimmermann et al. 2024). Studies and conditions must be compared at the same epoch length
- `compute_single_freq` and `pair_connectivity` said "every integer frequency from fmin to fmax"; `fmax` is excluded, `[8, 30]` gives 22 bins, not the 23 the example claimed. The multitaper bandwidth is stated: half-bandwidth of half the bin frequency, so neighbouring integer bins overlap strongly
- `compute_sync`'s `optimization` text said the option was only available for `accorr` and that `'auto'` was "torch > numba > numpy" (#293); it applies to every metric and `'auto'` picks per metric and machine, as the README's tables say

**tutorial/getting_started.ipynb** (cells 33, 35, 38, 39, 40)

- The statistics cells now say that they demonstrate the API on duplicated data (each "group" is two or three near-identical copies of one participant, so any difference is trivially significant), that `statsCond` on raw power with two participants as observations is such a demonstration, and what the two data layouts are
- The intra- and inter-brain cluster cells passed `(n_observations, 31, 31)` matrices with the 31 by 31 channel adjacency (or none), which MNE read as a time lattice over the first axis, or as a grid. They now pass one vector per observation in the order of a pair list, with the pair adjacency of `metaconn_matrix` and `metaconn_matrix_2brains` wrapped in `csr_matrix`, so they run on the current `master` and after the `fix/stats-adjacency-layout` branch alike

**Changes numerical results: no.** Documentation, docstrings and tutorial text; the tutorial's rewritten cells call public functions with their documented layouts.

## Measured

- `ruff format --check .` and `ruff check .` clean; `python -m mkdocs build --strict` passes (the docstrings feed mkdocstrings)
- The review's `scripts/infra/nb_drift.py` (every `hypyp.*` call of the nine notebooks resolved against the current signatures): no mismatch, including the rewritten cells
- Full suite on this branch: 358 passed, 42 skipped, the same as `master`, 2 min 50 s, Python 3.13.7, Linux x86-64
- Output comparison with `master` (55 arrays on the sample data, `scripts/compare/`): 55 bit-identical

## Left out on purpose

- Changing any value to match the literature (magnitude-squared coherence, signed ImCoh or CCorr, debiased wPLI, orthogonalised envelopes) is a method decision the maintainers have declined for this round; the review's section 5 lists them with their numerical consequences, and an issue draft collects them
- `statsCond`'s docstring is rewritten in `fix/stats-statscond`, not here, to avoid two branches editing the same lines
- README.md's "beta" status, badges and module list, and the changelog backfill of 0.5.0b14 to b18, are out of scope (repository hygiene, listed in the review)

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
