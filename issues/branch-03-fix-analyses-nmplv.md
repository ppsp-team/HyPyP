# fix(analyses): raise the channel axis, not time, to the n:m powers

**Fix branch:** [`fix/analyses-nmplv`](https://github.com/ppsp-team/HyPyP/tree/fix/analyses-nmplv), two commits: `fae5cd5` (the fix with its tests) and `6aa6254` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/master...fix/analyses-nmplv. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

Repairs `compute_nmPLV`, review finding F8 and issue #310.

**hypyp/analyses.py**

- The function lays its array out as `(n_epochs, n_freq, 2 * n_channels, n_times)` and then sliced the *last* axis (`phase[:, :, :, :n_ch]`), so the first `n_channels` time samples of every channel were raised to the power n and the rest to the power m; participant 1 and participant 2 were never told apart. The slices now address the channel axis: participant 1's phase to the power n, participant 2's to the power m
- The epoch average ran on the frequency axis, because the `swapaxes(0, 1)` that `compute_sync` does before averaging was missing; the result came back as `(n_epochs, 2 * n_channels, 2 * n_channels)` where the docstring promises `(n_freq, 2 * n_channels, 2 * n_channels)`. It is now the epoch average at every frequency
- A `.squeeze()` on the taper-averaged signal made a single epoch or a single channel raise `IndexError`; it is gone
- The deprecated `_multiply_conjugate` (a `DeprecationWarning` on every call) is replaced by `hypyp.sync.base.multiply_conjugate`

The docstring now says that both participants are taken at the same frequency bins, from `min(freq_range1)` to `max(freq_range2)`, and that the locking is computed within each epoch then averaged. Whether the two participants should instead be compared at f1 and f2 (Palva et al. 2005) is a method decision left to the maintainers; the review lists it.

**Changes numerical results: yes, for every call.** The shape changes from `(n_epochs, ...)` to `(n_freq, ...)` and every value changes.

## Measured

- `tests/test_nmplv.py` (new, 4 tests): one rebuilds the n:m PLV from its definition on the same multitaper phases (agreement to 1e-12, pins the axes), three use pure tones whose 1:2 locking is fixed by construction (locked pair above 0.95 at the two tone bins, pair one hertz off the ratio below 0.2 everywhere, single epoch and single channel accepted). On `master`: shape `(3, 4, 4)` instead of `(16, 4, 4)`, locked value 0.02 to 0.05, `IndexError` on one epoch (`evidence/analyses-nmplv.before.txt`). After the fix: 4 passed
- The existing `tests/test_stats.py::test_compute_nmPLV` (inequalities between blocks on the sample data) still passes with the new shape
- Full suite on this branch: 362 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 57 s, Python 3.13.7, Linux x86-64, mne 1.11.0
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `scripts/compare/`): 54 bit-identical. Changed: `compute_nmPLV` only, shape `(n_epochs, 2n, 2n)` → `(n_freq, 2n, 2n)`; every `compute_sync` and `pair_connectivity` array is identical

## Left out on purpose

- The same-bin design (both participants at bin f) versus Palva's φ_x(f1) against φ_y(f2): a method change, not a repair
- `compute_single_freq` excludes `fmax` from the bins (`np.arange(fmin, fmax)`) although its docstring says "from fmin to fmax"; that is a documentation item for the `docs/conventions` branch

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
