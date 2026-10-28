# test: reference tests for every metric, seeded signals, lazy XDF files

**Fix branch:** [`test/reference-suite`](https://github.com/ppsp-team/HyPyP/tree/test/reference-suite), two commits: `7038b53` (the fix with its tests) and `af0dbff` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/fix/utils-virtual-epoch...test/reference-suite. It is built on `fix/utils-virtual-epoch` (issue for branch 6), so adopt that one first; the compare link is against it, so only this branch's commits show. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

The 2026-10-13 review classified the 400 collected tests: 20 (5 %) assert a value derived independently of HyPyP, and seven of the nine `hypyp.sync` metrics had none. `AGENTS.md` asks contributors for exactly such tests. This branch adds them for the numerical core, fixes four test-suite defects the review found, and seeds every synthetic signal.

**tests/test_sync_reference.py** (new, 14 tests)

- Every metric against a scalar written from its definition with numpy and scipy alone (`np.angle`, `np.corrcoef`, `scipy.stats.circmean`, plain sums; none calls `hypyp.sync`): a random pair agrees to 1e-10 for all nine; a constant lag of a quarter turn gives PLV, CCorr, ACCorr, Coh, PLI and wPLI of 1 and ImCoh of sin(π/4); an envelope relation gives EnvCorr 1; identical channels give 1 on the amplitude and phase measures and 0 on ImCoh, PLI and wPLI; a pair of known coherence gives |C| = 1/√2, not the magnitude squared 0.5
- The conventions these pin are the ones the code has (coherence as |C|, ImCoh and CCorr as absolute values, wPLI without the debiasing term); the `docs/conventions` branch writes them into the docstrings

**tests/test_pair_connectivity_reference.py** (new, 23 tests)

- `compute_sync` end to end on narrowband 8 to 12 Hz signals through `compute_freq_bands(filter_signal=False)`: identical participants (1 on the amplitude and phase measures, 0 on the lag-sensitive ones), a quarter-turn shift (1 on PLV, Coh, ImCoh, PLI, wPLI), independent participants near chance, and one coupled pair that must land at `[i, j]` = (participant 1 channel i, participant 2 channel j) of the inter-brain block

**Suite repairs**

- `tests/test_sync.py` defined `class TestEnvCorr` twice; the second shadowed the first, whose five tests were never collected (#308). The dead copy is removed; the surviving class is a superset
- `tests/test_signal.py::add_noise` lacked the `test_` prefix and never ran; it is `test_add_noise` and also checks reproducibility
- `SyntheticSignal.add_noise` takes an `rng`, and the 35 synthetic signals of the wavelet, coherence and stats tests are seeded (`rng=0` for the first signal of a pair, `rng=1` for the second), so assertions such as "the first bin is more masked than the second" or `0 < coherence < 1` no longer depend on the draw. The two `generate_random_epoch` calls of `tests/test_stats.py` are seeded through the `rng` of `fix/utils-virtual-epoch`
- `tests/test_xdf.py` fetched its two sample files at import; they are module fixtures now, so `pytest` collects the suite offline (`tests/test_fnirs.py` still fetches at import: its helpers and a parametrize use the paths, and the fNIRS code is deferred to its own maintainer)
- `tests/data/README.md` states that the two `.fif` files are a cut of the public sample recordings

**Changes numerical results: no.** `SyntheticSignal.add_noise` draws from a `numpy.random.Generator` instead of the global `numpy.random`; nothing else under `hypyp/` changes.

## Measured

- New tests: 37 passed in 0.5 s (`test_sync_reference.py`, `test_pair_connectivity_reference.py`), written against the unmodified metrics and passing on the first run, which is the review's finding that the nine NumPy formulas are right
- Affected existing files after the changes: `test_xdf.py`, `test_signal.py`, `test_wavelet.py`, `test_wavelet_implementations.py`, `test_coherence_data_frame.py`, `test_sync.py`: 233 passed, 40 skipped; `test_stats.py`: 16 passed
- Full suite on this branch: 400 passed, 42 skipped (358 of master, 4 of the simulator branch, 37 new, and `test_add_noise`, which is now collected), 2 min 53 s, Python 3.13.7, Linux x86-64
- `ruff format --check .` and `ruff check .` clean
- Output comparison with `master` (55 arrays on the sample data, `scripts/compare/`): 52 bit-identical. Changed: `SyntheticSignal.add_noise` (numpy Generator instead of the global state, as declared) and the two simulators of the base branch. No analysis output changes

## Left out on purpose

- Reference tests for the wavelet coherence values (a sinusoid of known period, an independent Torrence and Compo CWT) belong with the fNIRS/wavelet work, which waits for #237; the reviewer's `scripts/fnirs/reference_tc2.py` and `wtc_synthetic.py` are ready to be turned into tests then
- Reference tests for the cluster statistics, `AR_local` and the simulator are in their own fix branches
- The GPU backends are untested by construction on this machine; the CI change that installs `numba` is a separate item

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
