# Changelog

## [Unreleased]

### Fixed
- `stats.statscluster(test="f oneway")` computed the F statistic on the first two groups only, whatever the number of groups given, while the degrees of freedom and the permutations used all of them; it now computes the one-way F on every group. **Changes results** for any call with more than two groups
- `stats.statscondCluster` and `stats.statscluster`: `F_obs_plot` and `Stat_obs_plot`, the statistic restricted to the significant clusters, looked each cluster up by its p-value and so kept only the first of several clusters sharing a p-value (the usual case for strong effects, all at `1/(n_permutations + 1)`), and kept nothing of a cluster whose statistic is negative. They now hold the statistic of every significant cluster, with its sign. **Changes results** for these two arrays only; `clusters` and `cluster_p_values` are unchanged
- `stats.statscluster`: `tail=-1` raised `ValueError: incompatible tail and threshold signs` for the t-tests, because the cluster-forming threshold was always positive; it is now negated for `tail=-1`. For the F-tests `tail=-1` is rejected with a message, since the statistic is non-negative
- `stats.statscondCluster`: `alpha` set the cluster-forming threshold only for `tail=0`; with `tail=1` MNE's fixed p = 0.05 threshold was used whatever `alpha` was. The F threshold is now computed from `alpha` for every tail. **Changes results** for calls with `tail=1` and `alpha != 0.05`

- `stats.con_matrix`, `stats.metaconn_matrix` and `stats.metaconn_matrix_2brains` built their across-frequency adjacency frequency-major (row `frequency * n_channels + channel`) while MNE flattens the `(n_observations, n_channels, n_frequencies)` data they are used with channel-major (row `channel * n_frequencies + frequency`), so every row of the adjacency described the wrong feature and cluster membership and p-values were wrong whenever more than one frequency was used, as in the tutorial. The three now return the channel-major (pair-major for the metaconnectivity) adjacency of `mne.stats.combine_adjacency`, which also drops the links between spatial neighbours at *neighbouring* frequencies the old expansion added, the convention of MNE and FieldTrip. **Changes results** for every cluster test with more than one frequency; with one frequency the adjacency is unchanged
- `stats.con_matrix(...).ch_con_freq`, `metaconn_matrix(...).metaconn_freq` and `metaconn_matrix_2brains(...).metaconn_freq` are now the `scipy.sparse.csr_matrix` their docstrings always announced; they were dense arrays, which MNE's cluster functions refuse (`If adjacency matrix is given, it must be a SciPy sparse matrix`). `scipy.sparse.csr_matrix(x)` and `bsr_matrix(x)` of the result still work. The pair rules are evaluated on arrays instead of element by element on the sparse matrix: 576 pairs at 3 frequencies take milliseconds instead of 4 s
- `stats.metaconn_matrix_2brains` inferred the number of channels from the pair list (`max(electrodes[:, 0]) + 1`) and raised `IndexError` for any list not containing the last channel of the first participant; it now takes it from `ch_con.shape[0]`
- `stats.statscondCluster` and `stats.statscluster` check that the adjacency has one row per feature of an observation and raise a `ValueError` naming the two accepted layouts otherwise; a dense adjacency is converted to sparse with a warning instead of failing inside MNE

- `stats.statscluster(test="rel ttest")` computed a paired t statistic but built its null with `permutation_cluster_test`, which permutes observations between the two groups: after a permutation, observation k of one group and observation k of the other are different pairs, so the null described var(A) + var(B) instead of var(A − B). When the pairs share a large component across features (every EEG participant has a topography) the test was wrong in both directions (p = 0.03 and 0.046 on the review's example where the sign-flip null gives 0.002 and 0.278) and could lose nearly all of its power. It now runs MNE's `permutation_cluster_1samp_test` on the within-pair differences, whose null flips the sign of each difference, and rejects groups of unequal size. **Changes results** for every `rel ttest` call

### Removed
- `stats.statscluster(test="f multipleway")`. With two factors it never ran (`f_threshold_mway_rm` returns one threshold per effect and MNE rejects the list), and with one factor it permuted observations between subjects under a repeated-measures F, which is not a valid null. The option now raises a `ValueError` that says so; `factor_levels` is kept in the signature, ignored, so that existing calls keep working

### Added
- `seed` argument on `stats.statscondCluster` and `stats.statscluster`, forwarded to MNE's permutation, so that a cluster test can be reproduced

## [0.6.2] - 2026-10-04

### Fixed
- `hypyp.sync`: asking for the Metal backend on a metric that has no Metal kernel (PLV, CCorr, Coh, ImCoh, EnvCorr, PowCorr) used to run in NumPy silently, while the metric object reported `metal`. The computation still runs in NumPy, so no computed value changes, but it now warns and the metric object reports `numpy`. This holds for `optimization='metal'` and for a `priority` list that reaches Metal on a machine where Metal is available: the backends that follow in the list are not tried, which is left to 0.7.0. Only PLI, wPLI and ACCorr have a Metal kernel; the default selection of `optimization='auto'` is unchanged for the nine built-in metrics (#299)
- `hypyp.sync`: when a backend of a `priority` list can neither run on the machine nor be computed by the metric, and no other GPU backend of the list can be used, the warning now says so (for example "'plv' has no Metal implementation") instead of "No GPU backend available"
- `hypyp.sync`: a metric whose `_backend` was set to a backend it does not implement used to compute in NumPy without notice. It still computes in NumPy, now with a warning. A `_backend` that is not the name of a backend raises a `ValueError` naming the metric and the backends it implements (#306)
- Tests: the six Metal tests that compared NumPy with NumPy are replaced by tests of the fallback itself (warning, backend and result), the dispatch of every metric to every backend it implements is checked without a GPU, and the four tests that run a Metal kernel now assert that the Metal method was called (#300)
- `import hypyp.analyses` failed when the import of torch, numba, cupy or the Metal bindings raised anything other than `ImportError`, for example the `OSError` of a missing shared library. The backend is now reported as unavailable, with a warning that gives the original error. An installed package that fails with an `ImportError` used to be skipped silently and now gets the same warning; a package that is not installed is still skipped silently. A package that imports but is broken further down is not covered
- `hypyp.sync.get_metric` rejected the names `envelope_corr`, `pow_corr` and `imaginary_coh`, which `compute_sync` accepts and the documentation uses. It now accepts them; the mapping is `hypyp.sync.METRIC_ALIASES`, and a metric registered in `METRICS` under one of these names takes precedence
- `hypyp.sync`: the warnings for a missing numba or torch suggested a `poetry install` command that no longer exists; they now give `pip install "hypyp[numba]"` and `pip install "hypyp[torch]"`
- `hypyp.fnirs.Recording.load_raw` raised a `TypeError` for a `Raw` object without `subject_info`, as built in memory with `mne.io.RawArray`, and a `KeyError` when `subject_info` had no `his_id`. Both now get the random label that was intended
- `hypyp.fnirs.Study.compute_wtcs_shuffle` raised a `TypeError` when given `with_intra=False`, so the keyword arguments of `compute_wtcs` could not be reused. That redundant argument is now accepted; `with_intra=True` is still refused
- `XDFImport(select_matches=[...])`: an integer that is not the id of a stream raised a bare `KeyError`; it now raises the same `ValueError` as an unknown stream name. A numpy integer is recognised as a stream id
- `compute_sync` and `pair_connectivity` reported every `ValueError` as `Metric type "..." not supported`, so an unknown `optimization` was announced as an unsupported metric. Only an unknown metric name gets that message now; any other error keeps its own (#306)
- `pair_connectivity` crashed with an `IndexError` when the frequencies were given as a list and the data had a single epoch, a single channel per participant or a single frequency. Results for data without such a dimension are unchanged
- `utils.normalizing` raised an `UnboundLocalError` for a `type` name other than `'Zscore'` or `'Logratio'`; it now raises a `ValueError` naming the two valid types
- `utils.generate_virtual_epoch` failed inside the solver for an odd number of channels; it now raises a `ValueError` saying that the number must be even, and its docstring example uses four oscillators
- `utils.epochs_from_tasks` failed with `min() iterable argument is empty` when the onset event of a task was absent from the recording; the `ValueError` now names the task and the event
- `utils.create_epochs` warns when the two participants give different numbers of epochs, or, when both are given as lists, different numbers of recordings. It still returns what it returned before
- `stats.statscluster` raised an `UnboundLocalError` for an unknown `test`; it now raises a `ValueError` listing the valid names

### Added
- `stats.metaconn_matrix` has a `plot` argument. It defaults to `True`, the earlier behaviour; `plot=False` computes the matrices without drawing
- `BaseMetric.supports(backend)` tells whether a metric implements a backend, for example `PLI.supports('metal')`

### Changed
- `hypyp.sync`: backend dispatch now lives in `BaseMetric.compute`, and each built-in metric implements `_compute_numpy` plus the optional `_compute_numba`, `_compute_torch`, `_compute_metal` and `_compute_cuda`. This is the recommended way to write a new metric: set `_dispatch_via_table = True` in the class. A subclass written the earlier way, which overrides `compute` and does its own dispatch, is still granted the backend it requests and is not subject to the capability check, whether it derives from `BaseMetric` or from a built-in metric. A subclass of a built-in metric that overrides `compute` and delegates to `super().compute()` keeps the earlier results: a backend without a `_compute_*` method computes in NumPy, now with a warning, and the CPU fallback of `optimization='auto'` still selects numba for it when numba is installed. A `_compute_*` method that a subclass adds for a backend its parent does not implement was never called before and is still not, unless the subclass sets `_dispatch_via_table = True` itself; a method of the parent that the subclass overrides is used, as before
- The whole code base is formatted with `ruff format` (line length 88). The change is purely cosmetic: the syntax tree of every file is unchanged, apart from whitespace inside docstrings, and the Python examples of `hypyp/sync/README.md` are formatted too. The vendored `hypyp/ext` and the tutorial notebooks are left untouched. The formatting commit is listed in `.git-blame-ignore-revs`, so `git blame` skips it (run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once in your clone; GitHub applies it automatically)
- The CI now checks formatting (`ruff format --check`) and a small set of lint rules that only catch certain bugs: syntax errors, invalid comparisons and undefined names. Ruff comes from the new `lint` dependency group, which the `dev` group includes
- `black` is removed from the `dev` dependency group, since `ruff format` replaces it

## [0.6.1] - 2026-10-03

### Fixed
- Metal backend: a single command queue is now cached per device instead of one being created on every call. Long loops (for example surrogate tests) used to exhaust the device's command queues and crash with `'NoneType' object has no attribute 'commandBuffer'` after about 7,800 calls (#279)

### Security
- Upgraded the 20 packages of the lock file that had known vulnerabilities: anyio, bleach, click, idna, jupyter-server, jupyterlab, mistune, mkdocs-material, nbconvert, notebook, pillow, pip, pygments, pymdown-extensions, python-multipart, setuptools, soupsieve, starlette, tornado and urllib3. None of the numerical dependencies (numpy, scipy, mne, numba, torch) changes version
- torch stays at 2.10.0 in the lock file for now. Its two open advisories concern `torch.jit.script` (CVE-2025-3000) and the loading of `.pt2` files (CVE-2026-4538), neither of which HyPyP uses

### Changed
- `mistune`, `pillow` and `urllib3` are no longer direct dependencies. HyPyP never imported them; they were listed only to force minimum versions of indirect dependencies. `pillow` and `urllib3` are still installed through `scikit-image`, `matplotlib` and `requests`, but HyPyP no longer imposes a minimum version on them: a fresh install receives the current releases, while an existing environment keeps whatever versions it already has, so keeping them up to date is now the user's responsibility. The lock file pins patched versions for development with `uv` only
- New `docs` dependency group holding only the documentation tools. `docs/requirements.txt` is now exported from it (36 packages instead of the whole development environment) and no longer installs `hypyp` itself from PyPI, which the documentation build does not need. The `dev` group includes the new group
- Read the Docs now builds with Python 3.12 instead of 3.10, which the project no longer supports

### Documentation
- Richer docstrings for the nine `hypyp.sync` metrics and for the CUDA and Metal kernels, with intent and literature references (#278)

## [0.6.0] - 2026-04-21

### Added
- **New `hypyp.sync` module**: Modular architecture for connectivity metrics
  - Extracted 9 connectivity metrics into separate classes: `PLV`, `CCorr`, `ACCorr`, `Coh`, `ImCoh`, `PLI`, `WPLI`, `EnvCorr`, `PowCorr`
  - `BaseMetric` abstract class for uniform interface across all metrics
  - `get_metric(mode, optimization)` function for easy metric instantiation
  - Helper functions: `multiply_conjugate`, `multiply_conjugate_time`, `multiply_product`
- **GPU and numba backends for all 9 sync metrics**:
  - numba JIT with `prange`: PLV, CCorr, Coh, ImCoh, PLI, wPLI, EnvCorr, PowCorr
  - PyTorch (MPS/CUDA/CPU) via batched einsum: all 9 metrics
  - Metal compute shaders (Apple Silicon): PLI, wPLI, ACCorr
  - CUDA raw kernels via CuPy (NVIDIA GPUs): all 9 metrics
- Benchmark-driven `AUTO_PRIORITY` table for `optimization='auto'`, compiled from
  Mac M4 Max (131 runs) and Narval A100 (111 runs) benchmarks
- `priority` parameter on `get_metric()` and `compute_sync()` for custom backend ordering
- `hypyp/sync/kernels/` submodule with Metal and CUDA dispatch infrastructure
- New optional dependencies: `pyobjc-framework-Metal` (Apple), `cupy-cuda12x` (NVIDIA)
- `multiply_conjugate_torch` and `multiply_conjugate_time_torch` GPU helpers

### Changed
- **BREAKING**: `accorr` metric now returns raw connectivity values with shape `(n_epoch, n_freq, 2*n_ch, 2*n_ch)` like all other metrics. The `swapaxes` and `epochs_average` operations are now handled by `compute_sync()` instead of being applied inside the metric.
- Refactored `compute_sync()` to use the new `hypyp.sync` module internally

### Deprecated
- `_multiply_conjugate()` in analyses.py - use `hypyp.sync.multiply_conjugate` instead (will be removed in 1.0.0)
- `_multiply_conjugate_time()` in analyses.py - use `hypyp.sync.multiply_conjugate_time` instead (will be removed in 1.0.0)
- `_multiply_product()` in analyses.py - use `hypyp.sync.multiply_product` instead (will be removed in 1.0.0)
- `_accorr_hybrid()` in analyses.py - use `hypyp.sync.ACCorr` instead (will be removed in 1.0.0)

## [0.5.0b13] - 2025-09-18

### Security
- Removed unused `future` package dependency (CVE-2025-50817 - High severity)
- Verified security updates for critical dependencies:
  - urllib3 >= 2.5.0 (addresses CVE related to redirect control)
  - requests >= 2.32.4 (addresses .netrc credentials leak)
  - pillow >= 11.3.0 (addresses buffer overflow vulnerability)

## [0.5.0b12] - 2025-09-18

### Added
- Python 3.13 support

### Changed
- Updated Python version constraint to support Python 3.13 (>=3.10,<3.14)

## [0.5.0b10] - 2025-07-10

### Added
- Proper package inclusion for fnirs, shiny, wavelet, xdf modules
- Fixed Poetry configuration for PyPI publishing

### Fixed
- Resolved Poetry build issues with sub-packages
- Fixed missing modules in published package

### Changed
- Updated pyproject.toml configuration
- Migrated to proper PEP 621 format