# Changelog

## [Unreleased]

### Fixed
- `hypyp.sync`: asking for the Metal backend on a metric that has no Metal kernel (PLV, CCorr, Coh, ImCoh, EnvCorr, PowCorr) used to run in NumPy silently, while the metric object reported `metal`. `optimization='metal'` on these metrics now warns and falls back to NumPy, and a `priority` list skips Metal and moves on to its next backend. Only PLI, wPLI and ACCorr have a Metal kernel; the default selection of `optimization='auto'` is unchanged for the nine built-in metrics. No numerical implementation changes, but a request that used to run NumPy without saying so can now run the backend that comes next in its `priority` list: for example `priority=['metal', 'torch']` on PLV now runs torch, which computes in single precision on Apple GPUs (#299)
- `hypyp.sync`: when a backend of a `priority` list was skipped because the metric does not implement it and no other GPU backend of the list could be used, the warning now says so (for example "'plv' has no Metal implementation") instead of "No GPU backend available"
- `hypyp.sync`: a built-in metric whose backend is unknown or not implemented now raises a `ValueError` naming the metric and the backends it implements, instead of computing in NumPy without notice. Called through `compute_sync`, that error is still reworded as an unsupported metric (#306)
- Tests: the six Metal tests that compared NumPy with NumPy are replaced by tests of the fallback itself (warning, backend and result), the dispatch of every metric to every backend it implements is checked without a GPU, and the four tests that run a Metal kernel now assert that the Metal method was called (#300)

### Added
- `BaseMetric.supports(backend)` tells whether a metric implements a backend, for example `PLI.supports('metal')`

### Changed
- `hypyp.sync`: backend dispatch now lives in `BaseMetric.compute`, and each built-in metric implements `_compute_numpy` plus the optional `_compute_numba`, `_compute_torch`, `_compute_metal` and `_compute_cuda`. This is the recommended way to write a new metric (set `_dispatch_via_table = True` if the class also overrides `compute`). A subclass written the earlier way, which overrides `compute` and does its own dispatch, is still granted the backend it requests and is not subject to the capability check, whether it derives from `BaseMetric` or from a built-in metric. Two behaviours do change for a subclass of a built-in metric. Delegating to `super().compute()` with a backend that neither the subclass nor its parents implement as a `_compute_*` method now raises a `ValueError` instead of computing in NumPy. And when `optimization='auto'` falls back to the CPU, it selects numba only if a `_compute_numba` method exists: a subclass that handles numba inside its own `compute` gets NumPy unless numba is requested by name
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