# Changelog

## [Unreleased]

### Changed
- The whole code base is formatted with `ruff format` (line length 88). The change is purely cosmetic: the syntax tree of every file is unchanged, apart from whitespace inside docstrings. The vendored `hypyp/ext` and the tutorial notebooks are left untouched. The formatting commit is listed in `.git-blame-ignore-revs`, so `git blame` skips it (run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once in your clone; GitHub applies it automatically)
- The CI now checks formatting (`ruff format --check`) and a small set of lint rules that only catch certain bugs: syntax errors, invalid comparisons and undefined names. Ruff comes from the new `lint` dependency group, which the `dev` group includes

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