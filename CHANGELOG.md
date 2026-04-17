# Changelog

## [Unreleased]

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