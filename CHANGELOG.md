# Changelog

## [Unreleased]

### Added
- New connectivity metric: Adjusted Circular Correlation (`accorr`) in `analyses.py`
  - Hybrid implementation with vectorized numerator and exact denominator computation
  - Progress bar support via `tqdm` for monitoring computation progress
  - Available through `pair_connectivity()` and `compute_sync()` functions with `mode='accorr'`

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