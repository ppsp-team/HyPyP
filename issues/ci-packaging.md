# Issue draft: CI, packaging and repository hygiene

Found by the 2026-10-13 review of 0.6.2 (section 4). None of this changes a computed value; all of it is quick.

## CI (`.github/workflows/Build.yml`, `Build_on_pull_request.yml`)

- `uv sync --group dev` installs no accelerated extra, so numba/torch/Metal/CUDA paths are never exercised: 32 of the 41 backend-consistency tests skip on CI and `hypyp/sync/kernels/*` is at 0 % coverage. Add `--extra numba` to one job (pure pip, CPU) and a torch-CPU job; keep Metal and CUDA documented as untested
- No `uv build` plus install-from-wheel smoke test: tests run on the editable tree, so a package-data omission would pass CI
- No `mkdocs build --strict` (it passes locally in under a second)
- No coverage upload, no `concurrency:` cancel, actions pinned by major tag only
- The 3.14 job has `continue-on-error`, so its result is invisible; nothing in the lock blocks 3.14, so make it a real job
- `on: [push, workflow_call]` plus the PR workflow runs same-repo PRs twice
- `Build_on_pull_request.yml:5` filters on a branch `fNIRS` that does not exist
- Cache `~/mne_data` (51 MB) across jobs; `tests/test_fnirs.py` still downloads at import (the XDF tests no longer do after `test/reference-suite`)
- Run `scripts/infra/nb_drift.py` from the review (an AST check that every `hypyp.*` call of the nine notebooks still resolves) as a cheap CI step; executing `getting_started.ipynb` headless is blocked by `ICA_choice_comp`'s `input()`

## Packaging (`pyproject.toml`)

- `certifi`, `requests` and `snirf` are declared and never imported; `importlib-resources` duplicates the stdlib on `requires-python >= 3.11` (`viz.py:22, 2061, 2161`); `mne-connectivity` and `seaborn` are imported only by `plots.py` (fNIRS dataframe plots and the Shiny viewer)
- `numpy>=2.2.3`, `pyarrow>=20.0.0`, `scikit-image>=0.25.2` were set in the Poetry-to-uv migration commit (6fa8a9e) with no rationale; the numpy floor excludes every numpy 1.x environment and nothing in the code needs numpy 2. Either test the real minimums in a CI job or state the floors as "tested with"
- The sdist ships `tests/test_*.py` but not `conftest.py`, `accorr_reference.py` or `tests/data`, so its tests cannot run
- `hypyp/io.py` is empty while the README lists "io.py: Loaders"; `shiny/study_viewer.py:12-13` does `sys.path.append(<repo root>)` and `matlab_wavelet.py:35-40` expects a directory four levels above the package, both broken from an installed wheel

## Repository

- `.DS_Store` and `tutorial/.DS_Store` are tracked despite `.gitignore`
- `scripts/run_notebooks.sh` hardcodes a personal macOS path; `dashboards/*.sh` bind `0.0.0.0`; neither directory is referenced from the README or docs
- README: "beta" against `Development Status :: 5 - Production/Stable`; a "Mattermost" badge links to Discord; `datasets`, `sync` and `xdf` are not mentioned
- `CHANGELOG.md`: no entries for v0.5.0b14 to b18; the 0.6.0 entry omits the uv migration, the Python 3.10 drop, the new `datasets` module and the removal of sample data from the repository
- `hypyp/ext/mpl3d`: BSD headers are in the files but no `LICENSE` file, no provenance note, imports rewritten so "kept as upstream wrote it" is not literal; `mesh.py:65, 67` reference undefined names in `mode="back"`; `lighting.py` is dead
- HypypData (`ppsp-team/HypypData`) is public, holds real recordings (a DCARE dyad, a dated parent recording), has no `LICENSE` and no provenance or consent statement; `datasets.py:149` points at branch `main` with no pin, so a changed remote file fails the hash check with no recovery short of a HyPyP release
