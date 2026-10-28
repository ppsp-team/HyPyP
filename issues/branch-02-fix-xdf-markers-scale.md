# fix(xdf): merge marker streams by time, attach annotations, scale µV to V

**Fix branch:** [`fix/xdf-markers-scale`](https://github.com/ppsp-team/HyPyP/tree/fix/xdf-markers-scale), two commits: `55e2a46` (the fix with its tests) and `90ef250` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/master...fix/xdf-markers-scale. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

Four repairs to the XDF importer, found by the 2026-10-13 review (F16, F19 and two of the MAJOR items), each reproduced on a synthetic two-stream file written by the new `tests/xdf_writer.py`.

**hypyp/xdf/xdf_stream.py**

- `Markers.__add__` concatenated `self.timestamps` with itself. With two marker streams selected (one per headset, as in the shipped `dyad-example-noise.xdf`), `XDFImport.markers` carried the first stream's times under the second stream's descriptions: `Markers([1, 2]) + Markers([100, 200])` gave timestamps `[1, 2, 1, 2]`. The two sets are now concatenated and sorted by time
- `as_mne_annotations` assumed sorted timestamps and walked to the first non-negative onset, so an unsorted list kept negative onsets or dropped later markers; it now keeps every marker at or after the reference time, in time order
- `scale="auto"` multiplied by `10e-6`, which is 1e-5 and neither the µV nor the mV factor: 20 µV of noise came out as 2.0e-4 V. It now multiplies by 1e-6

**hypyp/xdf/xdf_import.py**

- The conversion loop looked for markers on the signal streams, which only marker streams carry (`XDFStream.markers` is set when `srate == 0`), so no Raw ever received annotations although the docstring, `tutorial/import_from_xdf.ipynb` and `datasets.xdf_dyad_with_markers` say it does. The markers of the selected marker streams are now set on every signal Raw, with onsets relative to that stream's first sample

**Changes numerical results: yes.** Merged marker timestamps whenever more than one marker stream is selected; `scale="auto"` values (no test or tutorial passes it); and Raws now carry annotations they did not have, so `raw.annotations` and anything epoched from them changes.

## Measured

- `tests/test_xdf_synthetic.py` (new, 5 tests): two EEG streams at 100 Hz, the second starting 0.5 s after the first, two marker streams at +1.0 s and +2.5 s. Written before the fix; on `master` all five fail on the substance (timestamps `[1, 3, 1, 3]`, onsets `[-1, 3, 4]`, merged times `[1001, 1001]`, no annotations, std 2.0e-4 V). After the fix: 5 passed. The 26 existing XDF tests still pass (31 passed in 1.9 s)
- The review's `scripts/io/align_test.py` before and after: `evidence/xdf-markers-scale.{before,after}.txt` (sections D, G and J are the ones this branch changes)
- Full suite on this branch: 363 passed, 42 skipped (no numba, torch, Metal or CuPy installed), Python 3.13.7, Linux x86-64, mne 1.11.0, pyxdf as locked
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `scripts/compare/`): 53 bit-identical. Changed: `xdf.raw0.n_annotations` and `xdf.raw1.n_annotations` on `dyad-example-with-markers.xdf`, 0 → 12 (the markers are now attached). The Raw data arrays and the marker timestamps are identical (that file has one marker stream; the merge repair needs two, which the synthetic test covers)

## Left out on purpose

- The Raw time axis is still built from the nominal rate and the bare sample array, so a gap or a drift in a stream shifts every later sample against the markers and against the other participant, and the two Raws are not cropped to a common interval (review F17, F18). That needs a regular time grid from the dejittered timestamps and an `aligned_raws()` helper; it is a larger change and is left for its own branch, with an issue drafted
- `scale="auto"` still guesses from the amplitude instead of reading the XDF `<unit>` tag
- `select_matches` integers are stream ids although the docstring says indices; empty signal streams crash the default import; duplicate stream names collapse `mne_raws_dict`. All documented in the review, none touched here

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
