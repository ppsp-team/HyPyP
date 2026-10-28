# Issue draft (for Patrice, after #237): three defects in the fNIRS/wavelet pipeline

Found by the 2026-10-13 review of 0.6.2 (`REVIEW-2026-10-13.md`, F13 to F15, reproduction scripts in `scripts/fnirs/`). Not patched, because draft PR #237 renames `fnirs/dyad.py`, `recording.py`, `study.py` and rewrites part of `wavelet/wtc.py`; the fixes below are described against 0.6.2 so that they can be applied on the new layout.

## 1. CWT cache key collision on identical subject labels (default path, wrong inter-subject matrix)

`base_wavelet.py:466-495` keys the CWT cache on `label_s1`/`label_s2`, channel, task, epoch and time range. `recording.py:173-180` takes the label from `subject_info["his_id"]`. Both bundled DCARE SNIRF files carry `his_id == "default"`, caching is on by default, and `Dyad.compute_wtcs(wavelet=None)` creates a caching `ComplexMorletWavelet()` (`dyad.py:342-343`). After the first row of the channel cross product, every lookup of `[default][ch]` returns subject 2's CWT, so rows 2 and beyond of the inter-subject matrix are subject 2's *intra* coherence.

Reproduced on three channels (`ch_match=["S1_D1 760", "S1_D2 760", "S2_D1 760"]`, `with_intra=False`):

```
no cache (correct)              default wavelet
S1_D1  0.224 0.254 0.234        0.224 0.254 0.234
S1_D2  0.226 0.242 0.225        0.254 1.000 0.500
S2_D1  0.216 0.235 0.219        0.234 0.500 1.000
```

`tests/test_fnirs.py:403-411` exercises exactly this collision and checks only counts.

Fix: give each `Recording` a unique id at load (`uuid4().hex` or `id(self)`), carry it in `PairSignals` (new field, also in `sub()` at `pair_signals.py:132-148`, `Dyad._append_pairs` at `dyad.py:180-194`, and `shiny/wavelet_explorer.py:264`), and put it in the cache key instead of (or next to) the label. Raise in `Dyad.__init__` when `s1 is not s2 and s1.subject_label == s2.subject_label`, after relabelling the tests that load the same file twice (`test_fnirs.py:241, 265, 317, 406, 417, 512, 536, 555, 682`). Regression test: the three-channel matrix above, cached against uncached.

## 2. pywt's CWT kernel degrades at large scales (conditional: broadband or high-rate signals)

`pywavelets_wavelet.py:89-96` calls `pywt.cwt(method="fft")` with the default `precision=12` (4096-point wavelet, step 0.0039). pywt builds the scaled kernel by integer-indexing the integrated wavelet (`pywt/_cwt.py:149-153`), so once `scale * step` approaches 1 (scale above about 250 samples) the kernel alternates weights and above 256 samples contains zeros. Envelope error against a smooth Morlet: scale 20 gives 0.08, 100 gives 0.33, 200 gives 0.50, 500 and above give 1.00. Against an independent Torrence and Compo CWT with identical smoothing: 50 Hz white noise mean |Δ| 0.133 (max 0.749); 50 Hz band-limited 0.02 to 0.7 Hz 0.0001; the bundled DCARE at 7.8 Hz 0.0006; `cwt_params=dict(precision=16)` at 50 Hz 0.0017. Pure sinusoids pass at every scale, which is why no test sees it.

The documented path (band-passed HbO at about 8 Hz, periods to 20 s) is safe; unfiltered `MnePreprocessorAsIs` data at 25 Hz or more, EEG envelopes, or `period_range=(2, 100)` at 10 Hz (`wavelet_lag_exploration.ipynb` cell 7 on FCS01 at 7.8 Hz reaches scale 781) are silently wrong.

Fix: compute the Morlet CWT in the Fourier domain (Torrence and Compo 1998, eq. 6, as pycwt does; `scripts/fnirs/reference_tc2.py::tc_cwt` is a 30-line reference), or set `precision` from `max(scales)` so that `2 ** precision` is well above `16 * max_scale`, and raise when it cannot be.

## 3. The time axis of every pair is `linspace(0, n / sfreq, n)`

`dyad.py:131` (also `shiny/wavelet_explorer.py:171, 261`, `matlab_wavelet.py:88`, `signal.py:26`): the step is `n / (n - 1)` times too long. `PairSignals.dt = x[1] - x[0]` feeds scales, periods, times, the COI and `WTC.sfreq`: a 20 s epoch at 7.8125 Hz reports `sfreq = 7.763`, every period and the COI are 0.6 % short (1.3 % on a 10 s epoch). `PairSignals.sub()` (`pair_signals.py:123-127`) derives crop indices from the same `sfreq`.

Fix: `np.arange(n) / sfreq`. Regression test: `pair.dt * sfreq == 1` to 1e-12.

## Also in the same files (MAJOR in the review, same owner)

- Scale-smoothing boxcar twice Grinsted's width (`base_wavelet.py:414-417`, a `* 2` carried over from pycwt with a TODO): white-noise mean R² 0.223 against 0.321 at Grinsted's width, so literature significance thresholds do not transfer. A method decision; at least the docstring should state the effective width.
- Default pipeline correlates raw light intensities and pairs 760 nm with 850 nm (`recording.py:92-94`, `dyad.py:245-254`); pivots average across chromophores.
- Intra-subject rows duplicated in `Study.df` when a Recording is in several dyads (`dyad.py:382-430`, `study.py:227-234`).
- Pseudo-dyads enumerate all combinations with no identity check, so real dyads reappear as `is_pseudo=True` when a participant recurs (`study.py:169-189`).
- COI of `ComplexGaussianWavelet` uses the DOG Fourier factor while the period axis uses pywt's (`pywavelets_wavelet.py:180-185`), over-masking by about 1.6×.
- `keep_wtcs=False` clears `s1.intra_wtcs` twice and never `s2` (`dyad.py:419-420`).
