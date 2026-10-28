# Issue draft: `analyses.xwt` wavelet coherence does not smooth as Grinsted et al. 2004 do, and crashes on common input

Found by the 2026-10-13 review of 0.6.2 (F10 and the `xwt` items of the `analyses` section; `scripts/analyses/s8_xwt_null.py`, `s5_xwt.py`). This is the EEG `xwt` in `analyses.py`, distinct from `hypyp.wavelet`, whose smoothing is right.

## Smoothing

`analyses.py:1166-1167` builds a Gaussian of std = scale truncated to length `int(2 * s)`, i.e. ±1σ (a Gaussian needs about ±3σ), and applies no smoothing across scales at all (Grinsted eq. 9: a boxcar of 0.6 × scale; Torrence and Webster 1999). On independent white noise (10 epochs, 128 Hz, 2048 samples, 8 voices per octave, 6 cycles):

| smoothing | mean WTC | 95th percentile |
| --- | --- | --- |
| HyPyP | 0.660 | 0.968 |
| reference with the same ±1σ truncation | 0.654 | 0.966 |
| full Gaussian | 0.465 | 0.868 |
| full Gaussian + 0.6-scale boxcar | 0.332 | 0.735 |

The published WTC significance thresholds (about 0.7 to 0.8) are meaningless at this null.

Fix: a window of at least 6σ (or FFT convolution, as Grinsted's `smoothwavelet.m`), plus the 0.6-scale boxcar across the scale axis, which needs `freqs` log-spaced (say so in the docstring). Or delegate to `hypyp.wavelet.BaseWavelet.wtc`, which already implements both operators.

## Crashes and types

- one epoch: `np.squeeze` at `:1175` drops the epoch axis, then `IndexError`
- an unknown `mode` returns the *string* at `:1219-1221` instead of raising
- `freqs` as `int` is in the signature (`:1059`) but `len(freqs)` at `:1130` raises `TypeError`
- `cwt` is recomputed `2 * n_ch²` times in the double loop (`:1172-1186`); compute each channel's CWT once
- the output `cross_sigs + wtcs` is `n_ch² × n_epochs × n_freqs × n_times` complex128: 197 GB for 64 channels, 100 epochs, 30 frequencies, 1000 samples

## Also in `viz.plot_xwt`

`viz.py:3048-3070` passes `xwt`'s 5-D output straight to `imshow` and crashes on every input (`TypeError: Invalid shape (1, 1, 33, 4, 501) for image data`); it is called nowhere. Either delete it or reduce the array first.
