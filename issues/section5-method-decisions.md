# Issue draft: method decisions the maintainers should take explicitly

From section 5 of the 2026-10-13 review. Items 1, 2, 3, 6 and 9 are now *stated* in the documentation (`docs/conventions`, `fix/stats-statscond`) without changing any value; items 4, 5, 7 and 8 are neither changed nor documented yet. Each needs a decision, then a sentence of documentation or a declared value change.

| # | Where | What the code does | What the cited reference does | Numerical consequence of changing | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `sync/coh.py`, `imaginary_coh.py`, `ccorr.py` | coherence = \|C\|; ImCoh and CCorr = absolute values | magnitude-squared coherence (common in EEG); signed ImCoh (Nolte 2004) and CCorr (Burgess 2013) | every coherence value becomes its square (0.84 → 0.70); ImCoh and CCorr gain a sign and lose their positive null bias | documented |
| 2 | `analyses.compute_sync` | within-epoch statistic, averaged over epochs (single-trial form) | across trials at each time (Lachaux 1999); across-trial wPLI (Vinck 2011) | bias depends on epoch length (PLV 0.32 / 0.15 / 0.08 for 1 / 4 / 16 s) | documented |
| 3 | `sync/wpli.py`, `envelope_corr.py` | wPLI not debiased; envelopes not orthogonalised | debiased wPLI (Vinck 2011 eq. 11); orthogonalised envelopes (Hipp 2012) | 0.674 → 0.454 on the review's pair; within-brain envelope values drop by the volume-conduction share | documented |
| 4 | `wavelet/base_wavelet.py:414-417`, `pywavelets_wavelet.py:112-114, 143-147` | scale-smoothing window twice Grinsted's; ω0 = 2π; period axis in pywt's convention | boxcar of 0.6 scale (Grinsted 2004, T&C 1998 Table 2); ω0 = 6; T&C Fourier factor | white-noise mean R² 0.223 → 0.321; periods shift 1.25 % | not documented (fNIRS, after #237) |
| 5 | `analyses.compute_nmPLV` | both participants at the same frequency bin, raised to n and m | φ_x(f1) against φ_y(f2) (Palva 2005) | on pure tones the same-bin design works only through spectral leakage | documented in the docstring (fix/analyses-nmplv) |
| 6 | `analyses.compute_single_freq` | n_cycles = 4, time_bandwidth = 4: half-bandwidth = f/2 | narrower bins are usual for "single frequency" analyses | neighbouring integer bins overlap strongly | documented |
| 7 | `stats.con_matrix` and `metaconn_*` | after `fix/stats-adjacency-layout`: MNE's convention, no diagonal frequency links | the old expansion linked spatial neighbours at neighbouring frequencies (98 links against 58 on 4 × 3) | cluster membership | changed by the fix (declared in its changelog) |
| 8 | `fnirs/study.py:169-189` | pseudo-dyads = all n(n−1) combinations, no identity check, no seed, no `n_pseudo` | a sampled surrogate distribution that excludes real dyads | contaminated surrogates when a participant recurs | not documented (fNIRS, after #237) |
| 9 | `stats.statsCond` | one-sample sign-flip test against zero, max-T | the same, but the tutorial passes raw power | raw power is "significant" everywhere | documented (fix/stats-statscond) |

Recommended defaults if the maintainers want them: keep 1, 2, 3 and 6 as they are and documented (done); take 4 and 8 with the fNIRS rewrite; leave 5 to whoever uses n:m PLV in a study; 7 and 9 are closed.
