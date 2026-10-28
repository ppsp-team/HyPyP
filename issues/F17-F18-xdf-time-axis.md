# Issue draft: the XDF importer's Raw time axis ignores gaps and drift, and the two Raws are not aligned

Found by the 2026-10-13 review of 0.6.2 (F17, F18; `scripts/io/align_test.py` with the minimal writer `scripts/io/xdfwrite.py`, sections A, B, B2, B3, C). `fix/xdf-markers-scale` repairs the marker merge, the annotations and the scale; this is the remaining, larger item.

## What happens

`xdf_stream.py:204, 303, 316` build `RawArray(data, sfreq=nominal_srate)` from the bare sample array; pyxdf's `info["segments"]` and `info["effective_srate"]` are never read, and the markers are mapped through the dejittered timestamps (`as_mne_annotations`, onset = timestamp − first-sample timestamp). So:

- 100 Hz with a 6 s gap and a marker at +12 s: the Raw is 9.99 s long, the sample recorded at +12 s sits at t = 6.00 s, the marker is off by the gap and `set_annotations` keeps 0 of them
- below pyxdf's break threshold (`max(1 s, 500 samples)`), a 2 s gap is fitted by one line: effective rate 73.99 Hz, marker onset 8.65 s on a 7.99 s Raw
- on the shipped `data_with_clock_resets.xdf`: nominal 100 Hz, effective 92.93 Hz, two segments, 83 of 174 markers fall after the Raw ends
- a stream really running at 99.5 Hz (nominal 100) drifts by 50 ms over 10 s against its markers

For a dyad, a gap in one amplifier shifts every later sample of that participant relative to the other by the gap length, silently: every inter-brain measure computed afterwards is on non-simultaneous data.

Separately, each Raw has `times[0] = 0` at its own first sample and nothing crops to the common interval. On the shipped files the first-sample offset between the two EEG streams is 1.8 ms (noise file) and 13.8 ms with 6536 against 6538 samples (markers file, 1000 Hz: 14 samples, about 50° of alpha phase). `tutorial/import_from_xdf.ipynb` cell 8 presents `mne_raws` as ready; no docstring or test mentions `reference_time`.

## Fix

1. Build each Raw on a regular grid from the dejittered timestamps: `n = round((t_last - t_first) * nominal) + 1`, place each sample at `round((t - t_first) * nominal)`, fill the holes with zeros and add a `BAD_gap` annotation over each, so that `raw.times` is the recorder clock minus `t_first`. Markers then land where they were recorded.
2. Add `XDFImport.aligned_raws()` (or do it in `convert_streams_to_mne`): crop every signal Raw to `[max(t_first), min(t_last)]` on that grid, so that sample k is the same instant in every Raw; expose the common `reference_time`.
3. Read the XDF `<unit>` tag (the markers file declares `µV`) before `scale="auto"` guesses from the amplitude.
4. Tests on synthetic files (`tests/xdf_writer.py` from `fix/xdf-markers-scale`): a gap above and below the break threshold, a 0.5 % drift, two streams starting 0.5 s apart; assert sample-level alignment through the index channel the review's `eeg_stream` helper writes.

## Smaller items in the same module (review, MAJOR/MINOR)

- `select_matches` integers are XDF stream ids but documented as indices (`xdf_import.py:17, 201`; `test_match_stream_by_idx`)
- a declared signal stream with no samples crashes the default import (`xdf_stream.py:208, 303`)
- `datasets.fetch(force_download=True)` is a no-op (`datasets.py:167, 204`)
- the `time_series` orientation heuristic transposes wrongly when `n_samples == n_channels` (`xdf_stream.py:229-236`)
- duplicate stream names collapse `mne_raws_dict` (`xdf_import.py:108`)
- `verbose or True` always prints (`xdf_import.py:276-278`)
