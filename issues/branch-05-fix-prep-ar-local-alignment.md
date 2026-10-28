# fix(prep): keep every participant on the same epochs after autoreject

**Fix branch:** [`fix/prep-ar-local-alignment`](https://github.com/ppsp-team/HyPyP/tree/fix/prep-ar-local-alignment), two commits: `f16fafd` (the fix with its tests) and `ffca466` (its changelog entry). Diff: https://github.com/ppsp-team/HyPyP/compare/master...fix/prep-ar-local-alignment. To adopt it, open a pull request from that link (the commits are kept) or cherry-pick the two commits. Nothing is merged or proposed for merge here; the maintainer decides.

## What this does

Repairs `AR_local`, review finding F11 (the `prep` section's one CRITICAL item) and its two-participant assumption (M6).

**hypyp/prep.py**

- With `strategy="intersection"`, only the epochs bad for both participants were dropped from both; `ar.transform(epochs)` then recomputed each participant's own reject log and dropped that participant's remaining bad epochs, and `equalize_epoch_counts` matched the counts without matching the epochs. On a synthetic dyad with bad epoch 5 planted in participant 1 only and bad epoch 9 in participant 2 only, both came out with 39 epochs, participant 1 missing epoch 5 and participant 2 missing epoch 9: from index 5 on, every epoch of one was paired with a different second of the other. The docstring said these epochs were "repaired individually"; they were dropped
- The epochs dropped are now one list for every participant under both strategies. For the epochs kept, `transform` receives the fitted reject log restricted to them (`_kept_log`) with no epoch marked bad, so autoreject interpolates the channels it marked and drops nothing. The function then checks that every participant comes out with `n - len(dropped)` epochs and raises `RuntimeError` otherwise
- The reject logs of every participant are read, not just `reject_logs[0]` and `[1]`, so a third participant no longer goes out of alignment (its bad epochs were ignored before), and `dic_AR` carries `S1`, `S2`, `S3`, ... one per participant. `dic_AR["dyad"]` is the percentage of epochs dropped
- Docstring: what each strategy does, the alignment guarantee, the new `RuntimeError`

**tests/test_prep.py**

- `test_AR_local` asserted `S1 + S2 == dyad`, which is not an invariant (it fails whenever the two bad sets overlap, and under the intersection it fails whenever they differ: on the sample data it read `0 + 22.2 == 0`). It now checks the output count against `dyad` and the bounds each strategy implies: `max(S1, S2) <= dyad <= S1 + S2` for the union, `dyad <= min(S1, S2)` for the intersection

**Changes numerical results: yes, for `strategy="intersection"`:** its output epochs are now aligned across participants (the union strategy returns the same epochs as before), and its `dic_AR["dyad"]` counts the epochs dropped rather than `n_in - n_out` of participant 1, which is the same number now that nothing else is dropped.

## Measured

- `tests/test_ar_alignment.py` (new, 3 tests): twelve channels, forty one-second epochs, one epoch of fifty times the noise planted in one participant only; kept events compared sample for sample across two participants under both strategies, and across three participants. On `master`: the intersection case fails with participant 1 keeping `0..4, 6..` and participant 2 keeping `0..8, 10..`, the three-participant case fails (`evidence/prep-ar-local-alignment.before.txt`). After the fix: 3 passed, plus the 7 of `tests/test_prep.py` (29.6 s, autoreject fits)
- The review's `scripts/prep/ar_alignment.py` before and after: `evidence/prep-ar-local-alignment.{script-before,after}.txt` (cases A to C print `ALIGNED: True/False`)
- Full suite on this branch: 361 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 57 s, Python 3.13.7, Linux x86-64, autoreject 0.4.3, mne 1.11.0
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `scripts/compare/`): 55 bit-identical, including the kept events of both participants under both strategies. Autoreject finds no bad epoch in the two sample recordings, so neither strategy drops anything there and the intersection change cannot show on this data; the synthetic test with planted bad epochs is where it shows

## Left out on purpose

- `AutoReject`'s `random_state=42`, `n_interpolates`, `consensus_percs` and `verbose="tqdm_notebook"` are still fixed inside the function; `filt`, `merge` and `ICA_apply` still mutate their inputs undocumented; `ICA_choice_comp` still blocks on `input()`. All listed in the review's `prep` section, none touched here
- `filt`'s zero-phase FIR and `ICA_fit`'s per-participant thresholding were checked by the review and are correct

---
*From the 2026-10-13 review of 0.6.2 (umbrella issue: see the checklist there). Evidence files (`evidence/...`) and the comparison harness (`scripts/compare/`) are in the review bundle, branch `review/2026-10-13`.*
