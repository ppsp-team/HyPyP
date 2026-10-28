# fix(analyses,mvarica): repair the MVAR path

Branch: `fix/mvarica-arity-merge` (from `master` at dbeeb37). Open with:
`gh pr create --draft --base master --head fix/mvarica-arity-merge --title "fix(analyses,mvarica): repair the MVAR path" --body-file pr-bodies/fix-mvarica-arity-merge.md`

## What this does

Three repairs on the MVAR path, from the 2026-10-13 review (F9 and two MAJOR items of the `mvarica` section).

**hypyp/mvarica.py**

- `MVAR(delta=...)` with a non-zero ridge penalty, the call the class docstring shows, raised `TypeError: construct_equation() takes from 2 to 3 positional arguments but 4 were given`: `fit` passed the model order where the penalty goes and the penalty as a fourth argument. The penalty is now passed through
- `MVAR.stability()` raised `ValueError` for `model_order=1` while stacking a lower block of height zero. The companion matrix is now the coefficient blocks over `np.eye(n * (p - 1), n * p)`, which is empty for order 1 and identical to the previous block for every other order. It returns a plain `bool`

**hypyp/analyses.py**

- In `compute_conn_mvar`, answering "yes" to "Do you want to merge the epochs?" reshaped the `(n_epochs, n_freq, n_channels, n_times)` array straight to `(n_freq, n_channels, n_times * n_epochs)`; C-order memory is epoch-major, so the merged series of channel 0 held epoch 0's channels end to end rather than channel 0's epochs. The stability check that followed indexed `[:, 0, 0, :]`, channel 0 alone. `_merge_epochs` moves the epoch axis next to time before reshaping, and the check runs on every channel

**Changes numerical results: yes, on the merge path only**, which crashed or scrambled before. No other call returns a different value: `fit` with `delta=0`, `stability()` for order 2 and above, and the non-merge branches of `compute_conn_mvar` are unchanged.

## Measured

- `tests/test_mvarica.py` (new, 5 tests): a VAR(2) generated from known coefficients is recovered to 0.05 (reference), the ridge path fits, order-1 stability runs, an explosive model is detected, and `_merge_epochs` is checked on an array whose values are their own indices. On `master`: `TypeError` (ridge), `ValueError` (order 1), no `_merge_epochs` (`pr-bodies/evidence/mvarica-arity-merge.before.txt`); the two reference checks pass before and after. After the fix: 5 passed, plus `tests/test_pdc.py`
- Full suite on this branch: 363 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 3 min 0 s, Python 3.13.7, Linux x86-64, scipy 1.17.1
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `HyPyP-review/scripts/compare/`): 55 bit-identical, including `MVAR(2).fit` coefficients and `stability()`; the merge path is interactive and not exercised by the harness

## Left out on purpose

- `compute_conn_mvar` still calls `input()` twice and returns `None` or a string on a declared `np.ndarray` signature; `connectivity_mvarica` returns links between ICA sources with the mixing matrix discarded, so its indices are not channels although the docstring says so (review, `mvarica` M3 and M6). Both are API changes for a separate pull request
- The MVAR residuals keep the first `p` raw samples of every epoch and feed them to the ICA; the default `var_model=MVAR` is the class, not an instance; an unknown `measure_name` hits `UnboundLocalError`. All listed in the review, none touched here

🤖 Generated with [Claude Code](https://claude.com/claude-code)
