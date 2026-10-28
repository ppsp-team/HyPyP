# fix(stats): order the across-frequency adjacency like the data

Branch: `fix/stats-adjacency-layout`, **built on `fix/stats-cluster-maps`** (merge that one first; this branch contains its two commits). Open with:
`gh pr create --draft --base fix/stats-cluster-maps --head fix/stats-adjacency-layout --title "fix(stats): order the across-frequency adjacency like the data" --body-file pr-bodies/fix-stats-adjacency-layout.md`

(Opened against its base branch so that the diff shows only this branch's commits; GitHub retargets it to `master` once `fix/stats-cluster-maps` is merged.)

## What this does

Repairs review finding F1, the most consequential item of the `stats` section, and three items next to it (dense return, channel-count inference, no layout check).

**hypyp/stats.py**

- `con_matrix`, `metaconn_matrix` and `metaconn_matrix_2brains` expanded their adjacency over frequencies frequency-major (row `frequency * n_channels + channel`), while MNE flattens the `(n_observations, n_channels, n_frequencies)` data they are used with channel-major (row `channel * n_frequencies + frequency`). `tests/test_stats.py::test_stats` and tutorial cell 36 feed exactly that layout. Every row of the adjacency therefore described the wrong feature: on four channels and three frequencies, the neighbours the matrix gave to feature (A, f1) were (A, f0..f2), (B, f0..f2), (C, f0), (C, f1), the row having been built for (f0, B); an effect on channel A at f0 and f2 came out as one cluster instead of two. The three functions now return the channel-major (pair-major for the metaconnectivity) adjacency of `mne.stats.combine_adjacency` (`_across_frequencies`), which also drops the links between spatial neighbours at *neighbouring* frequencies that the hand-written expansion added (98 links against 58 on the four-by-three example), the convention of MNE and FieldTrip
- The across-frequency matrices are returned as the `scipy.sparse.csr_matrix` the docstrings always announced; they were dense arrays, which MNE refuses (`If adjacency matrix is given, it must be a SciPy sparse matrix`), so every caller wrapped them in `bsr_matrix` or `csr_matrix`; those wrappers still work
- The pair rules are evaluated on boolean arrays instead of element by element on the sparse matrix: 576 pairs at three frequencies take milliseconds instead of 4 s. `metaconn_matrix_2brains` takes the channel count from `ch_con.shape[0]` instead of the largest first-brain index in the pair list, which raised `IndexError` for any list not containing the last channel; second-brain indices below that count still wrap around, as some callers rely on
- `statscondCluster` and `statscluster` check that the adjacency has one row per feature of an observation and raise a `ValueError` naming the two accepted layouts otherwise (`(n_observations, n_channels, n_frequencies)` with `ch_con_freq`; one connectivity vector per observation in `electrodes` order with `metaconn_freq`); a dense adjacency is converted with a warning. `statscluster`'s note that advised flattening the full `(n_sensors, n_sensors)` matrix, which counts every pair twice and includes the diagonal, now says the opposite

**tests/test_stats.py**

- `test_metaconn_matrix_2brains` asserted the frequency-major layout (offsets of `n_channels` in a matrix whose blocks are `n_pairs` wide); rewritten for the pair-major order. `test_metaconn_matrix_plot_argument` compares the sparse results through `.toarray()`

**Changes numerical results: yes**, for every cluster test with more than one frequency: cluster membership and p-values. With one frequency the adjacency is the same as before (`combine_adjacency(A, 1)` is `A`), so `tests/test_stats.py::test_stats` and the UCLA-R21 pipeline, which use one frequency per band, are unchanged. The return type of the three `*_freq` matrices changes from `np.ndarray` to `scipy.sparse.csr_matrix`.

## Measured

- `tests/test_stats_adjacency.py` (new, 11 tests): the across-frequency and pair rules written out independently on a four-sensor montage (A~B, A~C, A!~D); an effect on channel A at frequencies 0 and 2 gives two clusters `{(A, f0)}` and `{(A, f2)}`; one frequency reproduces the channel adjacency; pair-major order of `metaconn_freq`; pair lists without the last channel; second-brain indices from zero; the wrong-size error and the dense-array warning. On `fix/stats-cluster-maps`: 7 of 11 fail (`pr-bodies/evidence/stats-adjacency-layout.before.txt`), the two pair-rule tests pass before and after, which is the review's finding that the rules themselves were right. After the fix: 11 passed, plus the 9 of `test_stats_cluster.py` and the 16 of `test_stats.py`
- The review's `scripts/stats/adj_check.py` and `followup_check.py` after the fix: `pr-bodies/evidence/stats-adjacency-layout.after.txt`. Its two "equals combine_adjacency(...)" lines print False only because the script compares the now-sparse result with a dense array through `np.array_equal`; `test_con_matrix_links_channel_major_features` makes the same comparison properly
- Full suite on this branch: 378 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 41 s, Python 3.13.7, Linux x86-64, mne 1.11.0
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `HyPyP-review/scripts/compare/`): 49 bit-identical. Changed beyond the four of its base branch: `con_matrix.3f.ch_con_freq` and `metaconn.2f.metaconn_freq`, the across-frequency layout. The one-frequency adjacencies (`con_matrix`, `metaconn_matrix`, `metaconn_matrix_2brains`) and `statscondCluster` on them are bit-identical, as the "with one frequency the adjacency is unchanged" statement requires

## Left out on purpose

- `rel ttest` and `f multipleway` (review F5, F7) are the subject of `fix/stats-paired-test`, built on this branch
- `statsCond`'s double correction (F6) is the subject of `fix/stats-statscond`
- `statscondCluster`'s unused `freqs_mean` argument and the two result tuples sharing the name `"statscondCluster"` are left alone
- Tutorial cell 38 passes `(n_observations, 31, 31)` connectivity matrices with the 31 by 31 channel adjacency, which MNE used to take as a time lattice over the first axis; it now raises the layout error, and the conventions pull request rewrites that cell around `metaconn_matrix`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
