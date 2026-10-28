# fix(utils): integrate the Kuramoto coupling at the sample step

Branch: `fix/utils-virtual-epoch` (from `master` at dbeeb37). Open with:
`gh pr create --draft --base master --head fix/utils-virtual-epoch --title "fix(utils): integrate the Kuramoto coupling at the sample step" --body-file pr-bodies/fix-utils-virtual-epoch.md`

## What this does

Repairs `generate_virtual_epoch`, review finding F12, and seeds both simulators.

**hypyp/utils.py**

- `solve_ivp` ran at its default tolerances (`rtol=1e-3`, `atol=1e-6`) on an unbounded phase of hundreds of radians, which allows an error of about half a radian per step; RK45 took some 230 evaluations for 2000 output samples. With `W[2, 0] = 20` and a frequency mismatch of 2.5 rad/s, a pair that must lock gave PLV 0.35 from the solver's own phases (1.000 at `rtol=1e-9`), and through the library call the coupled pair read 0.24 where an uncoupled pair read 0.66. The solver now takes steps no longer than one sample (`max_step=1/sfreq`); the model is unchanged
- The time grid was `np.linspace(0, tmax, Nt)`, whose step is `Nt / (Nt - 1)` times too long, so a free oscillator appeared 0.05 % fast on the sample grid; it is now `np.arange(Nt) / sfreq`
- `rng` on `generate_virtual_epoch` and `generate_random_epoch`: a `numpy.random.Generator` or a seed, in place of the unseeded global `numpy.random`
- The docstring example called the removed `mne.time_frequency.psd_welch`; it uses `Epochs.compute_psd`

**Changes numerical results: yes, for every call of `generate_virtual_epoch`**; any validation of a connectivity measure done with it should be rerun. `generate_random_epoch` draws the same distribution from a different generator, so its values differ but not their statistics. No other function is touched.

## Measured

- `tests/test_virtual_epoch.py` (new, 4 tests): a pair coupled at twenty times its mismatch settles to a constant phase difference (PLV above 0.99, standard deviation below 0.05 rad over the last epoch); a free oscillator with no frequency spread runs at 10.000 Hz on the sample grid to 1e-3 Hz (the `linspace` grid gives 10.005); a seed reproduces the output; `generate_random_epoch` takes a seed. Written before the fix; on `master` all four fail on the missing `rng` (`pr-bodies/evidence/utils-virtual-epoch.before.txt`). After the fix: 4 passed, plus the 23 of `tests/test_utils.py` and the 16 of `tests/test_stats.py`, which uses `generate_random_epoch`
- The review's `scripts/prep/followup.py` through the library call, before and after: `pr-bodies/evidence/utils-virtual-epoch.{script-before,after}.txt`
- Full suite on this branch: 362 passed, 42 skipped (no numba, torch, Metal or CuPy installed), 2 min 52 s, Python 3.13.7, Linux x86-64, scipy 1.17.1. The review script through the library call: coupled pair PLV 0.235 and uncoupled 0.662 before, 1.000 and 0.104 after
- `ruff format --check` and `ruff check` clean
- Output comparison with `master` (55 arrays on the sample data, `HyPyP-review/scripts/compare/`): 53 bit-identical. Changed: `generate_virtual_epoch` and `generate_random_epoch`, as declared (new integration step and grid; a numpy Generator instead of the global state). Nothing else

## Left out on purpose

- The phase noise is still drawn inside the right-hand side of an adaptive solver, which is not a well-defined stochastic integration; with the step now bounded by one sample its effect is bounded too, and at the default level (0.005 / n_samp per unit time) it is negligible. A fixed-step Euler-Maruyama scheme would be a model change for the maintainers to decide
- The first half of the oscillators still start at phase zero, which is the only reason for the even-channel requirement; unchanged

🤖 Generated with [Claude Code](https://claude.com/claude-code)
