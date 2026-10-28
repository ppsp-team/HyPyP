"""Shape conventions, mismatched shapes, frequency range inclusivity, bandwidth."""
import warnings
import traceback
import numpy as np
import mne
from hypyp import analyses

warnings.simplefilter("ignore")
rng = np.random.default_rng(1)
fs = 256


def attempt(label, fn):
    try:
        out = fn()
        print(f"[{label}] OK -> shape {getattr(out, 'shape', out)}")
    except Exception as exc:
        print(f"[{label}] {type(exc).__name__}: {str(exc)[:160]}")


good = rng.standard_normal((2, 3, 4, 512))
attempt("baseline dict", lambda: analyses.pair_connectivity(good, fs, {"a": [8, 12]}, "plv"))
attempt("baseline list", lambda: analyses.pair_connectivity(good, fs, [8, 12], "plv"))

# participant 2 has 3 channels instead of 4
ragged = [rng.standard_normal((3, 4, 512)), rng.standard_normal((3, 3, 512))]
attempt("n_ch mismatch dict", lambda: analyses.pair_connectivity(ragged, fs, {"a": [8, 12]}, "plv"))
attempt("n_ch mismatch list", lambda: analyses.pair_connectivity(ragged, fs, [8, 12], "plv"))

# participant 2 has different n_times
ragged_t = [rng.standard_normal((3, 4, 512)), rng.standard_normal((3, 4, 500))]
attempt("n_times mismatch dict", lambda: analyses.pair_connectivity(ragged_t, fs, {"a": [8, 12]}, "plv"))

# n_epochs mismatch
ragged_e = [rng.standard_normal((3, 4, 512)), rng.standard_normal((2, 4, 512))]
attempt("n_epochs mismatch", lambda: analyses.pair_connectivity(ragged_e, fs, {"a": [8, 12]}, "plv"))

# forgot participant axis: (n_epochs, n_channels, n_times) with n_epochs==2
three_d = rng.standard_normal((2, 4, 512))
attempt("3-D (2,4,512) dict", lambda: analyses.pair_connectivity(three_d, fs, {"a": [8, 12]}, "plv"))
attempt("3-D (2,4,512) list", lambda: analyses.pair_connectivity(three_d, fs, [8, 12], "plv"))

# compute_sync fed a 4-D array (missing freq axis)
cs4 = rng.standard_normal((2, 3, 4, 512)) + 1j * rng.standard_normal((2, 3, 4, 512))
attempt("compute_sync 4-D", lambda: analyses.compute_sync(cs4, "plv"))
# compute_sync fed (n_epochs, n_freq, 2*n_ch, n_times) internal layout by mistake
cs_int = rng.standard_normal((3, 1, 8, 512)) + 1j * rng.standard_normal((3, 1, 8, 512))
attempt("compute_sync internal 4-D", lambda: analyses.compute_sync(cs_int, "plv"))
# compute_sync with 3 "participants"
cs3 = rng.standard_normal((3, 3, 4, 1, 512)) + 1j
attempt("compute_sync 3 participants", lambda: analyses.compute_sync(cs3, "plv"))

# 2 participants, 1 time... different dtype: real input to compute_sync
attempt("compute_sync real input", lambda: analyses.compute_sync(rng.standard_normal((2, 3, 4, 1, 512)), "plv"))

print("\n=== frequency range inclusivity")
out = analyses.compute_single_freq(good, fs, [8, 12])
print("compute_single_freq [8,12] -> n_freq =", out.shape[4], "(docstring: 'every integer frequency from fmin to fmax')")
print("n_tapers =", out.shape[3])

print("\n=== multitaper effective bandwidth at 10 Hz bin (n_cycles=4, time_bandwidth=4 default)")
t = np.arange(fs * 4) / fs
for f_tone in [6, 8, 10, 12, 14, 16, 20]:
    d = np.zeros((2, 1, 1, t.size))
    d[:, 0, 0] = np.cos(2 * np.pi * f_tone * t)
    tf = analyses.compute_single_freq(d, fs, [10, 11])  # single bin 10 Hz
    pw = np.mean(np.abs(np.mean(tf, 3)[0, 0, 0, 0, fs:-fs]) ** 2)
    print(f"tone {f_tone:2d} Hz -> power at 10 Hz bin (rel.) {pw:.3f}")

print("\n=== taper averaging: complex mean across tapers")
d = np.zeros((2, 1, 1, t.size))
d[0, 0, 0] = np.cos(2 * np.pi * 10 * t)
d[1, 0, 0] = np.cos(2 * np.pi * 10 * t - np.pi / 2)  # p2 lags by 90 deg
tf = analyses.compute_single_freq(d, fs, [10, 11])
per_taper = np.angle(tf[0, 0, 0, :, 0, fs:-fs] * np.conj(tf[1, 0, 0, :, 0, fs:-fs])).mean(-1)
avg = np.angle((np.mean(tf, 3)[0, 0, 0, 0, fs:-fs]) * np.conj(np.mean(tf, 3)[1, 0, 0, 0, fs:-fs])).mean()
print("phase diff per taper (deg):", np.round(np.degrees(per_taper), 2), " after complex taper mean:", np.round(np.degrees(avg), 2))
amp_t = np.abs(tf[0, 0, 0, :, 0, fs:-fs]).mean(-1)
print("amplitude per taper:", np.round(amp_t, 3), " amp of complex taper mean:", np.round(np.abs(np.mean(tf, 3)[0, 0, 0, 0, fs:-fs]).mean(), 3),
      " sqrt(mean power over tapers):", np.round(np.sqrt((amp_t**2).mean()), 3))

print("\n=== compute_freq_bands on short epochs: filter longer than epoch?")
short = rng.standard_normal((2, 3, 4, 256))  # 1 s epochs
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cs = analyses.compute_freq_bands(short, fs, {"alpha": [8, 12]})
    print("warnings raised:", [str(x.message)[:100] for x in w])
print("shape:", cs.shape)
