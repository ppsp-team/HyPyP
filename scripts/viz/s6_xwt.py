"""plot_xwt: y tick labels for non-linear freqs, tmin units, output shape."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings, io, contextlib
import mne
from hypyp import viz
from hypyp.analyses import xwt

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=True)
e2 = mne.read_epochs("tests/data/preproc_S2.fif", preload=True)
sf = e1.info["sfreq"]
T = e1.times[-1] - e1.times[0]
print(f"epoch length {T:.3f}s, sfreq {sf}, n_times {len(e1.times)}")

# What does xwt return? plot_xwt passes it straight to imshow
freqs = np.array([4.0, 8.0, 16.0, 32.0])  # log-spaced
with contextlib.redirect_stdout(io.StringIO()):
    d = xwt(e1.copy().pick(["Fp1", "Fp2"]), e2.copy().pick(["Fp1", "Fp2"]), freqs, mode="wtc")
print("xwt(... mode='wtc') returns shape", d.shape, "-> imshow of a", d.ndim, "D array")

s1 = e1.copy().pick(["Fp1"]); s2 = e2.copy().pick(["Fp1"])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fig = viz.plot_xwt(s1, s2, sfreq=sf, freqs=freqs, time=T, analysis="wtc", figsize=(10, 4))
        ax = fig.axes[0]
        print("plot_xwt(log-spaced freqs 4,8,16,32): y tick labels =", [t.get_text() for t in ax.get_yticklabels()], "(expected 4,8,16,32)")
        print("  x tick count", len(ax.get_xticks()), "xlim", ax.get_xlim(), "(xmax is in samples:", T * sf, ")")
        print("  stdout:", buf.getvalue().strip()[:120])
        fig.savefig("/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/xwt.png", dpi=60)
    except Exception as ex:
        print("plot_xwt error:", type(ex).__name__, str(ex)[:200])
    print("  warnings:", sorted({f'{x.category.__name__}: {str(x.message)[:80]}' for x in w}))
plt.close("all")

# tmin != 0
try:
    with contextlib.redirect_stdout(io.StringIO()):
        fig = viz.plot_xwt(s1, s2, sfreq=sf, freqs=np.arange(4, 20, 4.0), time=T, analysis="power", tmin=1.0, figsize=(10, 4))
    ax = fig.axes[0]
    print(f"tmin=1.0: xlim={ax.get_xlim()} (xmin computed as tmin*dt={1.0/sf:.4f} samples instead of tmin/dt={sf:.0f})")
except Exception as ex:
    print("plot_xwt tmin=1.0 error:", type(ex).__name__, str(ex)[:200])
plt.close("all")

# the error mode: xticks vs labels length mismatch
for time_, xu in [(T, 100), (T, 7), (3.0, 7), (T, 33)]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fig = viz.plot_xwt(s1, s2, sfreq=sf, freqs=np.arange(4, 20, 4.0), time=time_, analysis="power", x_units=xu, figsize=(10, 4))
        print(f"time={time_:.3f}, x_units={xu}: ok")
    except Exception as ex:
        print(f"time={time_:.3f}, x_units={xu}: {type(ex).__name__}: {str(ex)[:100]}")
    plt.close("all")
