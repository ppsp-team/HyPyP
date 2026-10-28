"""Run every public entry point, capture warnings, count open figures, time it, save PNGs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings, time, atexit
import mne
from hypyp import viz

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=False)
e2 = mne.read_epochs("tests/data/preproc_S2.fif", preload=False)
n = len(e1.ch_names)
rng = np.random.default_rng(0)
C = rng.uniform(-1, 1, (n, n))
OUT = "/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/"

shows = []
plt.show = lambda *a, **k: shows.append(1)
calls = [
    ("viz_2D_topomap_inter", lambda: viz.viz_2D_topomap_inter(e1, e2, C, threshold=0.8, steps=10, lab=True)),
    ("viz_2D_topomap_intra", lambda: viz.viz_2D_topomap_intra(e1, e2, C, C, threshold=0.8, steps=2, lab=True)),
    ("viz_3D_inter", lambda: viz.viz_3D_inter(e1, e2, C, threshold=0.8, steps=10, lab=True)),
    ("viz_3D_intra", lambda: viz.viz_3D_intra(e1, e2, C, C, threshold=0.8, steps=10)),
    ("viz_2D_headmodel_inter", lambda: viz.viz_2D_headmodel_inter(e1, e2, C, threshold=0.8, steps=10, lab=True)),
    ("viz_2D_topomap_inter(children)", lambda: viz.viz_2D_topomap_inter(e1, e2, C, threshold=0.8, steps=10, children=True, child_head=True)),
    ("viz_3D_inter(children)", lambda: viz.viz_3D_inter(e1, e2, C, threshold=0.8, steps=10, children=True, child_head=True)),
    ("plot_significant_sensors(mask)", lambda: viz.plot_significant_sensors(np.r_[np.zeros(n - 3), [2.0, -2.0, 1.0]], e1, significant=np.r_[np.zeros(n - 3), [1, 1, 1]])),
    ("plot_significant_sensors(nomask)", lambda: viz.plot_significant_sensors(np.r_[np.zeros(n - 3), [2.0, -2.0, 1.0]], e1)),
]
n_atexit_before = len(atexit._ncallbacks() and [0] * atexit._ncallbacks())
for name, call in calls:
    before = plt.get_fignums()
    shows.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        ret = call()
        dt = time.perf_counter() - t0
    after = plt.get_fignums()
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w})
    print(f"{name:34s} {dt:5.1f}s  returns={type(ret).__name__:10s} new_figs={len(after) - len(before)} plt.show_calls={len(shows)} warnings={len(w)}")
    for m in msgs:
        print("      ", m)
    if after:
        plt.figure(after[-1]).savefig(OUT + name.replace("(", "_").replace(")", "") + ".png", dpi=70)
print(f"\nopen figures after all calls (never closed by library): {len(plt.get_fignums())}")
print(f"atexit callbacks registered: {atexit._ncallbacks()} (one per get_3d_heads_* call)")

# figure size: hardcoded?
fig = plt.figure(1)
print("figsize of viz_2D_topomap_inter figure:", fig.get_size_inches(), "(rcParams default", matplotlib.rcParams["figure.figsize"], ")")

# plt.plot on current axes: links land on the wrong subplot if the user has a grid
plt.close("all")
fig, axes = plt.subplots(1, 2)
viz.plot_2d_topomap_inter(axes[0])
viz.plot_sensors_2d_inter(e1, e2, lab=False)
Cs = np.zeros((n, n)); Cs[0, 1] = 1
viz.plot_links_2d_inter(e1, e2, Cs, threshold=0.5, steps=2)
print(f"\nsubplot grid: head outline patches on axes[0]={len(axes[0].patches)}, sensors+links on axes[0]={len(axes[0].lines)}, on axes[1]={len(axes[1].lines)}")
fig.savefig(OUT + "subplot_grid_mismatch.png", dpi=70)
plt.close("all")
