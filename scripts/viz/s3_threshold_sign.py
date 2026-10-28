"""threshold semantics, 'auto', negative values, all-negative matrices, NaN."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mne
from hypyp import viz

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=False)
e2 = mne.read_epochs("tests/data/preproc_S2.fif", preload=False)
n = len(e1.ch_names)

def drawn(ax):
    return [ln for ln in ax.get_lines() if ln.get_marker() == "None"]

# 1. threshold is absolute: 0.95 on a matrix in [0,0.5] draws nothing
C = np.random.default_rng(0).uniform(0, 0.5, (n, n))
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold=0.95, steps=2)
print(f"1. C in [0,0.5], threshold=0.95 -> {len(drawn(ax))} links (absolute, not percentile)")
plt.close(fig)

# 2. 'auto' value
print(f"2. 'auto' = max(median(C,0)) + max(std(C,0)) = {np.max(np.median(C, 0)) + np.max(np.std(C, 0)):.3f} (docstring says 'standard error')")

# 3. signed values: imcoh-like, both signs
C = np.zeros((n, n)); C[0, 1] = 0.9; C[2, 3] = -0.9
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
for ln in drawn(ax):
    print(f"3. signed: link color={ln.get_color()[:3]} lw={ln.get_linewidth():.2f}")
plt.close(fig)

# 4. all-negative matrix (e.g. imcoh with one sign dominant)
C = np.zeros((n, n)); C[0, 1] = -0.9; C[2, 3] = -0.6
fig, ax = plt.subplots()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
        print(f"4. all-negative C (max=0): drawn {len(drawn(ax))}, linewidths={[round(l.get_linewidth(),3) for l in drawn(ax)]}, warnings={[str(x.message)[:60] for x in w]}")
        fig.savefig("/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/neg_links.png")
    except Exception as ex:
        print(f"4. all-negative C: {type(ex).__name__}: {ex}")
plt.close(fig)
C = -np.abs(np.random.default_rng(1).normal(0.3, 0.3, (n, n)))  # strictly negative, max < 0
fig, ax = plt.subplots()
try:
    viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
    lws = [l.get_linewidth() for l in drawn(ax)]
    print(f"4b. strictly negative C (max={C.max():.3f}): drawn {len(lws)}, lw range [{min(lws):.3f}, {max(lws):.3f}]")
    fig.savefig("/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/neg_links_b.png")
except Exception as ex:
    print(f"4b. strictly negative C: {type(ex).__name__}: {ex}")
plt.close(fig)

# 5. NaN (bad channel) with 'auto' and with a float threshold
C = np.random.default_rng(2).uniform(0.5, 1.0, (n, n)); C[5, :] = np.nan
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold="auto", steps=2)
print(f"5a. one NaN row, threshold='auto': auto threshold={np.max(np.median(C, 0)) + np.max(np.std(C, 0))}, drawn {len(drawn(ax))} (of {np.isfinite(C).sum()} finite entries)")
plt.close(fig)
fig, ax = plt.subplots()
try:
    viz.plot_links_2d_inter(e1, e2, C, threshold=0.6, steps=2)
    print(f"5b. one NaN row, threshold=0.6: drawn {len(drawn(ax))}, norm_n vmin={np.min(C[:])}")
except Exception as ex:
    print(f"5b. one NaN row, threshold=0.6: {type(ex).__name__}: {ex}")
plt.close(fig)

# 6. threshold=0 draws zeros too
C = np.zeros((n, n)); C[0, 1] = 0.3
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold=0, steps=2)
print(f"6. threshold=0 on mostly-zero C: drawn {len(drawn(ax))} links (n*n={n*n})")
plt.close(fig)

# 7. threshold > max: Normalize vmin>vmax?
C = np.random.default_rng(3).uniform(0, 0.5, (n, n))
fig, ax = plt.subplots()
try:
    viz.plot_links_2d_inter(e1, e2, C, threshold=0.9, steps=2); print(f"7. threshold>max: ok, drawn {len(drawn(ax))}")
except Exception as ex:
    print(f"7. threshold>max: {type(ex).__name__}: {ex}")
plt.close(fig)

# 8. intra 'auto' on a symmetric matrix with diag=1: threshold value?
C1 = np.random.default_rng(4).uniform(0, 0.3, (n, n)); C1 = (C1 + C1.T) / 2; np.fill_diagonal(C1, 1.0)
thr = np.max([np.median(C1, 0), np.median(C1, 0)]) + np.max([np.std(C1, 0), np.std(C1, 0)])
fig, ax = plt.subplots(); viz.plot_links_2d_intra(e1, e2, C1, C1, threshold="auto", steps=10)
print(f"8. intra 'auto' with diag=1 (PLV style): threshold={thr:.3f}, off-diag max={C1[~np.eye(n, dtype=bool)].max():.3f}, segments drawn={len(drawn(ax))} (31 self-loops x 9 segs x 2 participants = {31*9*2})")
fig.savefig("/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/intra_diag_auto.png", dpi=80)
plt.close(fig)
