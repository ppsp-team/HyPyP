"""Which two sensors does C[i, j] join? 2D and 3D, inter and intra; 2n x 2n misread; diagonal; asymmetry."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
from hypyp import viz

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=False)
e2 = mne.read_epochs("tests/data/preproc_S2.fif", preload=False)
n = len(e1.ch_names)
raw1 = np.array([ch["loc"][:3] for ch in e1.info["chs"]])
raw2 = np.array([ch["loc"][:3] for ch in e2.info["chs"]])
L1 = viz.transform(raw1.copy(), traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi / 12), rotZ=(-np.pi / 2))
L2 = viz.transform(raw2.copy(), traX=0.17, traY=0, traZ=0.08, rotY=(np.pi / 12), rotZ=np.pi / 2)
I1 = viz.transform_2d_intra(raw1.copy(), traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi / 2))
I2 = viz.transform_2d_intra(raw2.copy(), traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi / 2))
K1 = viz.transform(raw1.copy(), traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi / 2))
K2 = viz.transform(raw2.copy(), traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi / 2))

def nearest(pt, locs_dict):
    best = None
    for part, (locs, names) in locs_dict.items():
        d = np.linalg.norm(locs[:, :len(pt)] - pt, axis=1)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], part, names[k], k)
    return best

def endpoints_2d(ax):
    out = []
    for ln in ax.get_lines():
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) == 2 and ln.get_marker() == "None":
            out.append((np.array([x[0], y[0]]), np.array([x[1], y[1]]), ln.get_linewidth(), ln.get_color()))
    return out

def endpoints_3d(ax):
    out = []
    for ln in ax.get_lines():
        if hasattr(ln, "_verts3d"):
            x, y, z = ln._verts3d
            if len(x) == 2 and ln.get_marker() == "None":
                out.append((np.array([x[0], y[0], z[0]]), np.array([x[1], y[1], z[1]]), ln.get_linewidth(), ln.get_color()))
    return out

i, j = e1.ch_names.index("Fp1"), e2.ch_names.index("O2")
print(f"Test entry: i={i} ({e1.ch_names[i]}), j={j} ({e2.ch_names[j]}), n={n}")

# --- A. (n x n) inter matrix, 2D ---
C = np.zeros((n, n)); C[i, j] = 1.0
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
segs = endpoints_2d(ax)
print(f"\nA. plot_links_2d_inter, C shape (n,n), C[{i},{j}]=1, steps=2: {len(segs)} segment(s) drawn")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3], f"lw={w:.2f}")
plt.close(fig)

# --- B. same, 3D ---
fig = plt.figure(); ax = fig.add_subplot(projection="3d"); viz.plot_links_3d_inter(ax, e1, e2, C, threshold=0.5, steps=2)
segs = endpoints_3d(ax)
print(f"B. plot_links_3d_inter, same C, steps=2: {len(segs)} segment(s)")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3], f"lw={w:.2f}")
plt.close(fig)

# --- C. full (2n x 2n) matrix with the entry in the INTER quadrant ---
Cfull = np.zeros((2 * n, 2 * n)); Cfull[i, n + j] = 1.0
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, Cfull, threshold=0.5, steps=2)
print(f"\nC. plot_links_2d_inter with FULL (2n,2n) matrix, Cfull[{i},{n + j}]=1 (inter quadrant): {len(endpoints_2d(ax))} segment(s) drawn, no error raised")
plt.close(fig)
# --- D. full matrix with entry in the INTRA quadrant of P1 ---
Cfull = np.zeros((2 * n, 2 * n)); Cfull[i, j] = 1.0
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, Cfull, threshold=0.5, steps=2)
segs = endpoints_2d(ax)
print(f"D. plot_links_2d_inter with FULL (2n,2n) matrix, Cfull[{i},{j}]=1 (P1 INTRA quadrant): {len(segs)} segment(s) drawn")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3])
plt.close(fig)
# --- D2. (n, 2n) i.e. result[0:n, :] (forgot to slice columns) ---
Chalf = np.zeros((n, 2 * n)); Chalf[i, n + j] = 1.0
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, Chalf, threshold=0.5, steps=2)
print(f"D2. plot_links_2d_inter with (n,2n) matrix, entry at [{i},{n + j}]: {len(endpoints_2d(ax))} segment(s), no error")
plt.close(fig)

# --- E. asymmetric inter matrix: C[i,j] and C[j,i] ---
C = np.zeros((n, n)); C[i, j] = 1.0; C[j, i] = 0.8
fig, ax = plt.subplots(); viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
segs = endpoints_2d(ax)
print(f"\nE. inter, C[{i},{j}]=1 and C[{j},{i}]=0.8: {len(segs)} segments")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (L1, e1.ch_names), "P2": (L2, e2.ch_names)})[1:3], f"lw={w:.2f}")
plt.close(fig)

# --- F. intra: symmetric matrix drawn twice? diagonal? ---
C1 = np.zeros((n, n)); C1[i, j] = C1[j, i] = 1.0
C2 = np.zeros((n, n))
fig, ax = plt.subplots(); viz.plot_links_2d_intra(e1, e2, C1, C2, threshold=0.5, steps=2)
segs = endpoints_2d(ax)
print(f"\nF. plot_links_2d_intra, symmetric C1[{i},{j}]=C1[{j},{i}]=1, C2=0, steps=2: {len(segs)} segment(s) (expect 1 if dedup'd)")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (I1, e1.ch_names), "P2": (I2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (I1, e1.ch_names), "P2": (I2, e2.ch_names)})[1:3])
plt.close(fig)

# --- G. diagonal = 1 (PLV/coh style), everything else 0 ---
C1 = np.eye(n); C2 = np.zeros((n, n))
for steps in (2, 10):
    fig, ax = plt.subplots(); viz.plot_links_2d_intra(e1, e2, C1, C2, threshold=0.5, steps=steps)
    lines = [ln for ln in ax.get_lines() if ln.get_marker() == "None"]
    lengths = [np.hypot(np.diff(ln.get_xdata())[0], np.diff(ln.get_ydata())[0]) for ln in lines]
    # how far does the self-loop reach from the sensor?
    print(f"G. plot_links_2d_intra, C1=eye(n), steps={steps}: {len(lines)} line segments drawn; "
          f"max segment length={max(lengths) if lengths else 0:.4f}")
    if steps == 10:
        # self-link of Fp1: furthest drawn point from Fp1
        pts = np.concatenate([np.c_[ln.get_xdata(), ln.get_ydata()] for ln in lines])
        d = np.linalg.norm(pts - I1[i, :2], axis=1)
        ctr = np.nanmean(I1, 0)
        print(f"   self-loop of Fp1 extends up to {d[d < 0.2].max():.3f} from the sensor "
              f"(sensor-to-head-centre distance {np.linalg.norm(I1[i,:2]-ctr[:2]):.3f}; head radius 0.1)")
    fig.savefig(f"/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/G_selfloops_steps{steps}.png", dpi=80)
    plt.close(fig)

# --- H. 3D intra mapping ---
C1 = np.zeros((n, n)); C1[i, j] = 1.0; C2 = np.zeros((n, n)); C2[j, i] = 1.0
fig = plt.figure(); ax = fig.add_subplot(projection="3d"); viz.plot_links_3d_intra(ax, e1, e2, C1, C2, threshold=0.5, steps=2)
segs = endpoints_3d(ax)
print(f"\nH. plot_links_3d_intra C1[{i},{j}]=1, C2[{j},{i}]=1, steps=2: {len(segs)} segments")
for p, q, w, col in segs:
    print("   from", nearest(p, {"P1": (K1, e1.ch_names), "P2": (K2, e2.ch_names)})[1:3],
          "to", nearest(q, {"P1": (K1, e1.ch_names), "P2": (K2, e2.ch_names)})[1:3])
plt.close(fig)

# --- I. wrong-size matrix (n-1): silently truncates or IndexError? ---
for shape in [(n - 1, n - 1), (n + 1, n + 1), (n, n - 1)]:
    Cw = np.zeros(shape); Cw[0, 0] = 1
    fig, ax = plt.subplots()
    try:
        viz.plot_links_2d_inter(e1, e2, Cw, threshold=0.5, steps=2)
        print(f"I. C shape {shape} vs n={n}: no error, {len(endpoints_2d(ax))} segment(s)")
    except Exception as ex:
        print(f"I. C shape {shape} vs n={n}: {type(ex).__name__}: {str(ex)[:80]}")
    plt.close(fig)
