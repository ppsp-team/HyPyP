"""(a) children/child_head: do outline and sensors shrink the same head? (b) negative linewidth rendering. (c) mesh faces."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne, meshio, importlib_resources
from hypyp import viz

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=False)
e2 = mne.read_epochs("tests/data/preproc_S2.fif", preload=False)
n = len(e1.ch_names)
plt.show = lambda *a, **k: None
OUT = "/home/gdumas/.claude/jobs/9fe7d438/tmp/viz/"

print("(a) 2D topomap inter, children=True: outline radius vs sensor-cloud radius per head")
for ch in (True, False):
    ax = viz.viz_2D_topomap_inter(e1, e2, np.zeros((n, n)), threshold=0.5, children=True, child_head=ch)
    circles = [p for p in ax.patches if isinstance(p, matplotlib.patches.Circle)]
    outline = {("left" if c.center[0] < 0 else "right"): c.radius for c in circles}
    pts = np.array([[l.get_xdata()[0], l.get_ydata()[0]] for l in ax.lines if l.get_marker() in ("o", "x")])
    left, right = pts[pts[:, 0] < 0], pts[pts[:, 0] > 0]
    rl = np.linalg.norm(left - left.mean(0), axis=1).max(); rr = np.linalg.norm(right - right.mean(0), axis=1).max()
    print(f"   child_head={ch!s:5s}: outline radius left={outline['left']:.3f} right={outline['right']:.3f} | sensor cloud radius left={rl:.3f} right={rr:.3f}"
          f"  -> {'MISMATCH' if (outline['left'] < outline['right']) != (rl < rr) else 'consistent'}")
    ax.figure.savefig(OUT + f"children_2d_childhead_{ch}.png", dpi=70); plt.close("all")

print("(b) 3D inter, children=True: head-mesh extent vs sensor extent per head (plot-x < 0 is head 1)")
for ch in (True, False):
    v, f = viz.get_3d_heads_inter(children=True, child_head=ch)
    x, y, z = v[:, 2], v[:, 0], v[:, 1]  # plot_3d_heads axis permutation
    h1, h2 = v[x < 0], v[x > 0]
    ext1 = np.ptp(h1[:, 0]); ext2 = np.ptp(h2[:, 0])
    raw1 = np.array([c["loc"][:3] for c in e1.info["chs"]]); raw2 = np.array([c["loc"][:3] for c in e2.info["chs"]])
    L1 = viz.transform(raw1.copy(), traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi / 12), rotZ=(-np.pi / 2), children=True, child_head=not ch)
    L2 = viz.transform(raw2.copy(), traX=0.17, traY=0, traZ=0.08, rotY=(np.pi / 12), rotZ=np.pi / 2, children=True, child_head=ch)
    print(f"   child_head={ch!s:5s}: head1 mesh width={ext1:.3f} head2 mesh width={ext2:.3f} | sensors P1 width={np.ptp(L1[:,1]):.3f} P2 width={np.ptp(L2[:,1]):.3f}"
          f"  -> {'MISMATCH' if (ext1 < ext2) != (np.ptp(L1[:,1]) < np.ptp(L2[:,1])) or (abs(ext1-ext2) < 1e-6) != (abs(np.ptp(L1[:,1]) - np.ptp(L2[:,1])) < 1e-3) else 'consistent'}")
    ax = viz.viz_3D_inter(e1, e2, np.zeros((n, n)), threshold=0.5, children=True, child_head=ch)
    ax.figure.savefig(OUT + f"children_3d_childhead_{ch}.png", dpi=70); plt.close("all")

print("(c) negative linewidth rendering under Agg")
for lw in (2.0, -1.08, -2.76):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50); ax.axis("off")
    ax.plot([0, 1], [0, 1], color="red", linewidth=lw)
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())
    print(f"   linewidth={lw:+.2f}: red pixels drawn={int(((img[..., 0] > 200) & (img[..., 1] < 100)).sum())}")
    plt.close(fig)

print("(d) Basehead.obj cells and the 4th wireframe edge")
ref = importlib_resources.files("hypyp") / "data/Basehead.obj"
m = meshio.read(str(ref))
print(f"   cell blocks: {[(c.type, c.data.shape) for c in m.cells]}; points {m.points.shape}")
print("   plot_3d_heads draws edges V0-V1, V1-V2, V2-V3, V3-V1 (viz.py:1995) -> V3-V0 edge never drawn, V1-V3 diagonal drawn instead")

print("(e) NaN with a negative link: colour passed to matplotlib")
C = np.zeros((n, n)); C[0, 1] = -0.9; C[1, 2] = 0.9; C[5, 5] = np.nan
fig, ax = plt.subplots()
try:
    viz.plot_links_2d_inter(e1, e2, C, threshold=0.5, steps=2)
    print("   colours:", [tuple(round(float(c), 2) for c in l.get_color()[:3]) for l in ax.lines if l.get_marker() == "None"])
except Exception as ex:
    print("   ", type(ex).__name__, str(ex)[:100])
plt.close("all")
