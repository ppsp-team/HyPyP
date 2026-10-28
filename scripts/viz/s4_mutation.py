"""Does any viz function mutate the caller's Epochs / Info / arrays?"""
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
C = np.random.default_rng(0).uniform(0, 1, (n, n))

def snap(e):
    return (np.array([ch["loc"].copy() for ch in e.info["chs"]]), list(e.info["bads"]), e.get_montage().get_positions()["ch_pos"]["Fp1"].copy())

b1, b2 = snap(e1), snap(e2)
C0 = C.copy()
plt.show = lambda *a, **k: None  # neutralise blocking show
for name, call in [
    ("viz_2D_topomap_inter", lambda: viz.viz_2D_topomap_inter(e1, e2, C, threshold=0.9, steps=3)),
    ("viz_2D_topomap_intra", lambda: viz.viz_2D_topomap_intra(e1, e2, C, C, threshold=0.9, steps=2)),
    ("viz_3D_inter", lambda: viz.viz_3D_inter(e1, e2, C, threshold=0.9, steps=3)),
    ("viz_3D_intra", lambda: viz.viz_3D_intra(e1, e2, C, C, threshold=0.9, steps=3)),
    ("viz_2D_headmodel_inter", lambda: viz.viz_2D_headmodel_inter(e1, e2, C, threshold=0.9, steps=3)),
    ("plot_significant_sensors", lambda: viz.plot_significant_sensors(np.r_[np.zeros(n - 3), [2.0, -2.0, 1.0]], e1, significant=np.r_[np.zeros(n - 3), [1, 1, 1]])),
]:
    call()
    a1, a2 = snap(e1), snap(e2)
    same = all(np.allclose(x, y) for x, y in zip(b1[0::2], a1[0::2])) and b1[1] == a1[1] and \
           all(np.allclose(x, y) for x, y in zip(b2[0::2], a2[0::2])) and b2[1] == a2[1]
    print(f"{name:26s} Epochs/Info/montage unchanged: {same}; C unchanged: {np.array_equal(C, C0)}")
    plt.close("all")

# transform() on a user array
locs = np.array([ch["loc"][:3] for ch in e1.info["chs"]])
locs_before = locs.copy()
out = viz.transform(locs, traX=-0.17, rotZ=-np.pi / 2)
print(f"\ntransform(locs): input array mutated in place: {not np.array_equal(locs, locs_before)}; returns same object: {out is locs}")
locs = np.array([ch["loc"][:3] for ch in e1.info["chs"]]); locs_before = locs.copy()
out = viz.transform_2d_intra(locs, traX=-0.178)
print(f"transform_2d_intra(locs): input mutated: {not np.array_equal(locs, locs_before)}; returns same object: {out is locs}")

# int array input to transform (common if user builds positions by hand)
li = np.array([[1, 0, 0], [0, 1, 0]])
try:
    viz.transform(li.copy()); print("transform(int array): ok")
except Exception as ex:
    print(f"transform(int array): {type(ex).__name__}: {str(ex)[:90]}")
