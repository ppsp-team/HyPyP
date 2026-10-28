"""Where do Fp1 / T7 / T8 / O1 land for each participant, and is handedness preserved?"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import mne
from hypyp import viz

mne.set_log_level("ERROR")
e1 = mne.read_epochs("tests/data/preproc_S1.fif", preload=False)
names = e1.ch_names
raw = np.array([ch["loc"][:3] for ch in e1.info["chs"]])
idx = {n: names.index(n) for n in ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Cz"]}

def signed_area(p_nose, p_left, p_right):
    a = p_left[:2] - p_nose[:2]
    b = p_right[:2] - p_nose[:2]
    return float(a[0] * b[1] - a[1] * b[0])

print("RAW montage (MNE head frame: x=right, y=anterior, z=up), xy only:")
for n in ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Cz"]:
    print(f"  {n:4s} x={raw[idx[n],0]:+.3f} y={raw[idx[n],1]:+.3f} z={raw[idx[n],2]:+.3f}")
print("  chirality raw (Fp1,T7,T8) signed area:", f"{signed_area(raw[idx['Fp1']], raw[idx['T7']], raw[idx['T8']]):+.5f}")

configs = {
    "2D/3D inter P1": dict(traX=-0.17, traY=0, traZ=0.08, rotY=(-np.pi / 12), rotZ=(-np.pi / 2)),
    "2D/3D inter P2": dict(traX=0.17, traY=0, traZ=0.08, rotY=(np.pi / 12), rotZ=np.pi / 2),
    "3D intra P1":    dict(traX=0, traY=0, traZ=0.04, rotY=0, rotZ=(-np.pi / 2)),
    "3D intra P2":    dict(traX=0, traY=0.5, traZ=0.04, rotY=0, rotZ=(-np.pi / 2)),
}
for label, kw in configs.items():
    L = viz.transform(raw.copy(), **kw)
    print(f"\n{label}: transform(**{kw})")
    for n in ["Fp1", "Fp2", "T7", "T8", "O1", "O2", "Cz"]:
        print(f"  {n:4s} X={L[idx[n],0]:+.3f} Y={L[idx[n],1]:+.3f} Z={L[idx[n],2]:+.3f}")
    nose_dir = (L[idx["Fp1"]] + L[idx["Fp2"]]) / 2 - (L[idx["O1"]] + L[idx["O2"]]) / 2
    print(f"  nose direction (Fp - O) in XY: ({nose_dir[0]:+.3f}, {nose_dir[1]:+.3f})")
    print(f"  chirality (Fp1,T7,T8) signed area: {signed_area(L[idx['Fp1']], L[idx['T7']], L[idx['T8']]):+.5f}"
          f"  -> {'same as raw (proper rotation)' if np.sign(signed_area(L[idx['Fp1']], L[idx['T7']], L[idx['T8']])) == np.sign(signed_area(raw[idx['Fp1']], raw[idx['T7']], raw[idx['T8']])) else 'FLIPPED (reflection)'}")
    print(f"  T7 (left ear) Y={L[idx['T7'],1]:+.3f}   T8 (right ear) Y={L[idx['T8'],1]:+.3f}")

print("\n2D intra P1/P2 (transform_2d_intra):")
for label, kw in {"P1": dict(traX=-0.178, traY=0.012, traZ=0, rotZ=(-np.pi / 2)),
                  "P2": dict(traX=0.178, traY=0.012, traZ=0, rotZ=(-np.pi / 2))}.items():
    L = viz.transform_2d_intra(raw.copy(), **kw)
    nose_dir = (L[idx["Fp1"]] + L[idx["Fp2"]]) / 2 - (L[idx["O1"]] + L[idx["O2"]]) / 2
    print(f"  {label}: Fp1 X={L[idx['Fp1'],0]:+.3f} Y={L[idx['Fp1'],1]:+.3f}; T7 X={L[idx['T7'],0]:+.3f}; T8 X={L[idx['T8'],0]:+.3f}; nose dir XY=({nose_dir[0]:+.3f},{nose_dir[1]:+.3f}); "
          f"chirality {signed_area(L[idx['Fp1']], L[idx['T7']], L[idx['T8']]):+.5f}")

# Does the head outline in plot_2d_topomap_inter agree with sensor radius?
L1 = viz.transform(raw.copy(), **configs["2D/3D inter P1"])
r = np.linalg.norm(L1[:, :2] - np.array([-0.178, 0]), axis=1)
print(f"\nInter 2D: sensor distance from drawn head centre (-0.178,0): min={r.min():.3f} max={r.max():.3f} (head radius drawn = 0.100)")
print(f"  sensors outside circle: {(r > 0.1).sum()} / {len(r)}")
