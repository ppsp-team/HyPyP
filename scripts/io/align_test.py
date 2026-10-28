"""Reproduce XDF alignment behaviour on synthetic two-stream files."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from xdfwrite import write_xdf  # noqa: E402

from hypyp.xdf import XDFImport  # noqa: E402
from hypyp.xdf.xdf_stream import Markers  # noqa: E402

warnings.simplefilter("ignore")
TMP = os.path.dirname(__file__)
rng = np.random.default_rng(0)
T0 = 1000.0


def eeg_stream(sid, name, t_start, n, srate_true, clock_offset, labels=("Ch1", "Ch2", "Ch3", "Ch4"), jitter=5e-4, gap=None):
    true_t = t_start + np.arange(n) / srate_true
    if gap is not None:  # drop samples whose true time is inside [gap0, gap1)
        keep = ~((true_t >= gap[0]) & (true_t < gap[1]))
        true_t = true_t[keep]
    idx = np.arange(len(true_t))
    vals = np.column_stack([idx, true_t - T0, rng.normal(0, 20, len(idx)), rng.normal(0, 20, len(idx))])
    raw_ts = true_t - clock_offset + rng.normal(0, jitter, len(true_t))
    clock = [(T0 + k * 5.0, clock_offset) for k in range(4)]
    return dict(id=sid, name=name, type="EEG", srate=100, labels=list(labels), timestamps=raw_ts, values=vals, clock=clock)


def marker_stream(sid, name, times, descs):
    return dict(id=sid, name=name, type="Markers", srate=0, nch=1, fmt="string",
                timestamps=np.array(times), values=[[d] for d in descs], clock=[(T0 + k * 5.0, 0.0) for k in range(4)])


print("=== A. two EEG streams, stream 2 starts 0.5 s later with a +10 s clock offset; marker at T0+2.0 ===")
path = write_xdf(f"{TMP}/dyad_offset.xdf", [
    eeg_stream(1, "EEG1", T0, 1000, 100.0, 0.0),
    eeg_stream(2, "EEG2", T0 + 0.5, 1000, 100.0, 10.0),
    marker_stream(3, "Markers", [T0 + 2.0], ["go"]),
])
xdf = XDFImport(path)
for s in xdf.selected_signal_streams:
    r = s.mne_raw
    print(f"{s.name}: ref_time={s.reference_time:.4f} (true start {T0 if s.name=='EEG1' else T0+0.5}); "
          f"sfreq={r.info['sfreq']}, effective_srate={s.pyxdf_stream['info']['effective_srate']:.4f}, "
          f"n_times={r.n_times}, times[0]={r.times[0]}, annotations attached={len(r.annotations)}")
print("reference_time diff EEG2-EEG1 =", xdf.selected_signal_streams[1].reference_time - xdf.selected_signal_streams[0].reference_time)
m = xdf.markers
for s in xdf.selected_signal_streams:
    ann = m.as_mne_annotations(s.reference_time)
    print(f"  marker onset relative to {s.name}: {ann.onset[0]:.4f}  (expected {2.0 if s.name=='EEG1' else 1.5})")
# are the data arrays cropped to a common interval?
d1, d2 = xdf.mne_raws[0].get_data(), xdf.mne_raws[1].get_data()
print("sample 0 of EEG1 has true time", d1[1, 0], "; sample 0 of EEG2 has true time", d2[1, 0])
print("-> same sample index != same instant; nothing crops. Raw times both start at 0.")

print("\n=== B. 2 s gap in EEG1 (true 1005.0-1007.0 dropped), marker at T0+8.0 ===")
path = write_xdf(f"{TMP}/gap.xdf", [
    eeg_stream(1, "EEG1", T0, 1000, 100.0, 0.0, gap=(T0 + 5.0, T0 + 7.0)),
    marker_stream(3, "Markers", [T0 + 8.0], ["after_gap"]),
])
xdf = XDFImport(path)
s = xdf.selected_signal_streams[0]
r = s.mne_raw
print("segments:", s.pyxdf_stream["info"]["segments"], "n_times:", r.n_times, "raw duration:", r.times[-1])
ann = xdf.markers.as_mne_annotations(s.reference_time)
onset = ann.onset[0]
samp = int(round(onset * r.info["sfreq"]))
tt = r.get_data()[1]
true_idx = int(np.argmin(np.abs(tt - 8.0)))
print(f"marker onset={onset:.3f} s -> sample {samp} (Raw has {r.n_times}); the sample really recorded at T0+8.0 is index {true_idx} = Raw t={true_idx/100:.2f} s")
print("effective_srate reported by pyxdf:", s.pyxdf_stream["info"]["effective_srate"])

print("\n=== B2. 6 s gap in EEG1 (true 1005-1011 dropped; above the 5 s break threshold), marker at T0+12.0 ===")
path = write_xdf(f"{TMP}/gap6.xdf", [
    eeg_stream(1, "EEG1", T0, 1600, 100.0, 0.0, gap=(T0 + 5.0, T0 + 11.0)),
    marker_stream(3, "Markers", [T0 + 12.0], ["after_gap"]),
])
xdf = XDFImport(path)
s = xdf.selected_signal_streams[0]
r = s.mne_raw
ann = xdf.markers.as_mne_annotations(s.reference_time)
tt = r.get_data()[1]
true_idx = int(np.argmin(np.abs(tt - 12.0)))
print("segments:", s.pyxdf_stream["info"]["segments"], "n_times:", r.n_times, f"raw duration {r.times[-1]:.2f} s")
print(f"marker onset={ann.onset[0]:.3f} s; sample recorded at T0+12.0 is index {true_idx} = Raw t={true_idx/100:.2f} s -> off by {ann.onset[0]-true_idx/100:.2f} s")
r.set_annotations(ann)
print("after raw.set_annotations: n annotations kept =", len(r.annotations))

print("\n=== B3. real cached data_with_clock_resets.xdf ===")
from hypyp import datasets as _ds
xdf = XDFImport(_ds.xdf_clock_resets())
s = xdf.selected_signal_streams[0]
r = s.mne_raw
ann = xdf.markers.as_mne_annotations(s.reference_time)
print(f"nominal={r.info['sfreq']} effective={s.pyxdf_stream['info']['effective_srate']:.3f} segments={s.pyxdf_stream['info']['segments']} n_times={r.n_times} raw duration={r.times[-1]:.1f} s")
print(f"marker timestamps span {ann.onset[0]:.1f} .. {ann.onset[-1]:.1f} s relative to first EEG sample; {np.sum(ann.onset > r.times[-1])} of {len(ann)} markers fall after the Raw ends")

print("\n=== C. stream 2 really runs at 99.5 Hz (nominal 100) ===")
path = write_xdf(f"{TMP}/drift.xdf", [eeg_stream(2, "EEG2", T0, 1000, 99.5, 0.0)])
xdf = XDFImport(path)
s = xdf.selected_signal_streams[0]
print(f"nominal srate used by Raw: {s.mne_raw.info['sfreq']}, pyxdf effective_srate: {s.pyxdf_stream['info']['effective_srate']:.3f}")
print(f"last sample: Raw says t={s.mne_raw.times[-1]:.3f}, true t={s.mne_raw.get_data()[1, -1]:.3f}")

print("\n=== D. Markers.__add__ with two marker streams ===")
a = Markers(np.array([1.0, 2.0]), ["a1", "a2"])
b = Markers(np.array([100.0, 200.0]), ["b1", "b2"])
c = a + b
print("timestamps:", c.timestamps, "descriptions:", c.descriptions)
path = write_xdf(f"{TMP}/two_markers.xdf", [
    eeg_stream(1, "EEG1", T0, 300, 100.0, 0.0),
    marker_stream(3, "M1", [T0 + 0.5], ["m1"]),
    marker_stream(4, "M2", [T0 + 2.5], ["m2"]),
])
xdf = XDFImport(path)
print("XDFImport.markers merged:", xdf.markers.timestamps - T0, xdf.markers.descriptions)

print("\n=== E. two EEG streams with the same LSL name ===")
path = write_xdf(f"{TMP}/dup.xdf", [eeg_stream(1, "EEG", T0, 300, 100.0, 0.0), eeg_stream(2, "EEG", T0, 300, 100.0, 0.0)])
xdf = XDFImport(path)
print("len(mne_raws):", len(xdf.mne_raws), "len(mne_raws_dict):", len(xdf.mne_raws_dict), "unique names:", [s.unique_name for s in xdf.selected_signal_streams])
print("select_matches=['EEG'] picks index:", XDFImport(path, select_matches=["EEG"], convert_to_mne=False).selected_stream_indices)

print("\n=== F. time_series transpose heuristic: 4 channels, 4 samples ===")
path = write_xdf(f"{TMP}/square.xdf", [eeg_stream(1, "EEG1", T0, 4, 100.0, 0.0)])
xdf = XDFImport(path)
r = xdf.mne_raws[0]
print("raw shape:", r.get_data().shape, "channel 0 (should be 0,1,2,3 sample index):", r.get_data()[0])
path = write_xdf(f"{TMP}/short.xdf", [eeg_stream(1, "EEG1", T0, 3, 100.0, 0.0)])
try:
    XDFImport(path)
except Exception as e:
    print("3 samples x 4 channels ->", type(e).__name__, str(e)[:120])

print("\n=== G. scale='auto' on microvolt data (std 20 uV) ===")
path = write_xdf(f"{TMP}/scale.xdf", [eeg_stream(1, "EEG1", T0, 1000, 100.0, 0.0)])
xdf = XDFImport(path, scale="auto")
d = xdf.mne_raws[0].get_data()[2]
print(f"std after auto scaling = {d.std():.3e} V  (20 uV would be 2.0e-05 V)")

print("\n=== H. datasets.fetch(force_download=True) ===")
from hypyp import datasets  # noqa: E402


class Spy:
    def fetch(self, *a, **k):
        print("  _fetcher.fetch called with", a, k)
        return "x"


orig = datasets._fetcher
datasets._fetcher = Spy()
datasets.fetch("xdf/dyad-example-noise.xdf", force_download=True)
datasets._fetcher = orig

print("\n=== I. select_matches edge cases ===")
path = f"{TMP}/dyad_offset.xdf"
print("[2, 2] ->", XDFImport(path, select_matches=[2, 2], convert_to_mne=False).selected_stream_indices)
try:
    XDFImport(path, select_matches=[2.0], convert_to_mne=False)
except ValueError as e:
    print("[2.0] ->", e)
print("select_type='Markers' + select_matches=[1] ->", [s.type for s in XDFImport(path, select_type="Markers", select_matches=[1], convert_to_mne=False).selected_streams])

print("\n=== J. real cached file: are annotations attached? ===")
xdf = XDFImport(datasets.xdf_dyad_with_markers())
print("marker streams:", list(xdf.markers_dict), "n markers:", len(xdf.markers.timestamps))
print("annotations on raws:", [len(r.annotations) for r in xdf.mne_raws])

print("\n=== K. stream declared but no samples ===")
path = write_xdf(f"{TMP}/empty.xdf", [eeg_stream(1, "EEG1", T0, 300, 100.0, 0.0), dict(id=2, name="EEG2", type="EEG", srate=100, labels=["Ch1"], timestamps=np.array([]), values=np.zeros((0, 1)), clock=[]), marker_stream(3, "M", [T0 + 1], ["x"])])
try:
    XDFImport(path)
except Exception as e:
    print("->", type(e).__name__, e)
try:
    XDFImport(path, select_matches=[2])
except Exception as e:
    print("select_matches=[2] ->", type(e).__name__, e)

print("\n=== L. vendored Mesh(mode='back') ===")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hypyp.ext.mpl3d.mesh import Mesh  # noqa: E402

ax = plt.figure().add_subplot()
V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
F = np.array([[0, 1, 2], [0, 1, 3]])
try:
    Mesh(ax, np.eye(4), V, F, mode="back")
except Exception as e:
    print("->", type(e).__name__, e)
