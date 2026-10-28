"""fNIRS Recording/Dyad/Study checks on the bundled DCARE sample files."""

import logging
import warnings

import numpy as np

from hypyp import datasets
from hypyp.fnirs.recording import Recording
from hypyp.fnirs.dyad import Dyad
from hypyp.fnirs.study import Study
from hypyp.fnirs.channel_roi import ChannelROI
from hypyp.utils import Task
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet

logging.disable()
warnings.filterwarnings("ignore")

f1 = datasets.fnirs_dcare(subject=1)
f2 = datasets.fnirs_dcare(subject=2)

print("=== 1. dt of PairSignals built by Dyad._append_pairs ===")
s1 = Recording(subject_label="s1").load_file(f1)
s2 = Recording(subject_label="s2").load_file(f2)
sfreq = s1.preprocessed.info["sfreq"]
dyad = Dyad(s1, s2)
pair = dyad.get_pairs(s1, s2, ch_match="S1_D1 760")[0]
print("sfreq", sfreq, " 1/sfreq", 1 / sfreq, " pair.dt", pair.dt, " rel err", pair.dt * sfreq - 1, " n", pair.n)
t20 = [Task("t", onset_time=0, duration=20)]
s1b = Recording(subject_label="s1", tasks=t20).load_file(f1)
s2b = Recording(subject_label="s2", tasks=t20).load_file(f2)
pair20 = Dyad(s1b, s2b).get_pairs(s1b, s2b, ch_match="S1_D1 760")[0]
print("20 s task: n", pair20.n, " pair.dt*sfreq-1 =", pair20.dt * sfreq - 1)
w = ComplexMorletWavelet(disable_caching=True)
r = w.wtc(pair20)
print("WTC.sfreq reported", r.sfreq, " true", sfreq)

print("\n=== 2. default preprocessor: what channels get paired ===")
print("preprocessed ch types:", sorted(set(s1.preprocessed.get_channel_types())))
print("first 4 ch names:", s1.preprocessed.ch_names[:4])
pairs = dyad.get_pairs(s1, s2)
labels = {(p.label_ch1.split()[-1], p.label_ch2.split()[-1]) for p in pairs}
print("wavelength combinations paired with ch_match=None:", sorted(labels), " n pairs", len(pairs))

print("\n=== 3. channel order with ChannelROI: does pick() follow the ROI order? ===")
croi = ChannelROI.from_lionirs_file(datasets.fetch("fnirs/lionirs/channel_grouping_7ROI.mat"))
r1 = Recording(subject_label="r1", channel_roi=croi).load_file(f1)
r2 = Recording(subject_label="r2", channel_roi=croi).load_file(f2)
print("ordered_ch_names[:3]", r1.ordered_ch_names[:3], " raw ch_names[:3]", r1.preprocessed.ch_names[:3])
pr = Dyad(r1, r2).get_pairs(r1, r2)
p0 = pr[0]
truth = r1.preprocessed.get_data(picks=[p0.label_ch1])[0]
print("pairs[0] label_ch1", p0.label_ch1, " y1 matches that channel's data:", np.allclose(p0.y1, truth))
truth_first_raw = r1.preprocessed.get_data(picks=[r1.preprocessed.ch_names[0]])[0]
print("y1 matches raw first channel instead:", np.allclose(p0.y1, truth_first_raw))

print("\n=== 4. intra WTC duplication when a Recording is in two dyads ===")
sA = Recording(subject_label="A", tasks=t20).load_file(f1)
sB = Recording(subject_label="B", tasks=t20).load_file(f2)
sC = Recording(subject_label="C", tasks=t20).load_file(f2)
study = Study([Dyad(sA, sB), Dyad(sA, sC)])
study.compute_wtcs(ch_match="S1_D1 760", with_intra=True, show_time_estimation=False)
df = study.df
intra_A = df[(df.is_intra) & (df.subject1 == "A")]
print("rows in study.df:", len(df), " intra rows for A:", len(intra_A), " (expected 1)")

print("\n=== 5. keep_wtcs=False leaves s2.intra_wtcs ===")
d = Dyad(sA, sB)
d.compute_wtcs(ch_match="S1_D1 760", with_intra=True, keep_wtcs=False)
print("s1.intra_wtcs", len(sA.intra_wtcs), " s2.intra_wtcs", len(sB.intra_wtcs))

print("\n=== 6. pseudo-dyads duplicate real dyads when participants recur ===")
sh = study.get_dyads_shuffle()
print([f"{x.s1.subject_label}-{x.s2.subject_label}" for x in sh])

print("\n=== 7. t_ranges drops trailing partial bin ===")
x = np.arange(1000) * 0.1
rng = np.random.default_rng(0)
rr = w.wtc(PairSignals(x, rng.standard_normal(1000), rng.standard_normal(1000)), bin_seconds=30)
print("duration 100 s, bin 30 s ->", rr.t_ranges, " rows:", len(rr.as_frame_rows))

print("\n=== 8. ch_match substring semantics ===")
m = dyad.get_pairs(s1, s2, ch_match="S1_D1")
print("ch_match='S1_D1' picks s1 channels:", sorted({p.label_ch1 for p in m}))

print("\n=== 9. std=0 channel ===")
rz = w.wtc(PairSignals(x, np.zeros(1000), rng.standard_normal(1000)))
print("constant signal -> coherence_metric", rz.coherence_metric)
