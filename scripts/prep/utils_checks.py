"""utils.py checks: create_epochs, merge, split, epochs_from_tasks, generate_virtual_epoch."""

import warnings

import mne
import numpy as np
from scipy.signal import hilbert

from hypyp import utils

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def raw_of(duration, sfreq=100, n_ch=4, names=None, seed=0, first_samp=0):
    rng = np.random.RandomState(seed)
    names = names or [f"C{i}" for i in range(n_ch)]
    info = mne.create_info(names, sfreq, "eeg")
    return mne.io.RawArray(rng.randn(n_ch, int(sfreq * duration)), info, first_samp=first_samp)


print("===== create_epochs: 10 s at 100 Hz, duration=1.0")
e1, e2 = utils.create_epochs([raw_of(10)], [raw_of(10, seed=1)], duration=1.0)
print(f"  n_epochs={len(e1[0])}  n_times/epoch={len(e1[0].times)}  tmin={e1[0].tmin} tmax={e1[0].tmax}")
d = e1[0].get_data(copy=True)
print(f"  last sample of epoch 0 == first sample of epoch 1: {np.allclose(d[0, :, -1], d[1, :, 0])}")
print(f"  drop_log tail: {e1[0].drop_log[-2:]}")

print("\n===== merge: different sampling rates (100 vs 200 Hz)")
ea, _ = utils.create_epochs([raw_of(10, sfreq=100)], [raw_of(10, sfreq=100)], 1.0)
eb, _ = utils.create_epochs([raw_of(10, sfreq=200)], [raw_of(10, sfreq=200)], 1.0)
try:
    m = utils.merge(ea[0], eb[0])
    md = m.get_data(copy=True)
    print(f"  merged shape={md.shape} sfreq={m.info['sfreq']} n_nan={np.isnan(md).sum()} of {md.size}")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== merge: same sfreq, different epoch length (1.0 s vs 0.5 s)")
ec, _ = utils.create_epochs([raw_of(10)], [raw_of(10)], 0.5)
try:
    ec = ec[0][: len(ea[0])]
    m = utils.merge(ea[0], ec)
    md = m.get_data(copy=True)
    print(f"  merged shape={md.shape} n_nan={np.isnan(md).sum()} of {md.size}")
except Exception as e:  # noqa: BLE001
    print(f"  raised {type(e).__name__}: {e}")

print("\n===== merge: different channel names in S2 (same count)")
ed, _ = utils.create_epochs([raw_of(10, names=["X0", "X1", "X2", "X3"])], [raw_of(10)], 1.0)
m = utils.merge(ea[0], ed[0])
print(f"  merged ch_names: {m.ch_names}")

print("\n===== merge: input mutation when bads present, tmin/events preservation")
ep1 = ea[0].copy()
ep1.set_montage("standard_1020", on_missing="ignore")
ep1.info["ch_names"]
ep1 = mne.EpochsArray(
    ea[0].get_data(copy=True),
    mne.create_info(["Fp1", "Fp2", "C3", "C4"], 100, "eeg"),
    tmin=-0.2,
)
ep1.set_montage("standard_1020")
ep1.info["bads"] = ["Fp1"]
ep2 = ep1.copy()
ep2.info["bads"] = []
before = ep1.get_data(copy=True).copy()
m = utils.merge(ep1, ep2)
print(f"  ep1 bads after merge: {ep1.info['bads']}  ep1 data changed: {not np.array_equal(before, ep1.get_data(copy=True))}")
print(f"  ep1.tmin={ep1.tmin} merged.tmin={m.tmin}; merged events col0: {m.events[:3, 0].tolist()}")
print(f"  merged ch types: {set(m.get_channel_types())}; montage: {m.get_montage()}")

print("\n===== split: channel names with underscores, and EOG detection")
raw = raw_of(2, names=["Fp1_S1", "EOG_left_S1", "Fp1_S2", "EOG_left_S2"])
r1, r2 = utils.split(raw)
print(f"  split ch_names S1: {r1.ch_names}  types: {r1.get_channel_types()}")

print("\n===== epochs_from_tasks: TASK_BEGINNING / TASK_END with first_samp != 0")
raw = raw_of(10, first_samp=300)
raw.set_annotations(mne.Annotations(onset=[2, 5], duration=[0, 0], description=["1", "2"]))
for name, task in [
    ("BEGINNING->event2", utils.Task("t", onset_event_id=utils.TASK_BEGINNING, offset_event_id=2)),
    ("event1->END", utils.Task("t", onset_event_id=1, offset_event_id=utils.TASK_END)),
]:
    try:
        ep = utils.epochs_from_tasks(raw, [task])[0]
        print(f"  {name}: n_epochs={len(ep)} tmax={ep.tmax:.3f} (expected: BEGINNING->event2 tmax=5.0, event1->END tmax=8.0) events={ep.events.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: raised {type(e).__name__}: {e}")
raw0 = raw_of(10, first_samp=0)
raw0.set_annotations(mne.Annotations(onset=[2, 5], duration=[0, 0], description=["1", "2"]))
ep = utils.epochs_from_tasks(raw0, [utils.Task("t", onset_event_id=utils.TASK_BEGINNING, offset_event_id=2)])[0]
print(f"  first_samp=0 control BEGINNING->event2: n_epochs={len(ep)} tmax={ep.tmax:.3f}")

print("\n===== normalizing: axis and formula check")
rng = np.random.RandomState(0)
base = rng.randn(30, 2, 3) + 5
task = rng.randn(20, 2, 3) + 7
z = utils.normalizing(base, task, "Zscore")
ref = (task.mean(0) - base.mean(0)) / base.std(0, ddof=0)
print(f"  Zscore == (mean_task-mean_base)/std_base(ddof=0) over epochs: {np.allclose(z, ref)}; shape {z.shape}")

print("\n===== generate_virtual_epoch")
n_ch, n_epo, n_t, sfreq = 6, 4, 500, 250.0
info = mne.create_info([f"C{i}" for i in range(n_ch)], sfreq, "eeg")
tmpl = mne.EpochsArray(np.zeros((n_epo, n_ch, n_t)), info)

# a) time grid vs sfreq
Nt = n_t * n_epo
tv = np.linspace(0.0, n_t / sfreq * n_epo, Nt)
print(f"  a) grid dt={tv[1]-tv[0]:.8f} vs 1/sfreq={1/sfreq:.8f}; relative freq bias={(1/sfreq)/(tv[1]-tv[0]) - 1:.2e}")

# b) seed handling
np.random.seed(1)
s1 = utils.generate_virtual_epoch(tmpl, np.zeros((n_ch, n_ch))).get_data(copy=True)
np.random.seed(1)
s2 = utils.generate_virtual_epoch(tmpl, np.zeros((n_ch, n_ch))).get_data(copy=True)
s3 = utils.generate_virtual_epoch(tmpl, np.zeros((n_ch, n_ch))).get_data(copy=True)
print(f"  b) same np.random.seed -> identical: {np.array_equal(s1, s2)}; unseeded call differs: {not np.array_equal(s1, s3)}")


def plv(x, y):
    px, py = np.angle(hilbert(x)), np.angle(hilbert(y))
    return np.abs(np.mean(np.exp(1j * (px - py))))


# c) initial condition: first half of channels start at identical phase 0
np.random.seed(0)
sim = utils.generate_virtual_epoch(tmpl, np.zeros((n_ch, n_ch)), noise_amplitude_level=0.0).get_data(copy=True)
print(f"  c) W=0: PLV(ch0,ch1) first epoch (both in first half) = {plv(sim[0,0], sim[0,1]):.3f}; "
      f"PLV(ch3,ch4) first epoch (second half) = {plv(sim[0,3], sim[0,4]):.3f}; "
      f"PLV(ch0,ch1) last epoch = {plv(sim[-1,0], sim[-1,1]):.3f}")
print(f"     first-sample phases: {np.round(np.arcsin(np.clip(sim[0,:,0], -1, 1)), 3).tolist()}")

# d) coupling direction / effect: W[4,0] = strong coupling from 0 to 4 only
W = np.zeros((n_ch, n_ch))
W[4, 0] = 3.0
np.random.seed(0)
sim = utils.generate_virtual_epoch(tmpl, W, frequency_std=0.5, noise_amplitude_level=0.0).get_data(copy=True)
print(f"  d) W[4,0]=3: PLV(ch0,ch4) last epoch = {plv(sim[-1,0], sim[-1,4]):.3f}; uncoupled PLV(ch0,ch5) = {plv(sim[-1,0], sim[-1,5]):.3f}")

# e) peak frequency
psd = np.abs(np.fft.rfft(sim[-1, 0]))
freqs = np.fft.rfftfreq(n_t, 1 / sfreq)
print(f"  e) peak freq ch0 = {freqs[np.argmax(psd)]:.2f} Hz (frequency_mean=10)")
