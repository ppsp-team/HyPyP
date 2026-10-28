"""Compute a fixed set of HyPyP outputs on the sample data and save them.

Run from a checkout (the current directory's hypyp is imported) and compare
the resulting .npz files with compare.py. Every stochastic call is preceded
by np.random.seed so that versions drawing from the global generator agree;
calls that moved to a numpy Generator are expected to differ and are listed
as such in the hand-over. Only the call signatures of 0.6.2 are used, so the
script runs unchanged on master and on every fix branch.

Usage: python run_in_cwd.py outputs.py <out.npz>
"""

import os
import sys
import traceback
import warnings

import mne
import numpy as np
import scipy.sparse

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

OUT = {}


def record(key, value):
    if scipy.sparse.issparse(value):
        value = value.toarray()
    OUT[key] = np.asarray(value)


def attempt(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - we record the failure as a value
        OUT[f"{name}.ERROR"] = np.array(f"{type(exc).__name__}: {exc}"[:200])
        traceback.print_exc(limit=1)


from hypyp import analyses, prep, stats, utils  # noqa: E402
from hypyp.mvarica import MVAR  # noqa: E402
from hypyp.signal import SyntheticSignal  # noqa: E402

root = os.getcwd()
epo1 = mne.read_epochs(f"{root}/tests/data/preproc_S1.fif", preload=True, verbose=False)
epo2 = mne.read_epochs(f"{root}/tests/data/preproc_S2.fif", preload=True, verbose=False)
mne.epochs.equalize_epoch_counts([epo1, epo2])
SFREQ = epo1.info["sfreq"]
data = np.array([epo1.get_data(copy=True)[:4], epo2.get_data(copy=True)[:4]])
n_ch = data.shape[2]


def sync():
    complex_signal = analyses.compute_freq_bands(data, SFREQ, {"alpha": [8, 12]})
    for mode in ["plv", "ccorr", "accorr", "coh", "imcoh", "pli", "wpli", "envelope_corr", "pow_corr"]:
        record(f"sync.{mode}", analyses.compute_sync(complex_signal, mode))
    record("sync.plv.per_epoch", analyses.compute_sync(complex_signal, "plv", epochs_average=False))


def pair():
    record("pair_connectivity.plv.list", analyses.pair_connectivity(data[:, :, :4], SFREQ, [8, 12], "plv"))


def power():
    psd = analyses.pow(
        epo1, fmin=8, fmax=12, n_fft=256, n_per_seg=256, epochs_average=True
    )
    record("pow.psd", psd.psd)
    record("pow.freq_list", psd.freq_list)


def adjacency():
    one = stats.con_matrix(epo1, [10], draw=False)
    record("con_matrix.1f.ch_con", one.ch_con)
    record("con_matrix.1f.ch_con_freq", one.ch_con_freq)
    three = stats.con_matrix(epo1, [10, 11, 12], draw=False)
    record("con_matrix.3f.ch_con_freq", three.ch_con_freq)
    pairs_intra = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    intra = stats.metaconn_matrix(pairs_intra, one.ch_con, [10], plot=False)
    record("metaconn.1f.metaconn", intra.metaconn)
    record("metaconn.1f.metaconn_freq", intra.metaconn_freq)
    intra2 = stats.metaconn_matrix(pairs_intra, one.ch_con, [10, 11], plot=False)
    record("metaconn.2f.metaconn_freq", intra2.metaconn_freq)
    pairs_inter = [(i, n_ch + j) for i in range(n_ch) for j in range(n_ch)]
    inter = stats.metaconn_matrix_2brains(pairs_inter, one.ch_con, [10], plot=False)
    record("metaconn_2brains.1f.metaconn", inter.metaconn)
    record("metaconn_2brains.1f.metaconn_freq", inter.metaconn_freq)


def cluster_groups(seed):
    rng = np.random.default_rng(seed)
    groups = [rng.standard_normal((8, n_ch, 1)) for _ in range(2)]
    groups[1][:, :3, 0] += 2.5
    return groups


def cluster_tests():
    adjacency_1f = scipy.sparse.bsr_matrix(stats.con_matrix(epo1, [10], draw=False).ch_con_freq)
    groups = cluster_groups(0)
    np.random.seed(0)
    result = stats.statscondCluster(groups, [10], adjacency_1f, 1, 200, 0.05)
    record("statscondCluster.F_obs", result.F_obs)
    record("statscondCluster.n_clusters", len(result.clusters))
    record("statscondCluster.cluster_p_values", result.cluster_p_values)
    record("statscondCluster.F_obs_plot", result.F_obs_plot)

    chain = scipy.sparse.csr_matrix(np.eye(12) + np.eye(12, k=1) + np.eye(12, k=-1))
    rng = np.random.default_rng(1)
    two = [rng.standard_normal((10, 12)) for _ in range(2)]
    two[1][:, 3:6] += 2.0
    three = two + [rng.standard_normal((10, 12))]
    three[2][:, 8:10] += 2.0
    # pairs sharing a large offset across features, where the paired null matters
    offset = rng.standard_normal((20, 1)) * 3.0
    paired = [offset + rng.standard_normal((20, 12)) * 0.5 for _ in range(2)]
    paired[1][:, 3:6] += 0.5
    for test, groups in [("ind ttest", two), ("rel ttest", paired), ("f oneway", three)]:
        np.random.seed(0)
        tail = 1 if test == "f oneway" else 0
        result = stats.statscluster(groups, test, None, chain, tail, 200, 0.05)
        key = test.replace(" ", "_")
        record(f"statscluster.{key}.Stat_obs", result.Stat_obs)
        record(f"statscluster.{key}.cluster_p_values", result.cluster_p_values)
        record(f"statscluster.{key}.Stat_obs_plot", result.Stat_obs_plot)


def stats_cond():
    contrast = np.random.default_rng(2).standard_normal((8, n_ch, 3))
    contrast[:, 0, :] += 1.5
    np.random.seed(0)
    result = stats.statsCond(contrast, epo1, 200, 0.05)
    record("statsCond.T_obs", result.T_obs)
    record("statsCond.p_values", result.p_values)
    record("statsCond.adj_p1", result.adj_p[1])
    record("statsCond.T_obs_plot", result.T_obs_plot)


def nmplv():
    record("compute_nmPLV", analyses.compute_nmPLV(data[:, :2, :4], SFREQ, [4, 8], [10, 14]))


def simulators():
    template = epo1.copy().pick(epo1.ch_names[:4])[:2]
    np.random.seed(0)
    record("generate_virtual_epoch", utils.generate_virtual_epoch(template, np.zeros((4, 4))).get_data(copy=True))
    np.random.seed(0)
    record("generate_random_epoch", utils.generate_random_epoch(template, 0, 1.0).get_data(copy=True))
    np.random.seed(0)
    record("SyntheticSignal.add_noise", SyntheticSignal().add_noise().y)


def autoreject():
    for strategy in ("union", "intersection"):
        cleaned, dic = prep.AR_local([epo1, epo2], strategy=strategy, threshold=50.0)
        for k, epochs in enumerate(cleaned):
            record(f"AR_local.{strategy}.events_S{k + 1}", epochs.events[:, 0])
        record(f"AR_local.{strategy}.dyad_pct", dic["dyad"])


def xdf():
    from hypyp import datasets
    from hypyp.xdf import XDFImport

    imported = XDFImport(datasets.xdf_dyad_with_markers())
    record("xdf.raw0.data_head", imported.mne_raws[0].get_data()[:, :1000])
    record("xdf.raw1.data_head", imported.mne_raws[1].get_data()[:, :1000])
    record("xdf.markers.timestamps", imported.markers.timestamps)
    record("xdf.raw0.n_annotations", len(imported.mne_raws[0].annotations))
    record("xdf.raw1.n_annotations", len(imported.mne_raws[1].annotations))


def mvar():
    B1 = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.4], [0.0, 0.0, 0.5]])
    rng = np.random.default_rng(3)
    x = np.zeros((2, 3, 1500))
    for e in range(2):
        for t in range(2, 1500):
            x[e, :, t] = B1 @ x[e, :, t - 1] - 0.2 * x[e, :, t - 2] + rng.standard_normal(3)
    model = MVAR(model_order=2).fit(x)
    record("mvar.coeff", model.coeff)
    record("mvar.stable", model.stability())


for name, fn in [
    ("sync", sync),
    ("pair", pair),
    ("pow", power),
    ("adjacency", adjacency),
    ("cluster", cluster_tests),
    ("statsCond", stats_cond),
    ("nmplv", nmplv),
    ("simulators", simulators),
    ("autoreject", autoreject),
    ("xdf", xdf),
    ("mvar", mvar),
]:
    attempt(name, fn)

np.savez(sys.argv[-1], **OUT)
print(f"saved {len(OUT)} arrays to {sys.argv[-1]}")
