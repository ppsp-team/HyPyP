#!/usr/bin/env python
# coding=utf-8

import pytest
import random
import numpy as np
import scipy
import mne
from hypyp import stats
from hypyp import utils
from hypyp import analyses


def test_metaconn_matrix_2brains(epochs):
    """
    Test metaconn_matrix_2brains
    """
    # taking random freq-of-interest to test metaconn_freq
    freq = [11, 12, 13]
    # computing ch_con and sensors pairs for metaconn calculation
    con_matrixTuple = stats.con_matrix(epochs.epo1, freq, draw=False)
    ch_con_freq = con_matrixTuple.ch_con_freq
    sensor_pairs = analyses.indices_connectivity_interbrain(epochs.epoch_merge)

    # computing metaconn_freq and test it
    metaconn_matrix_2brainsTuple = stats.metaconn_matrix_2brains(
        sensor_pairs, con_matrixTuple.ch_con, freq
    )
    metaconn_freq = metaconn_matrix_2brainsTuple.metaconn_freq
    # take a random ch_name:
    random.seed(20)  # Init the random number generator for reproducibility
    # n = random.randrange(0, 63)
    # for our data taske into account EOG ch!!!
    n = random.randrange(0, len(epochs.epo1.info["ch_names"]))
    tot = len(epochs.epo1.info["ch_names"])
    p = random.randrange(
        len(epochs.epo1.info["ch_names"]), len(epochs.epoch_merge.info["ch_names"]) + 1
    )
    # checking for each pair in which ch_name is,
    # whether ch_name linked himself
    # (in neighbouring frequencies also)
    assert metaconn_freq[n + tot, p] == metaconn_freq[n, p]
    assert metaconn_freq[n - tot, p] == metaconn_freq[n, p]
    assert metaconn_freq[n + tot, p + tot] == metaconn_freq[n, p]
    assert metaconn_freq[n - tot, p - tot] == metaconn_freq[n, p]
    assert metaconn_freq[n, p + tot] == metaconn_freq[n, p]
    assert metaconn_freq[n, p - tot] == metaconn_freq[n, p]
    # and not in the other frequencies
    if metaconn_freq[n, p] == 1:
        for i in range(1, len(freq)):
            assert metaconn_freq[n + tot * (i + 1), p] != metaconn_freq[n, p]
            assert metaconn_freq[n - tot * (i + 1), p] != metaconn_freq[n, p]
            assert (
                metaconn_freq[n + tot * (i + 1), p + tot * (i + 1)]
                != metaconn_freq[n, p]
            )
            assert (
                metaconn_freq[n - tot * (i + 1), p - tot * (i + 1)]
                != metaconn_freq[n, p]
            )
            assert metaconn_freq[n, p + tot * (i + 1)] != metaconn_freq[n, p]
            assert metaconn_freq[n, p - tot * (i + 1)] != metaconn_freq[n, p]
            # check for each f if connects to the good other ch and not to more
            assert metaconn_freq[n + tot * i, p + tot * i] == ch_con_freq[n, p - tot]


def test_PSD(epochs):
    """
    Test PSD
    """
    fmin = 10
    fmax = 13
    psd_tuple = analyses.pow(
        epochs.epo1, fmin, fmax, n_fft=256, n_per_seg=None, epochs_average=True
    )
    psd = psd_tuple.psd
    freq_list = psd_tuple.freq_list
    assert type(psd) == np.ndarray
    assert psd.shape == (len(epochs.epo1.info["ch_names"]), len(freq_list))
    psd_tuple = analyses.pow(
        epochs.epo1, fmin, fmax, n_fft=256, n_per_seg=None, epochs_average=False
    )
    psd = psd_tuple.psd
    assert psd.shape == (
        len(epochs.epo1),
        len(epochs.epo1.info["ch_names"]),
        len(freq_list),
    )


def test_behav_corr(epochs):
    """
    Test data-behav correlation
    """
    # test for vector data
    # data = epochs.epo1
    data = np.arange(0, 10)
    step = len(data)
    behav = np.arange(0, step)
    assert len(data) == len(behav)

    p_thresh = 0.05

    corr_tuple = analyses.behav_corr(
        data,
        behav,
        data_name="epochs",
        behav_name="time",
        p_thresh=p_thresh,
        multiple_corr=False,
        verbose=False,
    )
    assert pytest.approx(corr_tuple.r) in [-1, 1]

    # test for connectivity values data
    # generate artificial group of 2 subjects repeated
    mne.epochs.equalize_epoch_counts([epochs.epo1, epochs.epo2])
    assert len(epochs.epo1) == len(epochs.epo2)
    con_ind = analyses.pair_connectivity(
        np.array([epochs.epo1, epochs.epo1]),
        sampling_rate=epochs.epo1.info["sfreq"],
        frequencies=[8, 10],
        mode="ccorr",
        epochs_average=True,
    )
    con_subj = analyses.pair_connectivity(
        np.array([epochs.epo1, epochs.epo2]),
        sampling_rate=epochs.epo1.info["sfreq"],
        frequencies=[8, 10],
        mode="ccorr",
        epochs_average=True,
    )
    # remove frequency dimension
    con_ind = np.mean(con_ind, axis=0)
    con_subj = np.mean(con_subj, axis=0)
    assert con_ind.shape == (62, 62)
    assert con_subj.shape == (62, 62)
    data = np.stack(
        (con_ind, con_subj, con_subj, con_subj, con_subj, con_subj, con_subj)
    )
    behav = np.array([0, 1, 1, 1, 1, 1, 1])
    # correlate connectivity and behaviour across pairs without multiple comparison correction
    corr_tuple = analyses.behav_corr(
        data,
        behav,
        data_name="ccorr",
        behav_name="imitation score",
        p_thresh=p_thresh,
        multiple_corr=False,
        verbose=True,
    )
    # test that there is a correlation (repeated measures)
    significant_r = []
    for i in range(0, corr_tuple.r.shape[0]):
        for j in range(0, corr_tuple.r.shape[1]):
            if corr_tuple.pvalue[i, j] <= p_thresh:
                significant_r.append(corr_tuple.r[i, j])
    assert len(significant_r) != 0

    # correlate connectivity and behaviour across pairs with multiple comparison correction
    corr_tuple = analyses.behav_corr(
        data,
        behav,
        data_name="ccorr",
        behav_name="imitation score",
        p_thresh=p_thresh,
        multiple_corr=True,
        verbose=True,
    )
    # test that there is a correlation (repeated measures)
    significant_r = []
    for i in range(0, corr_tuple.r.shape[0]):
        for j in range(0, corr_tuple.r.shape[1]):
            if corr_tuple.pvalue[i, j] <= p_thresh:
                significant_r.append(corr_tuple.r[i, j])
    assert len(significant_r) == 0

    # generate random subjects' connectivity data
    data = []
    for k in range(0, 5):
        random_r1 = utils.generate_random_epoch(epochs.epo1, mu=0, sigma=0.01)
        random_r2 = utils.generate_random_epoch(epochs.epo2, mu=4, sigma=0.01)
        con = analyses.pair_connectivity(
            np.array([random_r1, random_r2]),
            sampling_rate=epochs.epo1.info["sfreq"],
            frequencies=[8, 10],
            mode="ccorr",
            epochs_average=True,
        )
        data.append(np.mean(con, axis=0))
    data = np.mean(np.array([data]), axis=0)
    # correlate connectivity and behaviour across pairs
    dyads = data.shape[0]
    behav = np.arange(0, dyads)
    corr_tuple = analyses.behav_corr(
        data,
        behav,
        data_name="ccorr",
        behav_name="imitation score",
        p_thresh=p_thresh,
        multiple_corr=True,
        verbose=False,
    )
    # test that there is no correlation (random measures)
    for i in range(0, corr_tuple.r.shape[0]):
        for j in range(0, corr_tuple.r.shape[1]):
            assert corr_tuple.r[i, j] <= 2
            # not 0 because can have a significant correlation
            # for one connection by chance
            # but suppose very weak


def test_indexes_connectivity(epochs):
    """
    Test index intra- and inter-brains
    """
    electrodes = analyses.indices_connectivity_intrabrain(epochs.epo1)
    length = len(epochs.epo1.info["ch_names"])
    L = []
    for i in range(1, length):
        L.append(length - i)
    tot = sum(L)
    assert len(electrodes) == tot
    electrodes_hyper = analyses.indices_connectivity_interbrain(epochs.epoch_merge)
    assert len(electrodes_hyper) == length * length
    # format that do not work for mne.spectral_connectivity


def test_stats(epochs):
    """
    Test stats
    """
    # with PSD from Epochs with random values
    random_r1 = utils.generate_random_epoch(epochs.epo1, mu=0, sigma=0.01)
    random_r2 = utils.generate_random_epoch(epochs.epo2, mu=4, sigma=0.01)

    fmin = 10
    fmax = 13
    psd_tuple = analyses.pow(
        random_r1, fmin, fmax, n_fft=256, n_per_seg=None, epochs_average=False
    )
    psd = psd_tuple.psd

    statsCondTuple = stats.statsCond(psd, random_r1, 3000, 0.05)
    assert statsCondTuple.T_obs.shape[0] == len(epochs.epo1.info["ch_names"])

    for i in range(0, len(statsCondTuple.p_values)):
        assert statsCondTuple.p_values[i] <= statsCondTuple.adj_p[1][i]
    assert statsCondTuple.T_obs_plot.shape[0] == len(epochs.epo1.info["ch_names"])

    psd_tuple2 = analyses.pow(
        random_r2, fmin, fmax, n_fft=256, n_per_seg=None, epochs_average=False
    )
    psd2 = psd_tuple2.psd
    freq_list = psd_tuple2.freq_list

    data = [psd, psd2]
    con_matrixTuple = stats.con_matrix(random_r1, freq_list, draw=False)
    statscondClusterTuple = stats.statscondCluster(
        data,
        freq_list,
        scipy.sparse.bsr_matrix(con_matrixTuple.ch_con_freq),
        tail=0,
        n_permutations=3000,
        alpha=0.05,
    )
    assert statscondClusterTuple.F_obs.shape[0] == len(epochs.epo1.info["ch_names"])
    # for i in range(0, len(statscondClusterTuple.clusters)):
    #    assert len(np.where(statscondClusterTuple.clusters[i])=='True') < len(
    #        epochs.epo1.info['ch_names'])
    assert np.mean(statscondClusterTuple.cluster_p_values) != float(0)
    assert statscondClusterTuple.F_obs_plot.shape == statscondClusterTuple.F_obs.shape


def test_utils(epochs):
    """
    Test merge and split
    """
    ep_hyper = utils.merge(epochs.epo1, epochs.epo2)
    assert type(ep_hyper) == mne.epochs.EpochsArray
    # check channels number
    assert len(ep_hyper.info["ch_names"]) == 2 * len(epochs.epo1.info["ch_names"])
    # check EOG channels number

    # check data for S2 or 1 correspond in the ep_hyper, on channel n and
    # epoch n, randomnly assigned
    random.seed(10)
    nch = random.randrange(0, len(epochs.epo1.info["ch_names"]))
    ne = random.randrange(0, len(epochs.epo1))
    ch_name = epochs.epo1.info["ch_names"][nch]
    liste = ep_hyper.info["ch_names"]
    ch_index1 = liste.index(ch_name + "_S1")
    ch_index2 = liste.index(ch_name + "_S2")
    ep_hyper_data = ep_hyper.get_data(copy=True)
    epo1_data = epochs.epo1.get_data(copy=True)
    epo2_data = epochs.epo2.get_data(copy=True)
    for i in range(0, len(ep_hyper_data[ne][ch_index1])):
        assert ep_hyper_data[ne][ch_index1][i] == epo1_data[ne][nch][i]
        assert ep_hyper_data[ne][ch_index2][i] == epo2_data[ne][nch][i]


def test_compute_nmPLV(epochs):
    result = analyses.compute_nmPLV(
        data=epochs, sampling_rate=500, freq_range1=[4, 8], freq_range2=[10, 14]
    )
    hPLV = result[:, :31, 31:].mean()
    PLV1 = result[:, :31, :31].mean()
    PLV2 = result[:, 31:, 31:].mean()
    assert hPLV < PLV1
    assert hPLV < PLV2
    assert (PLV1 - PLV2) < 1e-2


def test_pair_connectivity_accorr(epochs):
    """
    Test adjusted circular correlation (accorr) connectivity metric.
    """
    mne.epochs.equalize_epoch_counts([epochs.epo1, epochs.epo2])

    # Test with epochs_average=True
    con = analyses.pair_connectivity(
        np.array([epochs.epo1, epochs.epo2]),
        sampling_rate=epochs.epo1.info["sfreq"],
        frequencies=[8, 12],
        mode="accorr",
        epochs_average=True,
    )

    n_channels = len(epochs.epo1.info["ch_names"])
    # Shape should be (n_freq, 2*n_channels, 2*n_channels)
    assert con.shape[1] == 2 * n_channels
    assert con.shape[2] == 2 * n_channels

    # Values should be bounded (typically between -1 and 1 for correlation)
    assert np.all(np.isfinite(con))

    # Test with epochs_average=False
    con_no_avg = analyses.pair_connectivity(
        np.array([epochs.epo1, epochs.epo2]),
        sampling_rate=epochs.epo1.info["sfreq"],
        frequencies=[8, 12],
        mode="accorr",
        epochs_average=False,
    )

    # Shape should be (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    assert con_no_avg.shape[1] == len(epochs.epo1)
    assert con_no_avg.shape[2] == 2 * n_channels
    assert con_no_avg.shape[3] == 2 * n_channels

    # Diagonal should have high values (self-correlation)
    diag_values = np.array([con[0, i, i] for i in range(con.shape[1])])
    assert np.mean(diag_values) > 0.5


def _synthetic_pair(n_epochs, n_channels, n_times=512, seed=0):
    """Random data shaped (2, n_epochs, n_channels, n_times), no download needed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, n_epochs, n_channels, n_times))


def _plv_from_definition(data, sampling_rate, frequencies):
    """PLV written from its definition, without compute_sync or hypyp.sync.

    Returns an array shaped (n_freq, n_epochs, 2*n_channels, 2*n_channels),
    with the channels of participant 1 first.
    """
    # (2, n_epochs, n_channels, n_freq, n_times), tapers averaged
    signal = np.mean(analyses.compute_single_freq(data, sampling_rate, frequencies), 3)
    phase = signal / np.abs(signal)
    both = np.concatenate([phase[0], phase[1]], axis=1)
    n_times = both.shape[-1]
    plv = np.abs(np.einsum("ecft,edft->fecd", both, both.conj())) / n_times
    return plv


@pytest.mark.parametrize(
    "n_epochs, n_channels, frequencies, n_freq",
    [
        (3, 4, [8, 12], 4),  # ordinary case, already worked
        (1, 4, [8, 12], 4),  # single epoch
        (3, 1, [8, 12], 4),  # single channel per participant
        (1, 1, [8, 12], 4),
        (3, 4, [10, 11], 1),  # single frequency
    ],
)
def test_pair_connectivity_frequency_list_singleton_dims(
    n_epochs, n_channels, frequencies, n_freq
):
    """A frequency list must not crash when a dimension has length one, and
    must give the PLV of the definition with the documented layout."""
    data = _synthetic_pair(n_epochs, n_channels)
    expected = _plv_from_definition(data, 256, frequencies)
    assert expected.shape == (n_freq, n_epochs, 2 * n_channels, 2 * n_channels)

    con = analyses.pair_connectivity(
        data,
        sampling_rate=256,
        frequencies=frequencies,
        mode="plv",
        epochs_average=False,
    )
    assert con.shape == expected.shape
    np.testing.assert_allclose(con, expected, rtol=0, atol=1e-12)

    con = analyses.pair_connectivity(
        data, sampling_rate=256, frequencies=frequencies, mode="plv"
    )
    assert con.shape == (n_freq, 2 * n_channels, 2 * n_channels)
    np.testing.assert_allclose(con, expected.mean(axis=1), rtol=0, atol=1e-12)


def test_compute_sync_reports_the_real_error():
    """Only an unknown metric is reported as an unsupported metric."""
    complex_signal = analyses.compute_freq_bands(
        _synthetic_pair(3, 4), 256, {"alpha": [8, 12]}
    )

    with pytest.raises(ValueError, match='Metric type "nope" not supported.'):
        analyses.compute_sync(complex_signal, "nope")

    with pytest.raises(ValueError) as excinfo:
        analyses.compute_sync(complex_signal, "plv", optimization="bogus")
    assert "not supported" not in str(excinfo.value)
    assert "bogus" in str(excinfo.value)


def _cluster_data(seed=0):
    """Two groups of 12 observations over 6 features; group 2 is larger on
    the first three features."""
    rng = np.random.default_rng(seed)
    low = rng.standard_normal((12, 6))
    high = rng.standard_normal((12, 6))
    high[:, :3] += 5.0
    adjacency = scipy.sparse.csr_matrix(np.eye(6) + np.eye(6, k=1) + np.eye(6, k=-1))
    return low, high, adjacency


def test_statscluster_unknown_test_name():
    low, high, adjacency = _cluster_data()
    with pytest.raises(ValueError, match="bogus"):
        stats.statscluster([low, high], "bogus", None, adjacency, 0, 50)


def test_metaconn_matrix_plot_argument():
    import matplotlib.pyplot as plt

    ch_con = scipy.sparse.csr_matrix(np.eye(4) + np.eye(4, k=1) + np.eye(4, k=-1))
    electrodes = [(0, 1), (1, 2), (2, 3)]

    plt.close("all")
    quiet = stats.metaconn_matrix(electrodes, ch_con, [10, 11], plot=False)
    assert plt.get_fignums() == []

    # the default still draws, as before
    drawn = stats.metaconn_matrix(electrodes, ch_con, [10, 11])
    assert len(plt.get_fignums()) == 1
    plt.close("all")

    np.testing.assert_array_equal(quiet.metaconn, drawn.metaconn)
    np.testing.assert_array_equal(quiet.metaconn_freq, drawn.metaconn_freq)
    assert quiet.metaconn_freq.shape == (6, 6)
