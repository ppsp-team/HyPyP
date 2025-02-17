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
    sensor_pairs = analyses.indices_connectivity_interbrain(
        epochs.epoch_merge)

    # computing metaconn_freq and test it
    metaconn_matrix_2brainsTuple = stats.metaconn_matrix_2brains(sensor_pairs,
                                                                 con_matrixTuple.ch_con, freq)
    metaconn_freq = metaconn_matrix_2brainsTuple.metaconn_freq
    # take a random ch_name:
    random.seed(20)  # Init the random number generator for reproducibility
    # n = random.randrange(0, 63)
    # for our data taske into account EOG ch!!!
    n = random.randrange(0, len(epochs.epo1.info['ch_names']))
    tot = len(epochs.epo1.info['ch_names'])
    p = random.randrange(len(epochs.epo1.info['ch_names']), len(
        epochs.epoch_merge.info['ch_names'])+1)
    # checking for each pair in which ch_name is,
    # whether ch_name linked himself
    # (in neighbouring frequencies also)
    assert metaconn_freq[n+tot, p] == metaconn_freq[n, p]
    assert metaconn_freq[n-tot, p] == metaconn_freq[n, p]
    assert metaconn_freq[n+tot, p+tot] == metaconn_freq[n, p]
    assert metaconn_freq[n-tot, p-tot] == metaconn_freq[n, p]
    assert metaconn_freq[n, p+tot] == metaconn_freq[n, p]
    assert metaconn_freq[n, p-tot] == metaconn_freq[n, p]
    # and not in the other frequencies
    if metaconn_freq[n, p] == 1:
        for i in range(1, len(freq)):
            assert metaconn_freq[n+tot*(i+1), p] != metaconn_freq[n, p]
            assert metaconn_freq[n-tot*(i+1), p] != metaconn_freq[n, p]
            assert metaconn_freq[n+tot *
                                 (i+1), p+tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n-tot *
                                 (i+1), p-tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n, p+tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n, p-tot*(i+1)] != metaconn_freq[n, p]
            # check for each f if connects to the good other ch and not to more
            assert metaconn_freq[n+tot*i, p+tot*i] == ch_con_freq[n, p-tot]


def test_PSD(epochs):
    """
    Test PSD
    """
    fmin = 10
    fmax = 13
    psd_tuple = analyses.pow(epochs.epo1,
                             fmin, fmax,
                             n_fft=256,
                             n_per_seg=None,
                             epochs_average=True)
    psd = psd_tuple.psd
    freq_list = psd_tuple.freq_list
    assert type(psd) == np.ndarray
    assert psd.shape == (
        len(epochs.epo1.info['ch_names']), len(freq_list))
    psd_tuple = analyses.pow(epochs.epo1,
                             fmin, fmax,
                             n_fft=256,
                             n_per_seg=None,
                             epochs_average=False)
    psd = psd_tuple.psd
    assert psd.shape == (len(epochs.epo1), len(
        epochs.epo1.info['ch_names']), len(freq_list))


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

    corr_tuple = analyses.behav_corr(data, behav,
                                     data_name='epochs',
                                     behav_name='time',
                                     p_thresh=p_thresh,
                                     multiple_corr=False,
                                     verbose=False)
    assert pytest.approx(corr_tuple.r) in [-1, 1]

    # test for connectivity values data
    # generate artificial group of 2 subjects repeated
    mne.epochs.equalize_epoch_counts([epochs.epo1, epochs.epo2])
    assert len(epochs.epo1) == len(epochs.epo2)
    con_ind = analyses.pair_connectivity(np.array([epochs.epo1, epochs.epo1]),
                                         sampling_rate=epochs.epo1.info['sfreq'],
                                         frequencies=[8, 10],
                                         mode='ccorr',
                                         epochs_average=True)
    con_subj = analyses.pair_connectivity(np.array([epochs.epo1, epochs.epo2]),
                                          sampling_rate=epochs.epo1.info['sfreq'],
                                          frequencies=[8, 10],
                                          mode='ccorr',
                                          epochs_average=True)
    # remove frequency dimension
    con_ind = np.mean(con_ind, axis=0)
    con_subj = np.mean(con_subj, axis=0)
    assert con_ind.shape == (62, 62)
    assert con_subj.shape == (62, 62)
    data = np.stack((con_ind, con_subj, con_subj, con_subj, con_subj, con_subj, con_subj))
    behav = np.array([0, 1, 1, 1, 1, 1, 1])
    # correlate connectivity and behaviour across pairs without multiple comparison correction
    corr_tuple = analyses.behav_corr(data, behav,
                                     data_name='ccorr',
                                     behav_name='imitation score',
                                     p_thresh=p_thresh,
                                     multiple_corr=False,
                                     verbose=True)
    # test that there is a correlation (repeated measures)
    significant_r = []
    for i in range(0, corr_tuple.r.shape[0]):
        for j in range(0, corr_tuple.r.shape[1]):
            if corr_tuple.pvalue[i, j] <= p_thresh:
                significant_r.append(corr_tuple.r[i, j])
    assert len(significant_r) != 0

    # correlate connectivity and behaviour across pairs with multiple comparison correction
    corr_tuple = analyses.behav_corr(data, behav,
                                     data_name='ccorr',
                                     behav_name='imitation score',
                                     p_thresh=p_thresh,
                                     multiple_corr=True,
                                     verbose=True)
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
        con = analyses.pair_connectivity(np.array([random_r1, random_r2]),
                                         sampling_rate=epochs.epo1.info['sfreq'],
                                         frequencies=[8, 10],
                                         mode='ccorr',
                                         epochs_average=True)
        data.append(np.mean(con, axis=0))
    data = np.mean(np.array([data]), axis=0)
    # correlate connectivity and behaviour across pairs
    dyads = data.shape[0]
    behav = np.arange(0, dyads)
    corr_tuple = analyses.behav_corr(data, behav,
                                     data_name='ccorr',
                                     behav_name='imitation score',
                                     p_thresh=p_thresh,
                                     multiple_corr=True,
                                     verbose=False)
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
    length = len(epochs.epo1.info['ch_names'])
    L = []
    for i in range(1, length):
        L.append(length-i)
    tot = sum(L)
    assert len(electrodes) == tot
    electrodes_hyper = analyses.indices_connectivity_interbrain(
        epochs.epoch_merge)
    assert len(electrodes_hyper) == length*length
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
    psd_tuple = analyses.pow(random_r1,
                             fmin, fmax,
                             n_fft=256,
                             n_per_seg=None,
                             epochs_average=False)
    psd = psd_tuple.psd

    statsCondTuple = stats.statsCond(psd, random_r1, 3000, 0.05)
    assert statsCondTuple.T_obs.shape[0] == len(epochs.epo1.info['ch_names'])

    for i in range(0, len(statsCondTuple.p_values)):
        assert statsCondTuple.p_values[i] <= statsCondTuple.adj_p[1][i]
    assert statsCondTuple.T_obs_plot.shape[0] == len(
        epochs.epo1.info['ch_names'])

    psd_tuple2 = analyses.pow(random_r2,
                              fmin, fmax,
                              n_fft=256,
                              n_per_seg=None,
                              epochs_average=False)
    psd2 = psd_tuple2.psd
    freq_list = psd_tuple2.freq_list

    data = [psd, psd2]
    con_matrixTuple = stats.con_matrix(random_r1, freq_list, draw=False)
    statscondClusterTuple = stats.statscondCluster(data,
                                                   freq_list,
                                                   scipy.sparse.bsr_matrix(
                                                       con_matrixTuple.ch_con_freq),
                                                   tail=0,
                                                   n_permutations=3000,
                                                   alpha=0.05)
    assert statscondClusterTuple.F_obs.shape[0] == len(
        epochs.epo1.info['ch_names'])
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
    assert len(ep_hyper.info['ch_names']) == 2 * \
        len(epochs.epo1.info['ch_names'])
    # check EOG channels number

    # check data for S2 or 1 correspond in the ep_hyper, on channel n and
    # epoch n, randomnly assigned
    random.seed(10)
    nch = random.randrange(0, len(epochs.epo1.info['ch_names']))
    ne = random.randrange(0, len(epochs.epo1))
    ch_name = epochs.epo1.info['ch_names'][nch]
    liste = ep_hyper.info['ch_names']
    ch_index1 = liste.index(ch_name + '_S1')
    ch_index2 = liste.index(ch_name + '_S2')
    ep_hyper_data = ep_hyper.get_data(copy=True)
    epo1_data = epochs.epo1.get_data(copy=True)
    epo2_data = epochs.epo2.get_data(copy=True)
    for i in range(0, len(ep_hyper_data[ne][ch_index1])):
        assert ep_hyper_data[ne][ch_index1][i] == epo1_data[ne][nch][i]
        assert ep_hyper_data[ne][ch_index2][i] == epo2_data[ne][nch][i]


def test_compute_nmPLV(epochs):
    result = analyses.compute_nmPLV(data=epochs, sampling_rate=500, freq_range1=[4, 8], freq_range2=[10, 14])
    hPLV = result[:, :31, 31:].mean()
    PLV1 = result[:, :31, :31].mean()
    PLV2 = result[:, 31:, 31:].mean()
    assert hPLV < PLV1
    assert hPLV < PLV2
    assert (PLV1 - PLV2) < 1e-2
