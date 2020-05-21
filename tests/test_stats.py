#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
import scipy
import mne
from hypyp import prep
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


def test_ICA(epochs):
    """
    Test ICA fit, ICA choice comp and ICA apply
    """
    ep = [epochs.epo1, epochs.epo2]
    icas = prep.ICA_fit(ep, n_components=15, method='fastica', random_state=97)
    # check that the number of componenents is similar between the two subjects
    for i in range(0, len(icas)-1):
        mne.preprocessing.ICA.get_components(
            icas[i]).shape == mne.preprocessing.ICA.get_components(icas[i+1]).shape
    # cleaned_epochs_ICA = prep.ICA_choice_comp(icas, ep) # pb interactive window
    # check signal better after than before
    # check bad channels are not deleted
    # assert epochs.epo1.info['ch_names'] == cleaned_epochs_ICA[0].info['ch_names']
    # assert epochs.epo2.info['ch_names'] == cleaned_epochs_ICA[1].info['ch_names']


def test_AR_local(epochs):
    """
    Test AR local
    """
    # test on epochs, but usually applied on cleaned epochs with ICA
    ep = [epochs.epo1, epochs.epo2]
    cleaned_epochs_AR = prep.AR_local(ep, verbose=False)
    assert len(epochs.epo1) >= len(cleaned_epochs_AR[0])
    assert len(epochs.epo2) >= len(cleaned_epochs_AR[1])
    assert len(cleaned_epochs_AR[0]) == len(cleaned_epochs_AR[1])


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
    fmin = 10
    fmax = 13
    psd_tuple = analyses.pow(epochs.epo1,
                             fmin, fmax,
                             n_fft=256,
                             n_per_seg=None,
                             epochs_average=False)
    psd = psd_tuple.psd

    statsCondTuple = stats.statsCond(psd, epochs.epo1, 3000, 0.05, 0.05)
    assert statsCondTuple.T_obs.shape[0] == len(epochs.epo1.info['ch_names'])

    for i in range(0, len(statsCondTuple.p_values)):
        assert statsCondTuple.p_values[i] <= statsCondTuple.adj_p[1][i]
    assert statsCondTuple.T_obs_plot.shape[0] == len(
        epochs.epo1.info['ch_names'])

    psd_tuple2 = analyses.pow(epochs.epo2,
                              fmin, fmax,
                              n_fft=256,
                              n_per_seg=None,
                              epochs_average=False)
    psd2 = psd_tuple2.psd
    freq_list = psd_tuple2.freq_list

    data = [psd, psd2]
    con_matrixTuple = stats.con_matrix(epochs.epo1, freq_list, draw=False)
    statscondClusterTuple = stats.statscondCluster(data,
                                                   freq_list,
                                                   scipy.sparse.bsr_matrix(
                                                       con_matrixTuple.ch_con_freq),
                                                   tail=0,
                                                   n_permutations=3000,
                                                   alpha=0.05)
    assert statscondClusterTuple.F_obs.shape[0] == len(
        epochs.epo1.info['ch_names'])
    for i in range(0, len(statscondClusterTuple.clusters)):
        assert len(statscondClusterTuple.clusters[i]) < len(
            epochs.epo1.info['ch_names'])
    assert statscondClusterTuple.cluster_p_values.shape[0] == len(
        statscondClusterTuple.clusters)
    # test F_obs_plot (ntests,) with viz function


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
    ep_hyper_data = ep_hyper.get_data()
    epo1_data = epochs.epo1.get_data()
    epo2_data = epochs.epo2.get_data()
    for i in range(0, len(ep_hyper_data[ne][ch_index1])):
        assert ep_hyper_data[ne][ch_index1][i] == epo1_data[ne][nch][i]
        assert ep_hyper_data[ne][ch_index2][i] == epo2_data[ne][nch][i]
