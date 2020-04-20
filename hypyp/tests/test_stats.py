#!/usr/bin/env python
# coding=utf-8

import os
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
    # TODO: test metaconn_matrix and con_matrix
    # taking random freq-of-interest to test metaconn_freq
    freq = [11, 12, 13]
    # computing ch_con and sensors pairs for metaconn calculation
    ch_con, ch_con_freq = stats.con_matrix(epochs.epo1, freq, draw=False)
    sensor_pairs = analyses.indexes_connectivity_interbrains(epochs.epoch_merge)

    # computing metaconn_freq and test it
    _, metaconn_freq = stats.metaconn_matrix_2brains(sensor_pairs,
                                                     ch_con, freq)
    # take a random ch_name:
    random.seed(20)  # Init the random number generator for reproducibility
    # n = random.randrange(0, 63)
    # for our data taske into account EOG ch!!!
    n = random.randrange(0, len(epochs.epo1.info['ch_names']))
    tot = len(epochs.epo1.info['ch_names'])
    p = random.randrange(len(epochs.epo1.info['ch_names']), len(epochs.epoch_merge.info['ch_names'])+1)
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
            assert metaconn_freq[n+tot*(i+1), p+tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n-tot*(i+1), p-tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n, p+tot*(i+1)] != metaconn_freq[n, p]
            assert metaconn_freq[n, p-tot*(i+1)] != metaconn_freq[n, p]
            # check for each f if connects to the good other ch and not to more
            assert metaconn_freq[n+tot*i, p+tot*i] == ch_con_freq[n, p-tot]


def test_simple_corr(epochs):
    """
    Test simple_corr timing
    """
    import time

    # taking random freq-of-interest to test CSD measures
    frequencies = [11, 12, 13]
    # Note: fmin and fmax excluded, here n_freq = 1
    # (for MNE and Phoebe functions)

    # intra-ind CSD
    # data = np.array([epo1, epo1])
    # data_mne = epo1
    # sensors = None

    # inter-ind CSD
    data = np.array([epochs.epo1, epochs.epo2])
    data_mne = epochs.epoch_merge

    l = list(range(0,
                   int(len(epochs.epoch_merge.info['ch_names'])/2)))
    L = []
    M = []
    for i in range(0, len(l)):
        for p in range(0, len(l)):
            L.append(l[i])
    M = len(l)*list(range(len(l), len(l)*2))
    sensors = (np.array(L), np.array(M))

    # trace running time
    now = time.time()
    # mode to transform signal to analytic signal on which
    # synchrony is computed
    # mode = 'fourier'
    mode = 'multitaper'

    # Phoebe: multitaper: mne.time_frequency.tfr_array_multitaper
    # BUT step = 1s, while coh (including the multitaper step) < 1s...
    # optimized in MNE
    # how to optimize the mutitaper step in Phoebe script?
    # and then the second step: same question

    coh_mne,_,_,_,_ = mne.connectivity.spectral_connectivity(data=data_mne,
                                                             method='plv',
                                                             mode=mode,
                                                             indices=sensors,
                                                             sfreq=500,
                                                             fmin=11,
                                                             fmax=13,
                                                             faverage=True)
    now2 = time.time()
    # coh = analyses.simple_corr(data, frequencies, mode='plv',
    #                            epoch_wise=True,
    #                            time_resolved=True)
    # substeps cf. multitaper step too long?
    values = analyses.compute_single_freq(data, frequencies)
    now3 = time.time()
    result = analyses.compute_sync(values, mode='plv',
                                   epoch_wise=True,
                                   time_resolved=True)
    now4 = time.time()
    # convert time to pick seconds only in GTM ref
    now = time.localtime(now)
    now2 = time.localtime(now2)
    now3 = time.localtime(now3)
    now4 = time.localtime(now4)

    # assess time running equivalence for each script
    # assert (int(now2.tm_sec) - int(now.tm_sec)) == (int(now3.tm_sec) - int(now2.tm_sec))
    # takes 2 versus 0 second with MNE function
    # (and here n_channels 31, n_epochs not a lot, n_freq 1)
    # idem en inter-ind

    # test substeps
    assert (int(now2.tm_sec) - int(now.tm_sec)) == ((int(now4.tm_sec) - int(now3.tm_sec))+(int(now3.tm_sec) - int(now2.tm_sec)))
    # each of the two steps in Phoebe script takes 1 second
    # (while first step less than 1s for the MNE function cf. multitaper
    # calculation... = ?)

    # PB: test not stable, time changes!!

    # assess results: shape equivalence and values
    # not same output: MNE pairs of electrode (n_connections=31*31, freq=1),
    # Phoebe (31, 31, 1)
    # assert coh[0][0][0] == coh_mne[0][0]


def test_ICA(epochs):
    """
    Test ICA fit, ICA choice comp and ICA apply
    """
    ep = [epochs.epo1, epochs.epo2]
    icas = prep.ICA_fit(ep, n_components=15, method='fastica', random_state=97)
    # check that the number of componenents is similar between the two subjects
    for i in range(0, len(icas)-1):
        mne.preprocessing.ICA.get_components(icas[i]).shape == mne.preprocessing.ICA.get_components(icas[i+1]).shape
    # check corrmap ok on the 2 subj
    # check component choice good cf. compare with find eog or find ecg comp
    # cleaned_epochs_ICA = prep.ICA_choice_comp(icas, ep)
    # can not resolve the interactive window !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    # modify parameters cf. Denis email and check if works


# def test_filt():
#     check that output is always Raw (list of raws)
#     and cf. PSD: lfreq and hfreq OK
#     can not test it cf. do not have raws,
#     but only epochs for examples.


def test_PSD(epochs):
    """
    Test PSD
    """
    fmin = 10
    fmax = 13
    freqs_mean, PSD_welch = analyses.PSD(epochs.epo1,
                                         fmin, fmax,
                                         time_resolved=True)
    assert type(PSD_welch) == np.ndarray
    assert PSD_welch.shape == (len(epochs.epo1.info['ch_names']), len(freqs_mean))
    freqs_mean, PSD_welch = analyses.PSD(epochs.epo1,
                                         fmin, fmax,
                                         time_resolved=False)
    assert PSD_welch.shape == (len(epochs.epo1), len(epochs.epo1.info['ch_names']), len(freqs_mean))


def test_indexes_connectivity(epochs):
    """
    Test index intra- and inter-brains
    """
    electrodes = analyses.indexes_connectivity_intrabrain(epochs.epo1)
    length = len(epochs.epo1.info['ch_names'])
    L = []
    for i in range(1, length):
        L.append(length-i)
    tot = sum(L)
    assert len(electrodes) == tot
    electrodes_hyper = analyses.indexes_connectivity_interbrains(epochs.epoch_merge)
    assert len(electrodes_hyper) == length*length
    # check output type: list of tuples
    # format that do not work for mne.spectral_connectivity #TODO: change that
    # cf. do not needed for Phoebe simple_corr function!


def test_stats(epochs):
    """
    Test stats
    """
    fmin = 10
    fmax = 13
    freqs_mean, PSD_welch = analyses.PSD(epochs.epo1,
                                         fmin, fmax,
                                         time_resolved=False)

    statsCondTuple = stats.statsCond(PSD_welch, epochs.epo1, 3000, 0.05, 0.05)
    assert statsCondTuple.T_obs.shape[0] == len(epochs.epo1.info['ch_names'])
    # TODO: add an assert in the function to be sure PSD with epochs
    # len(shape) = 3
    # and retest with time_resolved=True
    for i in range(0, len(statsCondTuple.p_values)):
        assert statsCondTuple.p_values[i] <= statsCondTuple.adj_p[1][i]
    assert statsCondTuple.T_obs_plot.shape[0] == len(epochs.epo1.info['ch_names'])
    # test T_obs_plot with viz function

    _, PSD_welch2 = analyses.PSD(epochs.epo2,
                                 fmin, fmax,
                                 time_resolved=False)
    data = [PSD_welch, PSD_welch2]
    con_matrixTuple = stats.con_matrix(epochs.epo1, freqs_mean, draw=False)
    statscondClusterTuple = stats.statscondCluster(data,
                                                   freqs_mean,
                                                   scipy.sparse.bsr_matrix(con_matrixTuple.ch_con_freq),
                                                   tail=0,
                                                   n_permutations=3000,
                                                   alpha=0.05)
    assert statscondClusterTuple.F_obs.shape[0] == len(epochs.epo1.info['ch_names'])
    for i in range(0, len(statscondClusterTuple.clusters)):
        assert len(statscondClusterTuple.clusters[i]) < len(epochs.epo1.info['ch_names'])
    assert statscondClusterTuple.cluster_p_values.shape[0] == len(statscondClusterTuple.clusters)
    # test F_obs_plot (ntests,) with viz function


def test_utils(epochs):
    """
    Test merge and split
    """
    ep_hyper = utils.merge(epochs.epo1, epochs.epo2)
    assert type(ep_hyper) == mne.epochs.EpochsArray
    # check channels number
    assert len(ep_hyper.info['ch_names']) == 2*len(epochs.epo1.info['ch_names'])
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

    # split test
    # but done on raws... to preprocess subjects indep...
    # to be adapted for epochs?
    # raw_1020_S1, raw_1020_S2 = utils.merge(raw_merge)
    # check that raw1 = epo1 directly?

# test viz
