#!/usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np
import mne
#from hypyp import prep
from hypyp import stats
from hypyp import utils
from hypyp import analyses
from conftest import epochs
from epochs import epo1, epo2, epoch_merge

# TODO: include fixtures for epochs etc.
# TODO: remove () on assert

def test_metaconn():
    """
    Test that con indices are good
    """

    # # Loading data files & extracting sensor infos
    # epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"),
    #                        preload=True)
    # epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"),
    #                        preload=True)
    # mne.epochs.equalize_epoch_counts([epo1, epo2])
    # epoch_merge = utils.merge(epo1, epo2)

    # taking random freq-of-interest to test metaconn_freq
    frequencies = [11, 12, 13]
    # computing ch_con and sensors pairs for metaconn calculation
    ch_con, ch_con_freq = stats.con_matrix(epo1, frequencies, draw=False)
    sensor_pairs = analyses.indexes_connectivity_interbrains(epoch_merge)

    # computing metaconn_freq and test it
    metaconn, metaconn_freq = stats.metaconn_matrix_2brains(
        sensor_pairs, ch_con, frequencies)
    # take a random ch_name:
    random.seed(20)  # Init the random number generator for reproducibility
    # n = random.randrange(0, 63)
    # for our data taske into account EOG ch!!!
    n = random.randrange(0, len(epo1.info['ch_names']))
    tot = len(epo1.info['ch_names'])
    p = random.randrange(len(epo1.info['ch_names']), len(epoch_merge.info['ch_names'])+1)
    # checking for each pair in which ch_name is,
    # whether ch_name linked himself
    # (in neighbouring frequencies also)
    assert(metaconn_freq[n+tot, p] == metaconn_freq[n, p])
    assert(metaconn_freq[n-tot, p] == metaconn_freq[n, p])
    assert(metaconn_freq[n+tot, p+tot] == metaconn_freq[n, p])
    assert(metaconn_freq[n-tot, p-tot] == metaconn_freq[n, p])
    assert(metaconn_freq[n, p+tot] == metaconn_freq[n, p])
    assert(metaconn_freq[n, p-tot] == metaconn_freq[n, p])
    # and not in the other frequencies
    if metaconn_freq[n, p] == 1:
        for i in range(1, len(frequencies)):
            assert(metaconn_freq[n+tot*(i+1), p] != metaconn_freq[n, p])
            assert(metaconn_freq[n-tot*(i+1), p] != metaconn_freq[n, p])
            assert(metaconn_freq[n+tot*(i+1), p+tot*(i+1)] != metaconn_freq[n, p])
            assert(metaconn_freq[n-tot*(i+1), p-tot*(i+1)] != metaconn_freq[n, p])
            assert(metaconn_freq[n, p+tot*(i+1)] != metaconn_freq[n, p])
            assert(metaconn_freq[n, p-tot*(i+1)] != metaconn_freq[n, p])
            # check for each f if connects to the good other ch and not to more
            assert(metaconn_freq[n+tot*i, p+tot*i] == ch_con_freq[n, p-tot])

# def test_intraCSD():
#     """
#     Test that con indices are good
#     """
#     import time

#     # Loading data files & extracting sensor infos
#     epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"),
#                            preload=True)
#     epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"),
#                            preload=True)
#     mne.epochs.equalize_epoch_counts([epo1, epo2])

#     # taking random freq-of-interest to test CSD measures
#     frequencies = [11, 12, 13]
#     # Note: fmin and fmax excluded, here n_freq = 1 (for MNE and Phoebe functions)

#     # intra-ind CSD
#     # data = np.array([epo1, epo1])
#     # data_mne = epo1
#     # sensors = None

#     # inter-ind CSD
#     data = np.array([epo1, epo2])
#     epoch_hyper = utils.merge(epo1, epo2)
#     data_mne = epoch_hyper
#     l = list(range(0,int(len(epoch_hyper.info['ch_names'])/2)))
#     L = []
#     M = []
#     for i in range(0,len(l)):
#         for p in range(0,len(l)):
#             L.append(l[i])
#     M = len(l)*list(range(len(l),len(l)*2))
#     sensors = (np.array(L),np.array(M))

#     # trace running time
#     now = time.time()
#     # mode to transform signal to analytic signal on which synchrony is computed
#     # mode = 'fourier'
#     mode = 'multitaper'

#     # Phoebe: multitaper with mne.time_frequency.tfr_array_multitaper
#     # BUT step = 1s, while coh (including the multitaper step) < 1s... optimized in MNE
#     # how to optimize the mutitaper step in Phoebe script?
#     # and then the second step: same question

#     coh_mne, freqs, tim, epoch, taper = mne.connectivity.spectral_connectivity(data=data_mne,
#                                                                                 method='plv',
#                                                                                 mode=mode,
#                                                                                 indices=sensors,
#                                                                                 sfreq=500,
#                                                                                 fmin=11,
#                                                                                 fmax=13,
#                                                                                 faverage=True)
#     now2 = time.time()
#     # coh = analyses.simple_corr(data, frequencies, mode='plv', epoch_wise=True,
#     #                           time_resolved=True)
#     # substeps cf. multitaper step too long?
#     values = analyses.compute_single_freq(data, frequencies)
#     now3 = time.time()
#     result = analyses.compute_sync(values, mode='plv', epoch_wise=True,
#                           time_resolved=True)
#     now4 = time.time()
#     # convert time to pick seconds only in GTM ref
#     now = time.localtime(now)
#     now2 = time.localtime(now2)
#     now3 = time.localtime(now3)
#     now4 = time.localtime(now4)

#     # assess time running equivalence for each script 
#     # assert (int(now2.tm_sec) - int(now.tm_sec)) == (int(now3.tm_sec) - int(now2.tm_sec))
#     # takes 2 versus 0 seconds (MNE) (and here n_channels 31, n_epochs not a lot, n_freq 1)
#     # idem en inter-ind

#     # test substeps
#     assert (int(now2.tm_sec) - int(now.tm_sec)) == ((int(now4.tm_sec) - int(now3.tm_sec))+(int(now3.tm_sec) - int(now2.tm_sec)))
#     # one second per step in Phoebe script...
#     # test mne.time.frequencies.tfr_multitaper(all) = 1 second?
#     # spectral con with mode = 'multitaper' < 1s...

#     # assess results: shape equivalence and values
#     # not same output: MNE pairs of electrode (n_connections=31*31, freq=1), Phoebe (31, 31, 1)
#     # assert coh[0][0][0] == coh_mne[0][0]

def test_ICAfit():
    """
    Test ICA fit function
    """
    # Loading data files & extracting sensor infos
    epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"),
                           preload=True)
    epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"),
                           preload=True)
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    epochs = [epo1, epo2]

    icas = prep.ICA_fit(epochs, n_components=15, method='fastica', random_state=97)
    # check that the number of componenents is similar between the two subjects
    for i in range(0, len(icas)-1):
        assert len(icas[i]) == len(icas[i+1])
    # check whether epochs.info same length > ICA components same length