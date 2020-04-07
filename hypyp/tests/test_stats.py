#!/usr/bin/env python
# coding=utf-8

import os
import random
import mne
from hypyp import stats
from hypyp import utils
from hypyp import analyses


def test_metaconn():
    """
    Test that con indices are good
    """

    # Loading data files & extracting sensor infos
    epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"),
                           preload=True)
    epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"),
                           preload=True)
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    epoch_merge = utils.merge(epo1, epo2)

    # taking random freq-of-interest to test metaconn_freq
    frequencies = [11, 12, 13]
    # computing ch_con and sensors pairs for metaconn calculation
    ch_con, ch_con_freq = stats.con_matrix(epo1, frequencies, draw=False)
    sensor_pairs = analyses.indexes_connectivity_interbrains(epoch_merge)

    # computing metaconn_freq and test it
    metaconn, metaconn_freq = stats.metaconn_matrix_2brains(
        sensor_pairs, ch_con, frequencies)
    # take a random ch_name:
    random.seed(42)  # Init the random number generator for reproducibility
    n = random.randrange(0, 63)
    p = random.randrange(63, 125)
    # checking for each pair in which ch_name is,
    # whether ch_name linked himself
    # (in neighbouring frequencies also)
    assertEqual(metaconn_freq[n+62, p], metaconn_freq[n, p])
    assertEqual(metaconn_freq[n-62, p], metaconn_freq[n, p])
    assertEqual(metaconn_freq[n+62, p+62], metaconn_freq[n, p])
    assertEqual(metaconn_freq[n-62, p-62], metaconn_freq[n, p])
    assertEqual(metaconn_freq[n, p+62], metaconn_freq[n, p])
    assertEqual(metaconn_freq[n, p-62], metaconn_freq[n, p])
    # and not in the other frequencies
    for i in range(1, len(frequencies)):
        assertFalse(metaconn_freq[n+62*(i+1), p] == metaconn_freq[n, p])
        assertFalse(metaconn_freq[n-62*(i+1), p] == metaconn_freq[n, p])
        assertFalse(metaconn_freq[n+62*(i+1), p+62*(i+1)] == metaconn_freq[n, p])
        assertFalse(metaconn_freq[n-62*(i+1), p-62*(i+1)] == metaconn_freq[n, p])
        assertFalse(metaconn_freq[n, p+62*(i+1)] == metaconn_freq[n, p])
        assertFalse(metaconn_freq[n, p-62*(i+1)] == metaconn_freq[n, p])
        # check for each f if connects to the good other ch and not to more
        assertEqual(metaconn_freq[n+62*i, p+62*i], ch_con_freq[n, p-62])
