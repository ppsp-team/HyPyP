#!/usr/bin/env python
# coding=utf-8

import os
import random
from hypyp import stats
from hypyp import utils
from hypyp import analyses


def test_metacon(self):
    """
    Test that con indices are good
    """
    frequencies = [11, 12, 13]
    ch_con, ch_con_freq = stats.con_matrix(epoch_S1[0], frequencies)
    sensor_pairs = analyses.indexes_connectivity_interbrains(epoch_merge)
    metaconn, metaconn_freq = stats.metaconn_matrix_2brains(
        sensor_pairs, ch_con, frequencies)
    # take a random ch_name:
    random.seed(42)  # Init the random number generator for reproducibility
    n = random.randrange(0, 63)
    p = random.randrange(63, 125)
    # for each pair in which ch_name is, check
    # whether ch_name linked with himself, also
    # in neighbouring frequencies
    self.assertEqual(metaconn_freq[n+62, p], metaconn_freq[n, p])
    self.assertEqual(metaconn_freq[n-62, p], metaconn_freq[n, p])
    self.assertEqual(metaconn_freq[n+62, p+62], metaconn_freq[n, p])
    self.assertEqual(metaconn_freq[n-62, p-62], metaconn_freq[n, p])
    self.assertEqual(metaconn_freq[n, p+62], metaconn_freq[n, p])
    self.assertEqual(metaconn_freq[n, p-62], metaconn_freq[n, p])
    # and not in the other frequencies
    for i in range(1, len(frequencies)):
        self.assertFalse(
            metaconn_freq[n+62*(i+1), p] == metaconn_freq[n, p])
        self.assertFalse(
            metaconn_freq[n-62*(i+1), p] == metaconn_freq[n, p])
        self.assertFalse(
            metaconn_freq[n+62*(i+1), p+62*(i+1)] == metaconn_freq[n, p])
        self.assertFalse(
            metaconn_freq[n-62*(i+1), p-62*(i+1)] == metaconn_freq[n, p])
        self.assertFalse(
            metaconn_freq[n, p+62*(i+1)] == metaconn_freq[n, p])
        self.assertFalse(
            metaconn_freq[n, p-62*(i+1)] == metaconn_freq[n, p])
        # check for each f if connects to the good other ch
        # and not to more
        self.assertEqual(
            metaconn_freq[n+62*i, p+62*i], ch_con_freq[n, p-62])
