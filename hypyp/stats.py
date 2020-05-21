#!/usr/bin/env python
# coding=utf-8

"""
Statistical functions

| Option | Description |
| ------ | ----------- |
| title           | stats.py |
| authors         | Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


from collections import namedtuple
import numpy as np
import scipy
import matplotlib.pylab as plt
import mne
from mne.channels import find_ch_connectivity
from mne.stats import permutation_cluster_test


def statsCond(PSDs_task_normLog: np.ndarray, epochs: mne.Epochs, n_permutations: int, alpha_bonferroni: float, alpha: float) -> tuple:
    """
    Computes statistical t test on Power Spectral Density values
    (PSD) for a condition.

    Arguments:
        PSDs_task_normLog: array of subjects PSD Logratio (ndarray) for
          a condition (n_samples, n_tests, nfreq: n_tests the channels).
          PSD values will be averaged on nfreq for statistics.
        epochs: Epochs object for a condition from a random subject, only
          used to get parameters from the info (sampling frequencies for example).
        n_permutations: the number of permutations, int. Should be at least 2*n
          sample, can be set to 50000 for example.
        alpha_bonferroni: the threshold for bonferroni correction, float, can be set to 0.05.
        alpha: the threshold for ttest, float, can be set to 0.05.

    Note:
        This ttest calculates if the observed mean significantly deviates
        from 0; it does not compare two periods, but one period with the null
        hypothesis. Randomized data are generated with random sign flips.
        The tail is set to 0 by default (= the alternative hypothesis is that
        the data mean is different from 0).
        To reduce false positive due to multiple comparisons, bonferroni
        correction is applied to the p values.
        Note that the frequency dimension is reduced to one for the test
        (average in the frequency band-of-interest).
        To take frequencies into account, use cluster statistics
        (see statscondCluster function in the toolbox).
        For visualization, use plot_significant_sensors function in the toolbox.

    Returns:
        T_obs, p_values, H0, adj_p, T_obs_plot:
        - T_obs: T-statistic observed for all variables, array of shape (n_tests).

        - p_values: p-values for all the tests, array of shape (n_tests).

        - H0: T-statistic obtained by permutations and t-max trick for multiple
          comparisons, array of shape (n_permutations).

        - adj_p: adjusted p values from bonferroni correction, array of shape
          (n_tests, n_tests), with boolean assessment for p values
          and p values corrected.

        - T_obs_plot: statistical values to plot, from sensors above alpha threshold,
          array of shape (n_tests,).
    """
    # checking whether data have the same size
    assert(len(PSDs_task_normLog.shape) == 3), "PSD does not have the appropriate shape!"

    # averaging across frequencies (compute stats only in ch space)
    power = np.mean(PSDs_task_normLog, axis=2)
    T_obs, p_values, H0 = mne.stats.permutation_t_test(power, n_permutations,
                                                       tail=0, n_jobs=1)
    adj_p = mne.stats.bonferroni_correction(p_values, alpha=alpha_bonferroni)

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c in adj_p[1]:
        if c <= alpha:
            i = np.where(adj_p[1] == c)
            T_obs_plot[i] = T_obs[i]
    T_obs_plot = np.nan_to_num(T_obs_plot)

    # retrieving sensor position
    pos = np.array([[0, 0]])
    for i in range(0, len(epochs.info['ch_names'])):
        cor = np.array([epochs.info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis=0)
    pos = pos[1:]

    statsCondTuple = namedtuple(
        'statsCond', ['T_obs', 'p_values', 'H0', 'adj_p', 'T_obs_plot'])

    return statsCondTuple(
        T_obs=T_obs,
        p_values=p_values,
        H0=H0,
        adj_p=adj_p,
        T_obs_plot=T_obs_plot)


def con_matrix(epochs: mne.Epochs, freqs_mean: list, draw: bool = False) -> tuple:
    """
    Computes a priori channel connectivity across space and frequencies.

    Arguments:
        epochs: one subject Epochs object; contains channel information.
        freqs_mean: list of frequencies in frequency-band-of-interest used
          by MNE for power or coherence spectral density calculation.
        draw: option to plot the connectivity matrices, boolean.

    Returns:
        ch_con, ch_con_freq:

        - ch_con: connectivity matrix between channels along space based on
          their position, scipy.sparse.csr_matrix of shape
          (n_channels, n_channels).

        - ch_con_freq: connectivity matrix between channels along space and
          frequencies, scipy.sparse.csr_matrix of shape
          (n_channels*len(freqs_mean), n_channels*len(freqs_mean)).
    """

    # creating channel-to-channel connectivity matrix in space
    ch_con, ch_names_con = find_ch_connectivity(epochs.info,
                                                ch_type='eeg')

    ch_con_arr = ch_con.toarray()

    # duplicating the array 'freqs_mean' or 'freqs' times (PSD or CSD)
    # to take channel connectivity across neighboring frequencies into
    # account
    l_freq = len(freqs_mean)
    init = np.zeros((l_freq*len(ch_names_con),
                     l_freq*len(ch_names_con)))
    for i in range(0, l_freq*len(ch_names_con)):
        for p in range(0, l_freq*len(ch_names_con)):
            if (p//len(ch_names_con) == i//len(ch_names_con)) or (p//len(ch_names_con) == i//len(ch_names_con) + 1) or (p//len(ch_names_con) == i//len(ch_names_con) - 1):
                init[i][p] = 1

    ch_con_mult = np.tile(ch_con_arr, (l_freq, l_freq))
    ch_con_freq = np.multiply(init, ch_con_mult)

    if draw:
        plt.figure()
        # visualizing the matrix and transforming it into array
        plt.subplot(1, 2, 1)
        plt.spy(ch_con)
        plt.title("Connectivity matrix")
        plt.subplot(1, 2, 2)
        plt.spy(ch_con_freq)
        plt.title("Meta-connectivity matrix")

    con_matrixTuple = namedtuple('con_matrix', ['ch_con', 'ch_con_freq'])

    return con_matrixTuple(
        ch_con=ch_con,
        ch_con_freq=ch_con_freq)


def metaconn_matrix_2brains(electrodes: list, ch_con: scipy.sparse.csr_matrix, freqs_mean: list, plot: bool = False) -> tuple:
    """
    Computes a priori connectivity across space and frequencies
    between pairs of channels for which connectivity indices have
    been calculated, to merge data (2 brains).

    Arguments:
        electrodes: electrode pairs for which connectivity indices have
          been computed, list of tuples with channels indexes, see
          indices_connectivity_interbrain function in toolbox (analyses).
        ch_con: connectivity matrix between channels along space based on their
          position, scipy.sparse.csr_matrix of shape (n_channels, n_channels).
        freqs_mean: list of frequencies in the frequency-band-of-interest used
          by MNE for coherence spectral density calculation (connectivity indices).
        plot: option to plot the connectivity matrices, boolean.

    Note:
        It is assumed that there was no a priori connectivity
        between channels from the two subjects.

    Returns:
        metaconn, metaconn_freq:

        - metaconn: a priori connectivity based on channel location, between
          pairs of channels for which connectivity indices have been calculated,
          to merge data, matrix of shape (len(electrodes), len(electrodes)).

        - metaconn_freq: a priori connectivity between pairs of channels for which
          connectivity indices have been calculated, across space and
          frequencies, to merge data, matrix of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean)).
    """

    n = np.max(electrodes, axis=0)[0]+1
    # n = 62
    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11, e12) in enumerate(electrodes):
        for ne2, (e21, e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            # considering no a priori connectivity between the 2 brains
            metaconn[ne1, ne2] = (((ch_con[e11, e21]) and (ch_con[e12-n, e22-n])) or
                                  ((ch_con[e11, e21]) and (e12 == e22)) or
                                  ((ch_con[e12-n, e22-n]) and (e11 == e21)) or
                                  ((e12 == e22) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channel connectivity
    # across neighboring frequencies into account
    l_freq = len(freqs_mean)

    init = np.zeros((l_freq*len(electrodes),
                     l_freq*len(electrodes)))
    for i in range(0, l_freq*len(electrodes)):
        for p in range(0, l_freq*len(electrodes)):
            if (p//len(electrodes) == i//len(electrodes)) or (p//len(electrodes) == i//len(electrodes) + 1) or (p//len(electrodes) == i//len(electrodes) - 1):
                init[i][p] = 1

    metaconn_mult = np.tile(metaconn, (l_freq, l_freq))
    metaconn_freq = np.multiply(init, metaconn_mult)

    if plot:
        # vizualising the array
        plt.figure()
        plt.spy(metaconn_freq)
        plt.title("Meta-connectivity matrix")

    metaconn_matrix_2brainsTuple = namedtuple(
        'metaconn_matrix_2brains', ['metaconn', 'metaconn_freq'])

    return metaconn_matrix_2brainsTuple(
        metaconn=metaconn,
        metaconn_freq=metaconn_freq)


def metaconn_matrix(electrodes: list, ch_con: scipy.sparse.csr_matrix, freqs_mean: list) -> tuple:
    """
    Computes a priori connectivity between pairs of sensors for which
    connectivity indices have been calculated, across space and frequencies
    (based on channel location).

    Arguments:
        electrodes: electrode pairs for which connectivity has been computed,
          list of tuples with channel indices, see indices_connectivity
          intrabrain function in toolbox (analyses).
        ch_con: connectivity matrix between sensors along space based on their
          position, scipy.sparse.csr_matrix of shape (n_channels, n_channels).
        freqs_mean: list of frequencies in the frequency-band-of-interest used
          by MNE for coherence spectral density calculation (connectivity indices).

    Returns:
        metaconn, metaconn_freq:

        - metaconn: a priori connectivity based on channel location, between
          pairs of channels for which connectivity indices have been calculated,
          matrix of shape (len(electrodes), len(electrodes)).

        - metaconn_freq: a priori connectivity between pairs of channels for which
          connectivity indices have been calculated, across space and
          frequencies, for merge data, matrix of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean)).
    """

    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11, e12) in enumerate(electrodes):
        for ne2, (e21, e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            metaconn[ne1, ne2] = (((ch_con[e11, e21]) and (ch_con[e12, e22])) or
                                  ((ch_con[e11, e22]) and (ch_con[e12, e21])) or
                                  ((ch_con[e11, e21]) and (e12 == e22)) or
                                  ((ch_con[e11, e22]) and (e12 == e21)) or
                                  ((ch_con[e12, e21]) and (e11 == e22)) or
                                  ((ch_con[e12, e22]) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channels connectivity
    # across neighboring frequencies into account
    l_freq = len(freqs_mean)

    init = np.zeros((l_freq*len(electrodes),
                     l_freq*len(electrodes)))
    for i in range(0, l_freq*len(electrodes)):
        for p in range(0, l_freq*len(electrodes)):
            if (p//len(electrodes) == i//len(electrodes)) or (p//len(electrodes) == i//len(electrodes) + 1) or (p//len(electrodes) == i//len(electrodes) - 1):
                init[i][p] = 1

    metaconn_mult = np.tile(metaconn, (l_freq, l_freq))
    metaconn_freq = np.multiply(init, metaconn_mult)

    # TODO: option with verbose
    # vizualising the array
    plt.spy(metaconn_freq)

    metaconn_matrixTuple = namedtuple(
        'metaconn_matrix', ['metaconn', 'metaconn_freq'])

    return metaconn_matrixTuple(
        metaconn=metaconn,
        metaconn_freq=metaconn_freq)


def statscondCluster(data: list, freqs_mean: list, ch_con_freq: scipy.sparse.csr_matrix, tail: int, n_permutations: int, alpha: float) -> tuple:
    """
    Computes cluster-level statistical permutation test, corrected with
    channel connectivity across space and frequencies.

    Arguments:
        data: values from different conditions or different groups to compare,
          list of arrays (3d for time-frequency power or connectivity values).
        freqs_mean: frequencies in frequency-band-of-interest used by MNE
          for PSD or CSD calculation, list.
        ch_con_freq: connectivity or metaconnectivity matrix for PSD or CSD
          values to assess a priori connectivity between channels across
          space and frequencies based on their position, bsr_matrix.
        tail: direction of the ttest, can be set to 1, 0 or -1.
        n_permutations: number of permutations computed, can be set to 50000.
        alpha: threshold to consider clusters significant, can be set to 0.05
          or less.

    Returns:
        F_obs, clusters, cluster_pv, H0, F_obs_plot:

        - F_obs: statistic (F by default) observed for all variables,
          array of shape (n_tests,).

        - clusters: list where each sublist contains the indices of locations
          that together form a cluster, list.

        - cluster_p_values: p-value for each cluster, array.

        - H0: max cluster level stats observed under permutation, array of
          shape (n_permutations,).

        - F_obs_plot: statistical values above alpha threshold, to plot
          significant sensors (see plot_significant_sensors function in the toolbox)
          array of shape (n_tests,).
    """

    # computing the cluster permutation t test
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(data,
                                                                     threshold=None,
                                                                     n_permutations=n_permutations,
                                                                     tail=tail, connectivity=ch_con_freq,
                                                                     t_power=1, out_type='indices')
    # t_power = 1 weighs each location by its statistical score,
    # when set to 0 it gives a count of locations in each cluster

    # getting significant clusters for visualization
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c in cluster_p_values:
        if c <= alpha:
            i = np.where(cluster_p_values == c)
            F_obs_plot[i] = F_obs[i]
    F_obs_plot = np.nan_to_num(F_obs_plot)

    statscondClusterTuple = namedtuple('statscondCluster', [
                                       'F_obs', 'clusters', 'cluster_p_values', 'H0', 'F_obs_plot'])

    return statscondClusterTuple(
        F_obs=F_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        F_obs_plot=F_obs_plot)
