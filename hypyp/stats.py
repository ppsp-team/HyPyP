#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : prep.py
# description     : data preprocessing functions
# author          : Florence Brun, Guillaume Dumas
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

import mne
import numpy as np
import matplotlib.pylab as plt
from scipy import sparse
from mne.channels import find_ch_connectivity
from mne.stats import permutation_cluster_test


def statsCond(PSDs_task_normLog, epochs, n_permutations, alpha_bonferroni, alpha):
    """Compute statistical t test on Power Spectral Density values for a condition.

    Note that this ttest calculates if the observed mean significantly deviates
    from 0, it does not compare two periods, but one period with the null
    hypothesis. Randomized data are generated with random sign flips.
    The tail is set to 0 by default (= the alternative hypothesis is that mean
    of the data is different from 0).

    To reduce false positive due to multiple comparisons, bonferroni correction
    is applied to the p values.

    Note that the frequency dimension is reduced to one for the test (average
    in the frequency band-of-interest). To take frequencies into account, use
    cluster statistics (see statscondCluster function in the toolbox).

    For vizualisation, use plot_significant_sensors function in the toolbox.


    Parameters
    -----
    PSDs_task_normLog: array of subjects PSD Logratio (ndarray) for a condition
    (n_samples, n_tests with n_tests the different channels).

    epochs: Epochs object for a condition from a random subject, only used to
    get parameters from the info (sampling frequencies for example).

    n_permutations: the number of permutations, int. Should be at least 2*n
    sample, can be set to 50000 for example.

    alpha_bonferroni: the threshold for bonferroni correction, int.
    Can be set to 0.05.

    alpha: the threshold for ttest, int. Can be set to 0.05.

    Returns
    -----
    T_obs: T-statistic observed for all variables, array of shape (n_tests).

    p_values: p-values for all the tests, array of shape (n_tests).

    H0: T-statistic obtained by permutations and t-max trick for multiple
    comparison, array of shape (n_permutations).

    adj_p: adjusted p values from bonferroni correction, array of shape
    (n_tests, n_tests), with boolean assessment for p values and p values
    corrected.

    T_obs_plot : satistical values to plot, from sensors above alpha threshold,
    array of shape (n_tests,).

    """
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

    # getting sensors position
    pos = np.array([[0, 0]])
    for i in range(0, len(epochs.info['ch_names'])):
        cor = np.array([epochs.info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis=0)
    pos = pos[1:]

    return T_obs, p_values, H0, adj_p, T_obs_plot

def con_matrix(epochs, freqs_mean):
    """Compute a priori channels connectivity across space and frequencies.

    Parameters
    -----
    epochs : one subject Epochs object to sample channels information in info.

    freqs_mean : list of frequencies in frequency-band-of-interest used by MNE
    for power or coherence spectral density calculation.

    Returns
    -----
    ch_con : connectivity matrix between sensors along space based on their
    position, scipy.sparse.csr_matrix of shape (n_channels, n_channels).

    ch_con_freq : connectivity matrix between sensors along space and
    frequencies, scipy.sparse.csr_matrix of shape (n_channels*len(freqs_mean),
    n_channels*len(freqs_mean)).

    """

    # creating channels connectivity matrix in space
    ch_con, ch_names_con = find_ch_connectivity(epochs.info,
                                                ch_type='eeg')
    # vizualising the matrix and transforming it into array
    plt.spy(ch_con)
    ch_con_arr = ch_con.toarray()

    # duplicating the array 'freqs_mean' or 'freqs' times (PSD or CSD)
    # to take channels connectivity across frequencies into account
    l_freq = len(freqs_mean)
    ch_con_freq = np.tile(ch_con_arr, (l_freq, l_freq))
    # vizualising the array
    plt.spy(ch_con_freq)

    return ch_con, ch_con_freq



def metacon_matrix_2brains(electrodes, ch_con, freqs_mean):
    """Compute a priori connectivity across space and frequencies
    between pairs of sensors for which connectivity indices have
    been calculated, for merge data (2 brains).

    Note that it has been assumed that there was no a priori connectivity
    between electrodes from the 2 subjects.


    Parameters
    -----
    electrodes : electrodes pairs for which connectivity indices have
    been computed, list of tuples with channels indexes.

    ch_con : connectivity matrix between sensors along space based on their
    position, scipy.sparse.csr_matrix of shape (n_channels, n_channels).

    freqs_mean : list of frequencies in the frequency-band-of-interest used
    by MNE for coherence spectral density calculation (connectivity indices).

    Returns
    -----
    metaconn : a priori connectivity based on sensors location, between pairs
    of sensors for which connectivity indices have been calculated, for merge
    data, matrix of shape (len(electrodes), len(electrodes)).

    metaconn_freq : a priori connectivity between pairs of sensors for which
    connectivity indices have been calculated, across space and frequencies,
    for merge data, matrix of shape (len(electrodes)*len(freqs_mean),
    len(electrodes)*len(freqs_mean)).

    """

    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11,e12) in enumerate(electrodes):
        for ne2, (e21,e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            # considering no a priori connectivity between the 2 brains
            metaconn[ne1, ne2] = (((ch_con[e11,e21]) and (ch_con[e12,e22])) or
                                  ((ch_con[e11,e21]) and (e12 == e22)) or
                                  ((ch_con[e12,e22]) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channels connectivity
    # across frequencies into account
    l_freq = len(freqs_mean)
    metaconn_freq = np.tile(metaconn, (l_freq, l_freq))
    # vizualising the array
    plt.spy(metaconn_freq)

    return metaconn, metaconn_freq


def metacon_matrix(electrodes, ch_con, freqs_mean):
    """Compute a priori connectivity between pairs of sensors for which
    connectivity indices have been calculated, across space and frequencies
    (based on sensors location).


    Parameters
    -----
    electrodes : electrodes pairs for which connectivity has been computed,
    list of tuples with channels indexes.

    ch_con : connectivity matrix between sensors along space based on their
    position, scipy.sparse.csr_matrix of shape (n_channels, n_channels).

    freqs_mean : list of frequencies in the frequency-band-of-interest used
    by MNE for coherence spectral density calculation (connectivity indices).

    Returns
    -----
    metaconn : a priori connectivity based on sensors location, between pairs
    of sensors for which connectivity indices have been calculated,
    matrix of shape (len(electrodes), len(electrodes)).

    metaconn_freq : a priori connectivity between pairs of sensors for which
    connectivity indices have been calculated, across space and frequencies,
    for merge data, matrix of shape (len(electrodes)*len(freqs_mean),
    len(electrodes)*len(freqs_mean)).

    """

    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11,e12) in enumerate(electrodes):
        for ne2, (e21,e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            metaconn[ne1, ne2] = (((ch_con[e11,e21]) and (ch_con[e12,e22])) or
                                  ((ch_con[e11,e22]) and (ch_con[e12,e21])) or
                                  ((ch_con[e11,e21]) and (e12 == e22)) or
                                  ((ch_con[e12,e21]) and (e11 == e22)) or
                                  ((ch_con[e11,e22]) and (e12 == e21)) or 
                                  ((ch_con[e12,e22]) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channels connectivity
    # across frequencies into account
    l_freq = len(freqs_mean)
    metaconn_freq = np.tile(metaconn, (l_freq, l_freq))
    # vizualising the array
    plt.spy(metaconn_freq)

    return metaconn, metaconn_freq


def statscondCluster(data, freqs_mean, ch_con_freq, tail, n_permutations, alpha):
    """Compute cluster-level statistical permutation test, corrected with
    channels connectivity across space and frequencies.

    Parameters
    -----
    data : values from different conditions or different groups to compare,
    list of arrays (3d for time-frequency power or connectivity values).

    freqs_mean : frequencies in frequency-band-of-interest used by MNE for PSD
    or CSD calculation.

    ch_con_freq : connectivity or metaconnectivity matrix for PSD or CSD
    values to assess a priori connectivity between sensors across space and
    frequencies, based on their position.

    tail : direction of the ttest, can be set to 1, 0 or -1.

    n_permutations : number of permutations computed, can be set to 50000.

    alpha : threshold to consider clusters significant, can be set to 0.05
    or less.

    Returns
    -----
    F_obs : statistic (F by default) observed for all variables,
    array of shape (n_tests,).

    clusters : list where each sublist contains the indices of locations
    that together form a cluster, list.

    cluster_pv : p-value for each cluster, array.

    H0 : max cluster level stats observed under permutation, array of
    shape (n_permutations,).

    F_obs_plot : satistical values above alpha threshold, to plot significant
    sensors (see plot_significant_sensors function in the toolbox), array
    of shape (n_tests,).

    """

    # computing the cluster permutation t test
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(data,
                                                                     threshold=None,
                                                                     n_permutations=n_permutations,
                                                                     tail=tail, connectivity=ch_con_freq,
                                                                     t_power=1, out_type='indices')
    # t_power = 1 weights each location by its statistical score,
    # when set to 0 it gives a count of locations in each cluster

    # getting significant clusters for vizulisation
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c in cluster_p_values:
        if c <= alpha:
            i = np.where(cluster_p_values == c)
            F_obs_plot[i] = F_obs[i]
    F_obs_plot = np.nan_to_num(F_obs_plot)

    return F_obs, clusters, cluster_p_values, H0, F_obs_plot
