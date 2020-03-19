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


def statsCond(PSDs_task_normLog, Epochs_task,n_permutations, alpha_bonferroni, alpha):

    """Compute statistical t test on Power Spectral Density values for a condition.
    Note that this ttest calculates if the observed mean significantly deviates
    from 0, it does not compare two periods, but one period with the null
    hypothesis. Randomized data are generated with random sign flips.
    The tail is set to 0 by default (the alternative hypothesis is that
    data mean is different from 0).
    To reduce false positive due to multiple comparisons, bonferroni correction
    is applied to the p values.
    Note that the frequency dimension is reduced to one for the test
    (average in the frequency band-of-interest). To take frequencies
    into account, use cluster statistics.
    Parameters
    -----
    PSDs_task_normLog: array of subjects PSD Logratio (ndarray) for a condition
    (n_samples, n_tests with n_tests the different channels).
    Epochs_task: Epochs object for the condition 'task' for example, from a
    random subject, only used to get parameters from the info
    (sampling frequencies for example).
    n_permutations: the number of permutations, int. Should be at least
    2*n_samples, can be set to 50000 for example.
    alpha_bonferroni: the threshold for bonferroni correction, int. Can be set
    to 0.05.
    alpha: the threhold for ttest, int. Can be set to 0.05.
    Plot
    -----
    topomap with T_statistic for the significant sensors.
    Returns
    -----
    T_obs: T-statistic observed for all variables, array of shape (n_tests).
    p_values: p-values for all the tests, array of shape (n_tests).
    H0: T-statistic obtained by permutations and t-max trick for multiple
    comparison, array of shape (n_permutations).
    adj_p: adjusted p values from bonferroni correction, tuple of 2 arrays
    of shape (n_tests, n_tests), with boolean assessment for p values
    and p values corrected.
    """

    # averaging across frequencies (compute stats only in ch space)
    p = np.mean(PSDs_task_normLog, axis=2)
    T_obs, p_values, H0 = mne.stats.permutation_t_test(
        p, n_permutations, tail=0, n_jobs=1)
    adj_p = mne.stats.bonferroni_correction(p_values, alpha=alpha_bonferroni)

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c in adj_p[1]:  # adj_p tuple 2 arrays
        if c <= alpha:
            i = np.where(adj_p[1] == c)
    T_obs_plot[i] = T_obs[i]
    T_obs_plot = np.nan_to_num(T_obs_plot)

    # getting sensors position
    pos = np.array([[0, 0]])
    for i in range(0, len(Epochs_task[0].info['ch_names'])):
        cor = np.array([Epochs_task[0].info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis=0)
    pos = pos[1:]

    # topoplot of significant sensors
    if np.max(np.abs(T_obs_plot)) != 0:
        vmax = np.max(np.abs(T_obs_plot))
        vmin = -vmax
    else:
        vmax = None
        vmin = None
    mne.viz.plot_topomap(T_obs_plot, pos, vmin=vmin, vmax=vmax, sensors=True)

    return T_obs, p_values, H0, adj_p
