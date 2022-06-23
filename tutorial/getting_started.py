#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : getting_started.py
# description     : Demonstration of HyPyP basics.
# author          : Guillaume Dumas, Anaël Ayrolles, Florence Brun
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================
import io
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mne
import requests

from hypyp import (
    prep,
)  # need pip install https://api.github.com/repos/autoreject/autoreject/zipball/master
from hypyp import analyses
from hypyp import stats
from hypyp import viz

plt.ion()

# Setting parameters

# Frequency bands used in the study
freq_bands = {
    "Theta": [4, 7],
    "Alpha-Low": [7.5, 11],
    "Alpha-High": [11.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 48],
}
#Full frequency band for PDC connectivity measure 
full_freq = { 'full_frq': [1, 48]}
freq_bands = OrderedDict(freq_bands)  # Force to keep order

# Specify sampling frequency
sampling_rate = 500  # Hz

# Loading datasets (see MNE functions mne.io.read_raw_format),
# convert them to MNE Epochs

# In our example, we load Epochs directly from EEG dataset in
# the fiff format
URL_TEMPLATE = (
    "https://github.com/ppsp-team/HyPyP/blob/master/data/participant{}-epo.fif?raw=true"
)


def get_data(idx):
    return io.BytesIO(requests.get(URL_TEMPLATE.format(idx)).content)


epo1 = mne.read_epochs(get_data(1), preload=True, )

epo2 = mne.read_epochs(get_data(2), preload=True, )

# In our example, since the dataset was not initially
# dedicate to hyperscanning, we need to equalize
# the number of epochs between our two participants
mne.epochs.equalize_epoch_counts([epo1, epo2])

# Preprocessing epochs

# Warning: here we directly load epochs data,
# for raw data we highly recommend to perform high-pass filtering
# with prep.filt function before converting raw to epochs

# Computing global AutoReject and Independent Components Analysis
# for each participant
icas = prep.ICA_fit(
    [epo1, epo2],
    n_components=15,
    method="infomax",
    fit_params=dict(extended=True),
    random_state=42,
)

# Selecting relevant Independent Components for artefact rejection
# on one participant, that will be transpose to the other participant
# and fitting the ICA
cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])
plt.close("all")

# For using PDC measure (Partial Coherence Connectivity)
# We should use the data that no ICA algorithm is applied on them.

cleaned_epochs_no_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])
plt.close("all")

# Applying local AutoReject for each participant
# rejecting bad epochs, rejecting or interpolating partially bad channels
# removing the same bad channels and epochs across participants
# plotting signal before and after (verbose=True)

# Applying on data obtained with ICA algorithm
cleaned_epochs_AR, dic_AR = prep.AR_local(
    cleaned_epochs_ICA, strategy="union", threshold=50.0, verbose=True
)
input("Press ENTER to continue")
plt.close("all")

# Applying on data obtained without ICA algorithm
no_ICA_cleaned_epochs_AR, no_ICA_dic_AR = prep.AR_local(cleaned_epochs_no_ICA,
                                                        strategy="union",
                                                        threshold=50.0,
                                                        verbose=True
                                                        )
input("Press ENTER to continue")
plt.close("all")

# Picking the preprocessed epochs for each participant
preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]
no_ICA_preproc_S1 = no_ICA_cleaned_epochs_AR[0]
no_ICA_preproc_S2 = no_ICA_cleaned_epochs_AR[1]

# Analysing data

# Welch Power Spectral Density
# Here for ex, the frequency-band-of-interest is restricted to Alpha_Low,
# frequencies for which power spectral density is actually computed
# are returned in freq_list,
# and PSD values are averaged across epochs

psd1 = analyses.pow(
    preproc_S1, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True
)
psd2 = analyses.pow(
    preproc_S2, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True
)
data_psd = np.array([psd1.psd, psd2.psd])

no_ICA_psd1 = analyses.pow(no_ICA_preproc_S1, fmin=7.5, fmax=11,
                           n_fft=1000, n_per_seg=1000, epochs_average=True)
no_ICA_psd2 = analyses.pow(no_ICA_preproc_S2, fmin=7.5, fmax=11,
                           n_fft=1000, n_per_seg=1000, epochs_average=True)
no_ICA_data_psd = np.array([no_ICA_psd1.psd, no_ICA_psd2.psd])

# Connectivity

# initializing data and storage

data_inter = np.array([preproc_S1, preproc_S2])
result_intra = []

no_ICA_data_inter = np.array([no_ICA_preproc_S1, no_ICA_preproc_S2])

# computing analytic signal per frequency band

complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

no_ICA_complex_signal = analyses.compute_freq_bands(no_ICA_data_inter, sampling_rate,
                                             full_freq)

# computing frequency- and time-frequency-domain connectivity,
# 'ccorr' for example
result = analyses.compute_sync(complex_signal, mode="ccorr")

# slicing results to get the Inter-brain part of the matrix
n_ch = len(epo1.info["ch_names"])
theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch: 2 * n_ch]

# choosing Alpha_Low for further analyses for example
values = alpha_low

# computing Cohens'D for further analyses for example
C = (values - np.mean(values[:])) / np.std(values[:])

# slicing results to get the Intra-brain part of the matrix
for i in [0, 1]:
    theta, alpha_low, alpha_high, beta, gamma = result[:, (i * n_ch):((i + 1) * n_ch), (i * n_ch): ((i + 1) * n_ch)]
    # choosing Alpha_Low for further analyses for example
    values_intra = alpha_low
    values_intra -= np.diag(np.diag(values_intra))
    # computing Cohens'D for further analyses for example
    C_intra = (values_intra - np.mean(values_intra[:])) / np.std(values_intra[:])
    # can also sample CSD values directly for statistical analyses
    result_intra.append(C_intra)

# Computing frequency- and time-frequency-domain connectivity measures
# obtained by MVARICA approach, based on MVAR models' coefficients.
# For instance: PDC measure, with MVAR model of order 2, extended infomax ICA method with checking the stability.
PDC = analyses.compute_conn_mvar(no_ICA_complex_signal, 
                                         mvar_params={"mvar_order": 2, "fitting_method":"default", "delta": 0},
                                         ica_params={"method": "infomax_extended", "random_state": None},
                                         measure_params={"name": "pdc", "n_fft": 512}
                                         )

# Slicing results to get the Inter-brain of the connectivity matrix.
# assigning the maximum value in the frequency spectrum
# mvar-based connectivity measures are calculated over a frequency range assigned by n_fft variable.
# here n_fft = 512
no_ICA_result_inter = []
for i in [0, 1]:
    PDC_q = PDC.squeeze()
    if i == 0 :
      mvar_spectrum =  PDC_q[n_ch:n_ch*2, 0:n_ch, :]
    else:
      mvar_spectrum =  PDC_q[0:n_ch, n_ch:n_ch*2, :]   
    PDC_fmax = np.zeros((n_ch, n_ch), dtype=PDC.dtype)
    for j in range(0, n_ch):
      for k in range(0, n_ch):
        PDC_fmax[j, k] = np.amax(mvar_spectrum[j,k,:])
    mvar_values_inter = PDC_fmax
    mvar_values_inter -= np.diag(np.diag(mvar_values_inter))
    # computing Cohens'D for further analyses for example
    mvar_C_inter = (mvar_values_inter -
               np.mean(mvar_values_inter[:])) / np.std(mvar_values_inter[:])
    # can also sample CSD values directly for statistical analyses
    no_ICA_result_inter.append(mvar_C_inter)

# Slicing results to get the Intra-brain of the connectivity matrix
# assigning the maximum value in the frequency spectrum
# mvar-based connectivity measures are calculated over a frequency range assigned by n_fft variable
# here n_fft = 512
no_ICA_result_intra = []
for i in [0, 1]:
    PDC = PDC.squeeze()
    mvar_spectrum =  PDC[i*n_ch:n_ch*(i+1), i*n_ch:n_ch*(i+1), :]
    # choosing the max values for futher analyses for example
    PDC_fmax = np.zeros((n_ch, n_ch), dtype=PDC.dtype)
    for j in range(0, n_ch):
      for k in range(0, n_ch):
        PDC_fmax[j, k] = np.amax(mvar_spectrum[j, k, :])
    mvar_values_intra = PDC_fmax
    mvar_values_intra -= np.diag(np.diag(mvar_values_intra))
    # computing Cohens'D for further analyses for example
    mvar_C_intra = (mvar_values_intra -
               np.mean(mvar_values_intra[:])) / np.std(mvar_values_intra[:])
    # can also sample CSD values directly for statistical analyses
    no_ICA_result_intra.append(mvar_C_intra)

# Statistical analyses

# Comparing PSD values to random signal
# Parametric t test
# 1/ MNE test without any correction
# This function takes samples (observations) by number
# of tests (variables i.e. channels),
# thus PSD values are averaged in the frequency dimension
psd1_mean = np.mean(psd1.psd, axis=1)
psd2_mean = np.mean(psd2.psd, axis=1)
X = np.array([psd1_mean, psd2_mean])
T_obs, p_values, H0 = mne.stats.permutation_t_test(
    X=X, n_permutations=5000, tail=0, n_jobs=1
)

# 2/ HyPyP parametric t test with FDR correction
# based on MNE function, the same things as above are true.
# FDR correction for multiple comparisons is added.
statsCondTuple = stats.statsCond(
    data=data_psd,
    epochs=preproc_S1,
    n_permutations=5000,
    alpha=0.05,
)

# 3/ Non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels
# across space and frequencies based on their position,
# in the Alpha_Low band for example
con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=psd1.freq_list)
ch_con_freq = con_matrixTuple.ch_con_freq
# constitute two artificial groups with 2 'participant1' and 2 'participant1'
data_group = [np.array([psd1.psd, psd1.psd]), np.array([psd2.psd, psd2.psd])]
statscondCluster = stats.statscondCluster(
    data=data_group,
    freqs_mean=psd1.freq_list,
    ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
    tail=0,
    n_permutations=5000,
    alpha=0.05,
)

# Comparing Intra-brain connectivity values between participants

# With 3/ non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels
# across space and frequencies based on their position

con_matrixTuple = stats.con_matrix(
    epochs=preproc_S1, freqs_mean=np.arange(7.5, 11), draw=False
)

# Note that for connectivity, values are computed for
# every integer in the frequency bin from fmin to fmax,
# freqs_mean=np.arange(fmin, fmax)
# whereas in PSD it depends on the n_fft parameter
# psd.freq_list
# But for CSD, values are averaged across each frequencies
# so you do not need to take frequency into account
# to correct clusters
ch_con = con_matrixTuple.ch_con

# constitute two artificial groups with 2 'participant1' and 2 'participant2'
# in Alpha_Low band for example (see above)
Alpha_Low = [
    np.array([result_intra[0], result_intra[0]]),
    np.array([result_intra[1], result_intra[1]]),
]

statscondCluster_intra = stats.statscondCluster(
    data=Alpha_Low,
    freqs_mean=np.arange(7.5, 11),
    ch_con_freq=scipy.sparse.bsr_matrix(ch_con),
    tail=0,
    n_permutations=5000,
    alpha=0.05,
)

# Comparing Inter-brain connectivity values to random signal

# No a priori connectivity between channels is considered
# between the two participants
# in Alpha_Low band for example (see above)
# constitute two artificial groups with 2 'participant1' and 2 'participant2'
data = [np.array([values, values]), np.array([result_intra[0], result_intra[0]])]

statscondCluster = stats.statscondCluster(
    data=data,
    freqs_mean=np.arange(7.5, 11),
    ch_con_freq=None,
    tail=0,
    n_permutations=5000,
    alpha=0.05,
)

# Visualization

# Visualization of T values for sensors
# (T_obs_plot = T_obs for 1/ or
# statsCondTuple.T_obs for 2/ or
# statscondCluster.F_obs_plot for 3/)
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs, epochs=preproc_S1)

# Visualize T values for significant sensors only
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs_plot, epochs=preproc_S1)

# Visualization of inter-brain links projected
# on either 2D or 3D head models

# can be applied to Cohen’s D (C as done here) or
# statistical values (statscondCluster.F_obs or F_obs_plot)
# of inter-individual brain connectivity

# defining manually bad channel for viz test
epo1.info["bads"] = ["F8", "Fp2", "Cz", "O2"]
epo2.info["bads"] = ["F7", "O1"]

# Warning, threshold='auto' must be used carefully,
# it is calculated specifically for the dyad,
# and therefore does not allow comparability between different dyads.

# Inter-brain Hilbert-based connectivity
viz.viz_2D_topomap_inter(epo1, epo2, C, threshold='auto', steps=10, lab=True)

# Inter-brain VAR-based connectivity
# Information flow from participant 1 to participant 2
viz.viz_2D_topomap_inter(epo1, epo2, no_ICA_result_inter[0], threshold='auto', steps=10, lab=True)

# Inter-brain VAR-based connectivity
# Information flow from participant 2 to participant 1
viz.viz_2D_topomap_inter(epo1, epo2, no_ICA_result_inter[1], threshold='auto', steps=10, lab=True)

# Visualization of inter-brain connectivity in 3D

# Inter-brain Hilbert-based connectivity
viz.viz_3D_inter(epo1, epo2, C, threshold='auto', steps=10, lab=False)

# Inter-brain VAR-based connectivity
# Information flow from participant 1 to participant 2
viz.viz_3D_inter(epo1, epo2, no_ICA_result_inter[0], threshold='auto', steps=10, lab=False)

# Inter-brain VAR-based connectivity
# Information flow from participant 2 to participant 1
viz.viz_3D_inter(epo1, epo2, no_ICA_result_inter[1], threshold='auto', steps=10, lab=False)

# Visualization of intra-brain connectivity in 2D

# Intra-brain Hilbert-based connectivity
viz.viz_2D_topomap_intra(epo1, epo2,
                         C1=result_intra[0],
                         C2=result_intra[1],
                         threshold='auto',
                         steps=2,
                         lab=False)

# Intra-brain VAR-based connectivity
viz.viz_2D_topomap_intra(epo1, epo2,
                         C1=no_ICA_result_intra[0],
                         C2=no_ICA_result_intra[1],
                         threshold='auto',
                         steps=2,
                         lab=False)

# Visualization of intra-brain connectivity in 3D

# Intra-brain Hilbert-based connectivity
viz.viz_3D_intra(epo1, epo2,
                 C1=result_intra[0],
                 C2=result_intra[1],
                 threshold='auto',
                 steps=10,
                 lab=False)

# Intra-brain VAR-based connectivity
viz.viz_3D_intra(epo1, epo2,
                 C1=no_ICA_result_intra[0],
                 C2=no_ICA_result_intra[0],
                 threshold='auto',
                 steps=10,
                 lab=False)
