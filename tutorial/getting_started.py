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
import os
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import mne

import mpl3d
from mpl3d import glm
from mpl3d.mesh import Mesh
from mpl3d.camera import Camera

from hypyp import prep
from hypyp import analyses
from hypyp import stats
from hypyp import viz

plt.ion()

## Setting parameters

# Frequency bands used in the study
freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}
freq_bands = OrderedDict(freq_bands)  # Force to keep order


## Loading datasets (see MNE functions mne.io.read_raw_format), convert them to MNE Epochs

# In our example, we load Epochs directly from EEG dataset in the fiff format
epo1 = mne.read_epochs(os.path.join(os.path.dirname(__file__),
       os.pardir,'data',"subject1-epo.fif"), preload=True)

epo2 = mne.read_epochs(os.path.join(os.path.dirname(__file__),
       os.pardir,'data',"subject2-epo.fif"), preload=True)

# In our example, since the dataset was not initially dedicate to hyperscanning,
# we need to equalize the number of epochs between our two subjects 
mne.epochs.equalize_epoch_counts([epo1, epo2])


## Preprocessing epochs

# Computing global AutoReject and Independant Components Analysis for each subject
icas = prep.ICA_fit([epo1, epo2],
               n_components=15,
               method='fastica',
               random_state=42)

# Selecting relevant Independant Components for artefact rejection
# on one subject, that will be transpose to the other subject
# and fitting the ICA
cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])
plt.close('all')

# Applying local AutoReject for each subject
# rejecting bad epochs, rejecting or interpolating partially bad channels
# removing the same bad channels and epochs across subjects
# plotting signal before and after (verbose=True)
cleaned_epochs_AR = prep.AR_local(cleaned_epochs_ICA, verbose=True)
input("Press ENTER to continue")
plt.close('all')

# Picking the preprocessed epochs for each subject
preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]


## Analysing data

# Welch Power Spectral Density
# Here, the frequency-band-of-interest is restricted to Alpha_Low for example,
# frequencies for which power spectral density is actually computed are returned in freq_list,
# and PSD values are averaged across epochs
psd1 = analyses.pow(preproc_S1, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True)
psd2 = analyses.pow(preproc_S2, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True)
data_psd = np.array([psd1.psd, psd2.psd])

# Connectivity

# initializing data and storage
data_inter = np.array([preproc_S1, preproc_S2])
result_intra = []
# computing analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, freq_bands)
# computing frequency- and time-frequency-domain connectivity, 'ccorr' for example
result = analyses.compute_sync(complex_signal, mode='ccorr')

# slicing results to get the Inter-brain part of the matrix
theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch:2*n_ch]
# choosing Alpha_Low for futher analyses for example
values = alpha_low
values -= np.diag(np.diag(values))
# computing Cohens'D for further analyses for example
C = (values - np.mean(values[:])) / np.std(values[:])

# slicing results to get the Intra-brain part of the matrix
for i in [0, 1]:
  theta, alpha_low, alpha_high, beta, gamma = result[:, i:i+n_ch, i:i+n_ch]
  # choosing Alpha_Low for futher analyses for example  
  values = alpha_low
  values -= np.diag(np.diag(values))
  # computing Cohens'D for further analyses for example
  C_intra = (values - np.mean(values[:])) / np.std(values[:])
  result_intra.append(C_intra)


## Statistical anlyses

# Comparing PSD values to random signal
# Parametric t test
# 1/ MNE test without any correction
# This function takes samples (observations) by number of tests (variables i.e. channels),
# thus PSD values are averaged in the frequency dimension
psd1_mean = np.mean(psd1.psd, axis=1)
psd2_mean = np.mean(psd2.psd, axis=1)
X = np.array([psd1_mean, psd2_mean])
T_obs, p_values, H0 = mne.stats.permutation_t_test(X=X, n_permutations=5000,
                                                   tail=0, n_jobs=1)

# 2/ HyPyP parametric t test with bonferrroni correction
# based on MNE function, the same things as above are true.
# Bonferroni correction for multiple comparisons is added.
statsCondTuple = stats.statsCond(data=data_psd, epochs=preproc_S1, n_permutations=5000,
                                 alpha_bonferroni=0.05, alpha=0.05)

# 3/ Non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels across space and frequencies
# based on their position, in the Alpha_Low band for example
con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=psd1.freq_list)
ch_con_freq = con_matrixTuple.ch_con_freq
# consitute two artificial groups with 2 'subject1' and 2 'subject1'
data_group = [np.array([psd1.psd, psd1.psd]), np.array([psd2.psd, psd2.psd])]
statscondCluster = stats.statscondCluster(data=data_group,
                                          freqs_mean=psd1.freq_list,
                                          ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)


# Comparing Intra-brain connectivity values between subjects

# With 3/ non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels
# across space and frequencies based on their position
# in Alpha_Low band for example (position 1 in freq_bands)
Alpha_Low = np.array([result_intra[0][1], result_intra[1][1]])
con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=[7.5, 11])
ch_con_freq = con_matrixTuple.ch_con_freq
statscondCluster = stats.statscondCluster(data=Alpha_Low,
                                          freqs_mean=[7.5, 11],
                                          ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)

# Comparing Inter-brain connectivity values to random signal
# TODO


## Vizualisation

# Vizualisation of T values for sensors
# (T_obs_plot = T_obs for 1/ or statsCondTuple.T_obs for 2/ or statscondCluster.F_obs_plot for 3/)
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs, epochs=preproc_S1)

# Vizualise T values for significant sensors only
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs_plot, epochs=preproc_S1)

# Vizulisation of inter-brain links projected on either 2D or 3D head models
# can be applied to Cohen’s D or statistical values of inter-individual brain connectivity

# defining manually bad channel for viz test
epo1.info['bads'] = ['F8', 'Fp2', 'Cz', 'O2']
epo2.info['bads'] = ['F7', 'O1']

# Visualization of inter-brain connectivity in 2D
# defining head model and adding sensors
fig, ax = plt.subplots(1,1)
ax.axis("off")
vertices, faces = viz.get_3d_heads()
camera = Camera("ortho", theta=90, phi=180, scale=1)
mesh = Mesh(ax, camera.transform @ glm.yrotate(90), vertices, faces,
            facecolors='white',  edgecolors='black', linewidths=.25)
camera.connect(ax, mesh.update)
plt.gca().set_aspect('equal', 'box')
plt.axis('off')
viz.plot_sensors_2d(epo1, epo2, lab=True) # bads are represented as squares
# plotting links according to sign (red for positive values, blue for negative) and
# value (line thickness increases with the strength of connectivity) 
viz.plot_links_2d(epo1, epo2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()

# Visualization of inter-brain connectivity in 3D
# defining head model and adding sensors
vertices, faces = viz.get_3d_heads()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis("off")
viz.plot_3d_heads(ax, vertices, faces)
viz.plot_sensors_3d(ax, epo1, epo2, lab=False) # bads are represented as squares
# plotting links according to sign (red for positive values, blue for negative) and
# value (line thickness increases with the strength of connectivity)
viz.plot_links_3d(ax, epo1, epo2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()
