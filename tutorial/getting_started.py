#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : getting_started.py
# description     : Demonstration of HyPyP basics.
# author          : Guillaume Dumas, Anaël Ayrolles
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

# Frequency bands used in the study
freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}
freq_bands = OrderedDict(freq_bands)  # Force to keep order

# Loading data files & extracting sensor infos
epo1 = mne.read_epochs(os.path.join(os.path.dirname(__file__),
       os.pardir,'data',"subject1-epo.fif"), preload=True)
loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
loc1 = viz.transform(loc1, traX=-0.155, traY=0, traZ=+0.01, rotZ=(-np.pi/2))
loc1 = viz.adjust_loc(loc1, traZ=+0.01)
lab1 = [ch for ch in epo1.ch_names]

epo2 = mne.read_epochs(os.path.join(os.path.dirname(__file__),
       os.pardir,'data',"subject2-epo.fif"), preload=True)
loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
loc2 = viz.transform(loc2, traX=+0.155, traY=0, traZ=+0.01, rotZ=np.pi/2)
loc2 = viz.adjust_loc(loc2, traZ=+0.01)
lab2 = [ch for ch in epo2.ch_names]

n_ch = len(epo1.ch_names)

# Equalize epochs size
mne.epochs.equalize_epoch_counts([epo1, epo2])

# concatenate epochs
epochs = [epo1, epo2]

# Preproc
# computing global AR and ICA on epochs,
icas = prep.ICA_fit(epochs,
               n_components=15,
               method='fastica',
               random_state=42)

# selecting components semi auto and fitting them
cleaned_epochs_ICA = prep.ICA_choice_comp(icas, epochs)  # no ICA_component selected
plt.close('all')

# applying local AR on subj epochs and rejecting epochs if bad for S1 or S2
cleaned_epochs_AR = prep.AR_local(cleaned_epochs_ICA, verbose=True)
input("Press ENTER to continue")
plt.close('all')

preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]

# Power spectral density
# Computing power spectral density with welch method, between 4 and 7 Hz for example
psd1 = analyses.pow(preproc_S1, fmin=4, fmax=7, n_fft=1000, n_per_seg=1000, epochs_average=True)
psd2 = analyses.pow(preproc_S2, fmin=4, fmax=7, n_fft=1000, n_per_seg=1000, epochs_average=True)
data_psd = np.array([psd1.psd, psd2.psd])

# Comparing power spectral of the epochs to random signal
# 1/ simple parametric t test
# averaging on frequency band of interest
psd1_mean = np.mean(psd1.psd, axis=1)
psd2_mean = np.mean(psd2.psd, axis=1)
X = np.array([psd1_mean, psd2_mean])
T_obs, p_values, H0 = mne.stats.permutation_t_test(X=X, n_permutations=5000,
                                                   tail=0, n_jobs=1)

# 2/ parametric t test with bonferrroni correction
statsCondTuple = stats.statsCond(data=data_psd, epochs=preproc_S1, n_permutations=5000,
                                 alpha_bonferroni=0.05, alpha=0.05)

# 3/ non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels across space and frequencies
# based on their position, in theta band for example
con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=psd1.freq_list)
ch_con_freq = con_matrixTuple.ch_con_freq
# consitute two artificial groups
data_group = [np.array([psd1.psd, psd1.psd]), np.array([psd2.psd, psd2.psd])]
statscondCluster = stats.statscondCluster(data=data_group,
                                          freqs_mean=psd1.freq_list,
                                          ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)

# Vizualise T values for sensors
# (T_obs_plot = T_obs for 1/ or statsCondTuple.T_obs for 2/ or statscondCluster.F_obs_plot for 3/)
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs, epochs=preproc_S1)
# Vizualise T values for significant sensors only
viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs_plot, epochs=preproc_S1)


# Connectivity
# Intra-brain
# Create array
data_intra1 = np.array([preproc_S1, preproc_S1])
data_intra2 = np.array([preproc_S2, preproc_S2])
result_intra = []
# for each subject
for data in [data_intra1, data_intra2]:
  # Compute analytic signal per frequency band
  complex_signal = analyses.compute_freq_bands(data, freq_bands)

  # Compute frequency- and time-frequency-domain connectivity measures.
  result_intra.append(analyses.compute_sync(complex_signal,
                      mode='ccorr'))

# Compare connectivity values to random signal
# with 3/ non-parametric cluster-based permutations
# creating matrix of a priori connectivity between channels
# across space and frequencies based on their position
# in theta band for example (position 0)
theta = np.array([result_intra[0][0], result_intra[1][0]])
con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=[4, 7])
ch_con_freq = con_matrixTuple.ch_con_freq
statscondCluster = stats.statscondCluster(data=theta,
                                          freqs_mean=[4, 7],
                                          ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)
# can vizualise or do other tests as decribed above

# Connectivity
# Inter brain
data_inter = np.array([preproc_S1, preproc_S2])

# Compute analytic signal per frequency band
complex_signal = analyses.compute_freq_bands(data_inter, freq_bands)

# Compute frequency- and time-frequency-domain connectivity measures.
result = analyses.compute_sync(complex_signal, mode='ccorr')

# slicing to get the inter-brain part of the matrix
theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch:2*n_ch]

# with more than two subjects (one pair), can do statitistics

values = alpha_low
values -= np.diag(np.diag(values))

C = (values - np.mean(values[:])) / np.std(values[:])

# Defined bad channel for viz test
epo1.info['bads'] = ['F8', 'Fp2', 'Cz', 'O2']
epo2.info['bads'] = ['F7', 'O1']

# Visualization of inter-brain connectivity in 2D
fig, ax = plt.subplots(1,1)
ax.axis("off")
vertices, faces = viz.get_3d_heads()
camera = Camera("ortho", theta=90, phi=180, scale=1)
mesh = Mesh(ax, camera.transform @ glm.yrotate(90), vertices, faces,
            facecolors='white',  edgecolors='black', linewidths=.25)
camera.connect(ax, mesh.update)

plt.gca().set_aspect('equal', 'box')
plt.axis('off')
viz.plot_sensors_2d(epo1, epo2, loc1, loc2, lab1, lab2)
viz.plot_links_2d(loc1, loc2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()


# Visualization of inter-brain connectivity in 3D with get_3D_heads

vertices, faces = viz.get_3d_heads()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis("off")
viz.plot_3d_heads(ax, vertices, faces)
viz.plot_sensors_3d(ax, epo1, epo2, loc1, loc2)
viz.plot_links_3d(ax, loc1, loc2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()
