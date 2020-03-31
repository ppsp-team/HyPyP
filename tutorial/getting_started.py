#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : getting_started.py
# description     : Demonstration of PyPyP basics.
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
import mne
from hypyp.viz import transform
from hypyp.viz import plot_sensors_2d, plot_links_2d
from hypyp.viz import plot_sensors_3d, plot_links_3d
from hypyp.prep import ICA_fit, ICA_choice_comp, AR_local
from hypyp.analyses import compute_freq_bands, compute_sync

plt.ion()

# Frequency bands used in the study
freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}
freq_bands = OrderedDict(freq_bands)  # Force to keep order

# Loading data files & extracting sensor infos
epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"), preload=True)
loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
lab1 = [ch + "_1" for ch in epo1.ch_names]

epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"), preload=True)
loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
lab2 = [ch + "_2" for ch in epo2.ch_names]
loc2 = transform(loc2)

# Equalize epochs size
mne.epochs.equalize_epoch_counts([epo1, epo2])

# concatenate epochs
epochs = [epo1, epo2]

# Preproc
# computing global AR and ICA on epochs,
icas = ICA_fit(epochs,
               n_components=15,
               method='fastica',
               random_state=42)

# selecting components semi auto and fitting them
cleaned_epochs_ICA = ICA_choice_comp(icas, epochs)  # no ICA_component selected
plt.close('all')

# applying local AR on subj epochs and rejecting epochs if bad for S1 or S2
cleaned_epochs_AR = AR_local(cleaned_epochs_ICA, plot=True)
input("Press ENTER to continue")
plt.close('all')

preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]

# Connectivity
# Create array
data = np.array([preproc_S1, preproc_S2])

# Compute analytic signal per frequency band
complex_signal = compute_freq_bands(data, freq_bands)

# Compute frequency- and time-frequency-domain connectivity measures.
result = compute_sync(complex_signal,
                      mode='plv',
                      epoch_wise=True,
                      time_resolved=True)

theta, alpha_low, alpha_high, beta, gamma = result

C = (alpha_low - np.mean(alpha_low[:])) / np.std(alpha_low[:])

# Visualization of inter-brain connectivity in 2D
plt.figure(figsize=(10, 20))
plt.gca().set_aspect('equal', 'box')
plt.axis('off')
plot_sensors_2d(loc1, loc2, lab1, lab2)
plot_links_2d(loc1, loc2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()

# Visualization of inter-brain connectivity in 3D
loc2 = transform(loc2, traY=0.15, rotZ=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('off')
plot_sensors_3d(ax, loc1, loc2, lab1, lab2)
plot_links_3d(ax, loc1, loc2, C=C, threshold=2, steps=10)
plt.tight_layout()
plt.show()
