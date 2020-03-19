#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : getting_started.py
# description     : Demonstration of PyPyP basics.
# author          : Guillaume Dumas
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

import os
import mne
#from hypyp.viz import transform, plot_sensors_2d, plot_links_2d
from copy import copy
import numpy as np
import matplotlib.pyplot as plt


os.chdir("D:/Amir/Dropbox/Studies BIU/Ruth Feldman/My Thesis/My Analysis/EEG/EEG/shared/HyPyP")

# Loading data files & extracting sensor infos
epo1 = mne.read_epochs(os.path.join("data", "subject1.fif"), preload=True)
loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
lab1 = [ch + "_1" for ch in epo1.ch_names]

epo2 = mne.read_epochs(os.path.join("data", "subject2.fif"), preload=True)
loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
lab2 = [ch + "_2" for ch in epo2.ch_names]
loc2 = transform(loc2)

# Connectivity measure and threshold
C = np.random.rand(len(loc1), len(loc2))
thresh = 0.99

# Visualization
plt.figure(figsize=(10, 20))
plt.gca().set_aspect('equal', 'box')
plt.axis('off')
plot_sensors_2d(loc1, loc2, lab1, lab2)
plot_links_2d(loc1, loc2, C=np.random.rand(len(loc1), len(loc2)), threshold=0.9, steps=10)
plt.tight_layout()
plt.show()
