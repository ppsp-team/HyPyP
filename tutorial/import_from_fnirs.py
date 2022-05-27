#!/usr/bin/env python
# coding: utf-8

# # Import fNIRS data into HyPyP
#
# Authors         : Ghazaleh Ranjbaran, Caitriona Douglas, Guillaume Dumas
#
# Date            : 2022-05-21

import numpy as np
import mne
import os

from hypyp.fnirs_tools import load_fnirs
from hypyp.fnirs_tools import make_fnirs_montage
from hypyp.fnirs_tools import fnirs_epoch
from hypyp.fnirs_tools import fnirs_montage_ui

path_1 = "../data/FNIRS/DCARE_02_sub1.snirf"

path_2 = "../data/FNIRS/DCARE_02_sub2.snirf"

fnirs_participant_1 = load_fnirs(path_1, path_2, attr=None, preload=False, verbose=None)[0]
fnirs_participant_2 = load_fnirs(path_1, path_2, attr=None, preload=False, verbose=None)[1]

# source_labels, detector_labels, Nz, RPA, LPA, head_size = fnirs_montage_ui()

# prob_mat_file = '../data/FNIRS/MCARE_01_probeInfo.mat'

# location = make_fnirs_montage(source_labels, detector_labels, prob_mat_file,
#                               Nz, RPA, LPA, head_size)

#Sources ' labels: S#
source_labels = ['S1','S2','S3','S4','S5','S6','S7','S8']
#Sources ' labels: D#
detector_labels = ['D1','D2','D3','D4','D5','D6','D7','D8']
#directory of the probeInfo.mat file
prob_mat_file = '../data/FNIRS/MCARE_01_probeInfo.mat'
#3D Coordination of the tip of the nose: [x, y, z] in mm
Nz_coord = [12.62, 17.33, 16.74]
#3D Coordination of the right preauricular: [x, y, z] in mm
RPA = [21.0121020904262, 15.9632489747085, 17.2796094659563]
#3D Coordination of the left preauricular: [x, y, z] in mm
LPA = [4.55522116441745, 14.6744377188919, 18.3544292678269]
#Head size in mm
head_size = 0.16

location = make_fnirs_montage(source_labels, detector_labels, prob_mat_file,
                              Nz_coord, RPA, LPA, head_size)

fnirs_epo1 = fnirs_epoch(fnirs_participant_1, fnirs_participant_2 ,tmin = -0.1, tmax = 1,
                    baseline = (None, 0), preload = True, event_repeated = 'merge')[0]
                    
fnirs_epo2 = fnirs_epoch(fnirs_participant_1, fnirs_participant_2, tmin = -0.1, tmax = 1,
                    baseline = (None, 0), preload = True, event_repeated = 'merge')[1]

# fnirs_epo1 and fnirs_epo2 are now compatible with tools provided by HyPyP (for visualization and statictical analyses check [this](https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb) tutorial)

fnirs_epo1.set_montage(location)
fnirs_epo2.set_montage(location)

fnirs_epo1.plot_sensors()

fnirs_epo2.plot_sensors()

fnirs_epo1.plot()

fnirs_epo2.plot()


