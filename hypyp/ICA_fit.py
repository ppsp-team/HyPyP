# coding=utf-8

import mne
from mne.preprocessing import ICA, corrmap
from autoreject import get_rejection_threshold, compute_thresholds, AutoReject, set_matplotlib_defaults 
import numpy as np 
import matplotlib.pyplot as plt

"""
Compute global Autorejection to fit Independant Components Analysis 
on Epochs, for each subject.

Pre requisite : install autoreject 
https://api.github.com/repos/autoreject/autoreject/zipball/master

Parameters
-----
epochs : list of 2 Epochs objects (for each subject). Epochs_S1 and Epochs_S2 
correspond to a condition and can result from the concatenation of epochs from 
different occurences of the condition across experiments.
Epochs are MNE objects (data are stored in an array of shape 
(n_epochs, n_channels, n_times) and info is a disctionnary sampling parameters).

n_components : the number of principal components that are passed to the ICA algorithm 
during fitting, for a first estimation, n_components can be set to 15.

method : the ICA method used, 'fastica', 'infomax' or 'picard'. 'Fastica' is the most 
frequently used.

random_state : the parameter used to compute random distributions for ICA calulation, 
int or None. It can be useful to fix random_state value to have reproducible results. 
For 15 components, random_state can be set to 97 for example.

Info
-----
If Autoreject and ICA take too much time, change the decim value (see MNE documentation).

Returns
-----
List of independant components for each subject. IC are MNE objects, see MNE documentation
for more details.

"""


def ICA_comp(epochs, n_components, method, random_state):

    icas = []
    for epoch in epochs: # per subj
             
        ## applying AR to find global rejection threshold 
        reject = get_rejection_threshold(epoch,ch_types='eeg') 
        print('The rejection dictionary is %s' % reject)

        ## fitting ICA on filt_raw after AR
        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        icas.append(ica.fit(epoch,reject=reject, tstep=1))
    
    return icas