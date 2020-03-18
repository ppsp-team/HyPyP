# coding=utf-8

import mne 
import numpy as np

"""
Concatenate a list of epochs in one epoch.

Parameters
-----
epoch_S1, epoch_S2 : list of epochs for each subject (for example the list
samples the different occurences of the baseline condition across experiments). 
Epochs are MNE objects (data are stored in an array of shape 
(n_epochs, n_channels, n_times) and info is a disctionnary sampling parameters).

Returns
-----
List of concatenate epochs (for example one epoch with all the occurences of the
baseline condition across experiments) for each subject.

"""

def concatenate_epochs(epoch_S1, epoch_S2):

    epoch_S1_concat   = mne.concatenate_epochs(epoch_S1)
    epoch_S2_concat   = mne.concatenate_epochs(epoch_S2)

    return epoch_S1_concat, epoch_S2_concat
  
