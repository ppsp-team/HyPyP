# coding=utf-8

import mne 
import numpy as np
from mne.io.constants import FIFF
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt

"""
Compute epochs from raws and vizualize PSD on average epochs.

Parameters
-----
raw_S1, raw_S2 : list of raws for each subject (for example first raw for subject 1 
corresponds to the baseline, second raw corresponds to the task). Raws are MNE objects
(data are ndarray and info is a disctionnary sampling parameters).

freq_bands : list of tuple summarizing frequency bands-of-interest.

Plots
-----
Power spectral density calculated with welch FFT for each epoch and each subject, 
average in each frequency band-of-interest.

Returns
-----
List of epochs (for example first epoch for subject 1 corresponds to the baseline, 
second epoch corresponds to the task) for each subject.

"""

def create_epochs(raw_S1, raw_S2, freq_bands):

    epoch_S1 = []
    epoch_S2 = []

    for raw1,raw2 in zip(raw_S1,raw_S2):
        
        ## creating fixed events 
        fixed_events1 = mne.make_fixed_length_events(raw1, id=1, start=0, stop=None, duration=1.0, first_samp=True, overlap=0.0)
        fixed_events2 = mne.make_fixed_length_events(raw2, id=1, start=0, stop=None, duration=1.0, first_samp=True, overlap=0.0)
        
        ## epoching the events per time window
        epoch1        = mne.Epochs(raw1, fixed_events1, event_id=1, tmin=0, tmax=1, baseline=None, preload=True, proj=True) # reject=reject_criteria, no baseline correction
                                                                                                                            # preload needed after
        epoch2        = mne.Epochs(raw2, fixed_events2, event_id=1, tmin=0, tmax=1, baseline=None, preload=True, proj=True)

        ## vizu topoplots of PSD for epochs
        # epoch1.plot()
        epoch1.plot_psd_topomap(bands=freq_bands) # welch FFT
        epoch1.plot_psd_topomap(bands=freq_bands) # welch FFT
        
        epoch_S1.append(epoch1)
        epoch_S2.append(epoch2)

    return epoch_S1, epoch_S2
  
