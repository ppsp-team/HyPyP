# coding=utf-8

import mne
import numpy as np 
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import seaborn 
import statistics

"""
Compute Power Spectral Density calculation on Epochs for a condition and the baseline,
normalize PSD in the condition by the baseline.


Parameters
-----
epochs_baseline, epochs_task : Epochs object for the baseline and a condition 
('task' for example), for a subject. epochs_baseline, epochs_task can result 
from the concatenation of epochs from different occurences of the condition 
across experiments. Epochs are MNE objects (data are stored in an array of shape 
(n_epochs, n_channels, n_times) and info is a disctionnary sampling parameters).


Note that the function can be iterated on the group and/or on conditions:
for epochs_baseline,epochs_task in zip(epochs['epochs_%s_%s_%s_baseline' 
% (subj,group,cond_name)], epochs['epochs_%s_%s_%s_task' % (subj,group,cond_name)]).

You can then visualize PSD distribution on the group with the toolbox vizualisation
to check normality for statistics for example.

fmin, fmax : minimum and maximum frequencies for Power Spectral Density calculation (in Hz).


Returns 
-----
m_baseline, psds_welch_task_m : PSD average across epochs for each channel and each frequency,
for baseline and 'task' condition respectively.

psd_mean_task_normZ, psd_mean_task_normLog : Zscore and Logratio of average PSD 
during 'task' condition (normalisation by baseline condition).

"""

def PSD(epochs_baseline, epochs_task, fmin, fmax):
    
    ## dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs_baseline.info['chs']:
        if ch['kind'] == 202 : # FIFFV_EOG_CH
            epochs_baseline.drop_channels([ch['ch_name']])
    for ch in epochs_task.info['chs']:
        if ch['kind'] == 202 : # FIFFV_EOG_CH
            epochs_task.drop_channels([ch['ch_name']])
    
    ## computing power spectral density on epochs signal
    # average in the 1second window around event (mean but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=1) 
    psds_welch_baseline, freqs_mean = psd_welch(epochs_baseline, **kwargs, average='mean', picks='all') # or median
    psds_welch_task, freqs_mean = psd_welch(epochs_task, **kwargs, average='mean', picks='all') # or median

    ## averaging power across epochs for each ch and each f
    m_baseline = np.mean(psds_welch_baseline, axis=0)
    std_baseline = np.std(psds_welch_baseline, axis=0)
    psds_welch_task_m = np.mean(psds_welch_task, axis=0)
    
    # normalizing power during task by baseline average power across events
    # Z score
    s = np.subtract(psds_welch_task_m,m_baseline)
    psd_mean_task_normZ = np.divide(s,std_baseline)
    # Log ratio
    d = np.divide(psds_welch_task_m,m_baseline)
    psd_mean_task_normLog = np.log10(d)
    
    return m_baseline, psds_welch_task_m, psd_mean_task_normZ, psd_mean_task_normLog