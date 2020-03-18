# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import mne
from mne.io.constants import FIFF

"""
Split merged raw data into 2 subjects raw data.

Note that subject's raw data is set to the standard montage 1020 available in MNE.
An average is computed to avoid reference bias (see MNE documentation about set_eeg_reference).

Parameters
-----
raw_merge : Raw data for the dyad with data from subject 1 and data from subject 2
(channels name are defined with the suffix S1 or S2 respectively).

Returns 
-----
raw_1020_S1, raw_1020_S2 : Raw data for each subject separately.
Raws are MNE objects (data are ndarray with shape (n_channels, n_times) 
and info is a disctionnary sampling parameters).

"""

def split(raw_merge):
    
    ch_S1 = []
    ch_S2 = []
    ch = []
    for name in raw_merge.info['ch_names']:
        if name.endswith('S1'):
            ch_S1.append(name) 
            ch.append(name.split('_')[0])
        elif name.endswith('S2'):
            ch_S2.append(name)

    ## picking individual subject data
    data_S1 = raw_merge.get_data(picks=ch_S1)
    data_S2 = raw_merge.get_data(picks=ch_S2)
    
    ## creating info for raws
    info    = mne.create_info(ch, raw_merge.info['sfreq'], ch_types='eeg', 
                 montage=None, verbose=None)
    raw_S1  = mne.io.RawArray(data_S1, info)
    raw_S2  = mne.io.RawArray(data_S2, info)
    
    ## setting info about channels and task
    raw_1020_S1.info['bads'] = [ch.split('_')[0] for ch in ch_S1 if ch in raw_merge.info['bads']]
    raw_1020_S2.info['bads'] = [ch.split('_')[0] for ch in ch_S2 if ch in raw_merge.info['bads']]
    for raws in (raw_S1,raw_S2):
        raws.info['description'] = raw_merge.info['description']
        raws.info['events'] = raw_merge.info['events']

    ## setting montage 94 electrodes (ignore somes to correspond to our data)
        for ch in raws.info['chs']:
            if ch['ch_name'].startswith('MOh') or ch['ch_name'].startswith('MOb'):
                # print('emg')
                ch['kind'] = FIFF.FIFFV_EOG_CH
            else:
                ch['kind'] = FIFF.FIFFV_EEG_CH
    montage  = mne.channels.make_standard_montage('standard_1020')
    raw_1020_S1 = raw_S1.copy().set_montage(montage)
    raw_1020_S2 = raw_S2.copy().set_montage(montage)
    # raw_1020_S1.plot_sensors()

    ## set reference to electrodes average (instate of initial ref to avoid ref biais)
    ## and storing it in raw.info['projs']: applied when Epochs
    raw_1020_S1, _ = mne.set_eeg_reference(raw_1020_S1, 'average', projection=True)
    raw_1020_S2, _ = mne.set_eeg_reference(raw_1020_S2, 'average', projection=True)

    # TO DO annotations, subj name, events, task description different across subj

    # raw_1020_S1.plot()
    # raw_1020_S1.plot_psd()
    
    return raw_1020_S1, raw_1020_S2
