# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import mne
from mne.io.constants import FIFF

"""
Merging Epochs from 2 subjects after interpolation of bad channels for each subject.

Note that bad channels info is removed.

Note that average on reference can not be done anymore. Similarly, montage can not be set
to the data and as a result topographies in MNE are not possible anymore. Use toolbox 
vizualisations instead. 

Parameters
-----
epoch_S1,epoch_S2 : Epochs objects for each subject. epoch_S1 and epoch_S2 
correspond to a condition and can result from the concatenation of epochs from 
different occurences of the condition across experiments. 
Epochs are MNE objects (data are stored in an array of shape 
(n_epochs, n_channels, n_times) and info is a disctionnary sampling parameters).

Returns
-----
ep_hyper : Epochs object for the dyad (with merged data of the 2 subjects). The time alignement has 
been done qt raw data creation.

"""

def merge(epoch_S1,epoch_S2):
        
    ## checking bad ch for epochs, interpolating and removing them from 'bads' if needed
    if len(epoch_S1_concat.info['bads'])>0:
        epoch_S1_concat    = mne.Epochs.interpolate_bads(epoch_S1_concat, reset_bads=True, mode='accurate', origin='auto', verbose=None) # head-digitization-based origin fit
    if len(epoch_S2_concat.info['bads'])>0:
        epoch_S2_concat    = mne.Epochs.interpolate_bads(epoch_S2_concat, reset_bads=True, mode='accurate', origin='auto', verbose=None) 
    
    sfreq = epoch_S1_concat[0].info['sfreq']
    ch_names = epoch_S1_concat[0].info['ch_names']

    ## creating channels label for each subject
    ch_names1= []
    for i in ch_names:
        ch_names1.append(i+'_S1')
    ch_names2= []
    for i in ch_names:
        ch_names2.append(i+'_S2')

    merges=[]

    ## picking data per epoch
    for l in range(0,len(epoch_S1_concat)):
        data_S1 = epoch_S1_concat[l].get_data()
        data_S2 = epoch_S2_concat[l].get_data()

        data_S1=np.squeeze(data_S1,axis=0)
        data_S2=np.squeeze(data_S2,axis=0)

        dicdata1 = {i : data_S1[:,i] for i in range(0,len(data_S1[0,:]))} 
        dicdata2 = {i : data_S2[:,i] for i in range(0,len(data_S2[0,:]))}

        ## creating dataframe to merge data for each time point
        dataframe1 = pd.DataFrame(dicdata1, index = ch_names1)
        dataframe2 = pd.DataFrame(dicdata2, index = ch_names2)
        merge      = pd.concat([dataframe1,dataframe2])
        
        ## reconverting to array and joining the info file
        merge_arr  = merge.to_numpy()
        merges.append(merge_arr)

    merged = np.array(merges)    
    ch_names_merged= ch_names1+ch_names2
    info     = mne.create_info(ch_names_merged, sfreq, ch_types='eeg', 
                     montage=None, verbose=None)
    ep_hyper = mne.EpochsArray(merged, info)
    
    ## info about task
    ep_hyper.info['description'] = epoch_S1_concat[0].info['description']

    # ep_hyper.plot()
    
    return ep_hyper
