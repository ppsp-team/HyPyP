#Load useful libs

#Data science
import numpy as np

#MNE
import mne

#Load data 
# For each participant, both .set file and .fdt files should be in the same directory.
# In this notebook, we use data that is preprocessed in EEGlab: data should be epoched.
# Note: it is not necessary that participants have the same number of epochs, but they must have same number of samples per epoch 

path_1 = "../data/EEGLAB/eeglab_data_epochs_ica.set"
path_2 = "../data/EEGLAB/eeglab_data_epochs_ica.set"

epo1 = mne.io.read_epochs_eeglab(
    path_1
)

epo2 = mne.io.read_epochs_eeglab(
    path_2
)

#We need to equalize the number of epochs between our two participants.
mne.epochs.equalize_epoch_counts([epo1, epo2])

#Choosing sensors in international standard 10/20 system for using the MNE template and overwrite the EEGlab position. 
#Note: EOG (Eyes) are removed in this analysis

StanSys = ['Nz', 'Fp1', 'Fpz', 'Fp2', 'F7', 'F9', 'PO3',
                     'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'CP3', 'PO4',
                     'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P3', 'PO7',
                     'P4', 'T6', 'O1', 'Oz', 'O2', 'Iz', 'P1', 'PO8',
                     'AF3', 'AF7', 'AF4', 'AF8', 'F6', 'F10', 'F2',  'F5', 
                     'FC1', 'FC3', 'FC5', 'FCz', 'FC2', 'FC4', 'FC6', 'F1',
                     'FT9', 'FT7', 'FT8', 'FT10', 'C5', 'C6', 'CPz', 'CP1', 
                     'CP5', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'TP7', 'TP9', 
                     'P5', 'P7', 'P9', 'P2', 'P4', 'P6', 'P8', 'P10', 'POz']

low_StanSys = []
for name in StanSys:
  low_StanSys.append(name.lower())

names_epo1 = np.array([ch['ch_name'] for ch in epo1.info['chs']])
names_epo2 = np.array([ch['ch_name'] for ch in epo2.info['chs']])

epo_ref1 = epo1.copy()
for idx in range(len(names_epo1)):
  aux_name = names_epo1[idx].lower()
  if aux_name in low_StanSys:
    ind = low_StanSys.index(aux_name)
    nw_ch = StanSys[ind]
    mne.rename_channels(epo_ref1.info, {names_epo1[idx]:nw_ch})
  else:
    epo_ref1.drop_channels(names_epo1[idx]) 

epo_ref2 = epo2.copy()
for idx in range(len(names_epo2)):
  aux_name = names_epo2[idx].lower()
  if aux_name in low_StanSys:
    ind = low_StanSys.index(aux_name)
    nw_ch = StanSys[ind]
    mne.rename_channels(epo_ref2.info, {names_epo2[idx]:nw_ch})
  else:
    epo_ref2.drop_channels(names_epo2[idx]) 

locations = mne.channels.make_standard_montage('standard_1020', head_size=0.092)

epo_ref2.set_montage(locations)

epo_ref1.set_montage(locations)

#epo_ref1 and epo_ref2 are now compatible with tools provided by HyPyP (for visualization and statictical analyses check (https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb) tutorial

