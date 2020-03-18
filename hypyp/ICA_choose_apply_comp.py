# coding=utf-8

import mne
from mne.preprocessing import ICA, corrmap
import numpy as np 
import tkinter as tk
from tkinter import simpledialog

"""
Plot Independant Components for each subject, let the user choose the relevant components
for artefacts rejection and apply ICA on Epochs.

Parameters
-----
icas : list of independant components for each subject, IC are MNE objects.

epochs : list of 2 Epochs objects (for each subject). Epochs_S1 and Epochs_S2 
correspond to a condition and can result from the concatenation of epochs from 
different occurences of the condition across experiments.
Epochs are MNE objects (data are stored in an array of shape 
(n_epochs, n_channels, n_times) and info is a disctionnary sampling parameters).

Returns
-----
List of 2 clean Epochs (for each subject).

"""


def ICA_choice_comp(icas, epochs):

    ## plotting Independant Components for each subject
    for ica in icas:
        ica.plot_components()

    ## choosing subject and its component as a template for the other subject
    # if do not want to apply ICA on the data, do not fill the answer
    window = tk.Tk()
    window.withdraw()
    subj_numb = simpledialog.askstring(title="choice ICA template",
    prompt="Which subject ICA do you want to use as a template for artifacts rejection?")
    comp_number = simpledialog.askstring(title="choice ICA template",
    prompt="Which IC do you want to use as a template?") 
    
    ## applyinf ICA
    if (len(subj_numb)!=0 and len(comp_number)!=0):
        cleaned_epochs_ICA = ICA_apply(icas,int(subj_numb),int(comp_number),epochs)
    else:
        cleaned_epochs_ICA = epochs
        
    return cleaned_epochs_ICA


def ICA_apply(icas, subj_number, comp_number, epochs):
    
    cleaned_epochs_ICA = []

    ## selecting which ICs in the other subject correspond to the template 
    template_eog_component = icas[subj_number].get_components()[:,comp_number]

    ## applying corrmap with at least 1 component detected for each subj
    fig_template, fig_detected = corrmap(icas, template=template_eog_component, threshold=0.9, label='blink', ch_type='eeg')

    ## labeling the ICs that capture blink artifacts
    print([ica.labels_ for ica in icas])

    ## selecting ICA components after viz
    for i in icas:
        i.exclude = i.labels_['blink']
    
    ## applying ica on clean_epochs
    for i in range(0,len(epochs)):
        for j in icas: # per subj
            j.apply(epochs[i])
            cleaned_epochs_ICA.append(epochs[i])

    return cleaned_epochs_ICA
