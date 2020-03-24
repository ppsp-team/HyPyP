#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : prep.py
# description     : data preprocessing functions
# author          : Anaël Ayrolles, Florence Brun, Guillaume Dumas,
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

from autoreject import get_rejection_threshold, AutoReject
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, corrmap
import numpy as np


def filt(epochs_concat):
    """Before ICA : Filter raw data to remove slow drifts
    Parameters
    ----------
    epoch_concat : instance of Epochs

    Returns
    -------
    epochs : instance of Epochs
    """
    epochs = []
    for epoch_concat in epochs_concat:  # per subj
        epochs.append(mne.filter.filter_data(
            epoch_concat, epoch_concat.info['sfreq'], l_freq=2., h_freq=None))
    return epochs


def ICA_choice_comp(icas, epochs):
    """Plot Independant Components for each subject, let the user
    choose the relevant components for artefacts rejection and apply ICA
    on Epochs.

    Parameters
    -----
    icas : list of independant components for each subject, IC are MNE objects.

    epochs : list of 2 Epochs objects (for each subject). Epochs_S1
    and Epochs_S2 correspond to a condition and can result from the
    concatenation of epochs from different occurences of the condition
    across experiments.
    Epochs are MNE objects (data are stored in an array of shape(n_epochs,
    n_channels, n_times) and info is a disctionnary sampling parameters).

    Returns
    -----
    List of 2 clean Epochs (for each subject).

    """
    # plotting Independant Components for each subject
    for ica in icas:
        ica.plot_components()

    # choosing subject and its component as a template for the other subject
    # if do not want to apply ICA on the data, do not fill the answer
    subj_numb = input("Which subject ICA do you want"
                      " to use as a template for artifacts rejection?"
                      " Index begins at zero. If you do not want to apply"
                      " ICA on your data, enter nothing.")
    comp_number = input("Which IC do you want to use as a template?"
                        " Index begins at zero. If you do not want to apply"
                        " ICA on your data, enter nothing.")

    # applying ICA
    if (len(subj_numb) != 0 and len(comp_number) != 0):
        cleaned_epochs_ICA = ICA_apply(icas,
                                       int(subj_numb),
                                       int(comp_number),
                                       epochs)
    else:
        cleaned_epochs_ICA = epochs

    return cleaned_epochs_ICA


def ICA_apply(icas, subj_number, comp_number, epochs):

    cleaned_epochs_ICA = []

    # selecting which ICs in the other subject correspond to the template
    template_eog_component = icas[subj_number].get_components()[:, comp_number]

    # applying corrmap with at least 1 component detected for each subj
    fig_template, fig_detected = corrmap(
        icas, template=template_eog_component, threshold=0.9,
        label='blink', ch_type='eeg')

    # labeling the ICs that capture blink artifacts
    print([ica.labels_ for ica in icas])

    # selecting ICA components after viz
    for i in icas:
        i.exclude = i.labels_['blink']

    # applying ica on clean_epochs
    for i in range(0, len(epochs)):
        for j in icas:  # per subj
            j.apply(epochs[i])
            cleaned_epochs_ICA.append(epochs[i])

    return cleaned_epochs_ICA


def ICA_fit(epochs, n_components, method, random_state):
    """Compute global Autorejection to fit Independant Components Analysis
    on Epochs, for each subject.

    Pre requisite : install autoreject
    https://api.github.com/repos/autoreject/autoreject/zipball/master

    Parameters
    -----
    epochs : list of 2 Epochs objects (for each subject).
             Epochs_S1 and Epochs_S2 correspond to a condition and can result
             from the concatenation of epochs from different occurences of the
             condition across experiments.
             Epochs are MNE objects (data are stored in an array of shape
             (n_epochs, n_channels, n_times) and info is a dictionnary
             sampling parameters).

    n_components : the number of principal components that are passed to the
                   ICA algorithm during fitting, for a first estimation,
                   n_components can be set to 15.

    method : the ICA method used, 'fastica', 'infomax' or 'picard'.
             Fastica' is the most frequently used.

    random_state : the parameter used to compute random distributions
                   for ICA calulation, int or None. It can be useful to fix
                   random_state value to have reproducible results. For 15
                   components, random_state can be set to 97 for example.

    Info
    -----
    If Autoreject and ICA take too much time, change the decim value (see MNE
    documentation).

    Returns
    -----
    List of independant components for each subject. IC are MNE objects, see
    MNE documentation for more details.
    """
    icas = []
    for epoch in epochs:  # per subj

        # applying AR to find global rejection threshold
        reject = get_rejection_threshold(epoch, ch_types='eeg')
        print('The rejection dictionary is %s' % reject)

        # fitting ICA on filt_raw after AR
        ica = ICA(n_components=n_components,
                  method=method, random_state=random_state)
        icas.append(ica.fit(epoch, reject=reject, tstep=1))

    return icas


def AR_local(cleaned_epochs_ICA):
    """Apply local Autoreject
    Parameters
    ----------
    clean_epoch_concat : instance of Epochs after global Autoreject
                         and ICA

    Returns
    -------
    cleaned_epochs_AR : instance of Epochs after local Autoreject
    """
    bad_epochs_AR = []

    # defaults values for n_interpolates and consensus_percs
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)

    for clean_epochs in cleaned_epochs_ICA:  # per subj

        picks = mne.pick_types(
            clean_epochs[0].info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[])

        ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                        thresh_method='random_search', random_state=42)

        # fitting AR to get bad epochs
        ar.fit(clean_epochs)
        reject_log = ar.get_reject_log(clean_epochs, picks=picks)
        bad_epochs_AR.append(reject_log)

    # taking bad epochs for min 1 subj (dyad)
    log1 = bad_epochs_AR[0]
    log2 = bad_epochs_AR[1]

    bad1 = np.where(log1.bad_epochs == True)
    bad2 = np.where(log2.bad_epochs == True)

    bad = list(set(bad1[0].tolist()).intersection(bad2[0].tolist()))
    print('%s percent of bad epochs' %
          int(len(bad)/len(list(log1.bad_epochs))*100))

    # picking good epochs for the two subj
    cleaned_epochs_AR = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        clean_epochs_AR = clean_epochs.drop(indices=bad)
        cleaned_epochs_AR.append(clean_epochs_AR)

    # Vizualisation before after AR
    evoked_before = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        evoked_before.append(clean_epochs.average())
    evoked_after_AR = []
    for clean in cleaned_epochs_AR:
        evoked_after_AR.append(clean.average())

    for i, j in zip(evoked_before, evoked_after_AR):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        for ax in axes:
            ax.tick_params(axis='x', which='both', bottom='off', top='off')
            ax.tick_params(axis='y', which='both', left='off', right='off')

        ylim = dict(grad=(-170, 200))
        i.pick_types(eeg=True, exclude=[])
        i.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
        axes[0].set_title('Before autoreject')
        j.pick_types(eeg=True, exclude=[])
        j.plot(exclude=[], axes=axes[1], ylim=ylim)
        # Problème titre ne s'affiche pas pour le deuxieme axe !!!
        axes[1].set_title('After autoreject')
        plt.tight_layout()
    return cleaned_epochs_AR
