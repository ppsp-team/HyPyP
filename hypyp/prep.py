#!/usr/bin/env python
# coding=utf-8

"""
Data preprocessing functions

| Option | Description |
| ------ | ----------- |
| title           | prep.py |
| authors         | Anaël Ayrolles, Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


import numpy as np
import matplotlib.pyplot as plt
import mne
from autoreject import get_rejection_threshold, AutoReject
from mne.preprocessing import ICA, corrmap


def filt(raw_S: list) -> list:
    """
    Filters list of raw data to remove slow drifts.

    Arguments:
        raw_S: list of Raw data (as an example: different occurences of
          a condition for a subject). Raws are MNE objects.

    Returns:
        raws: list of high-pass filtered raws.
    """
    # TODO: l_freq and h_freq as param
    raws = []
    for raw in raw_S:
        raws.append(mne.io.Raw.filter(raw, l_freq=2., h_freq=None))

    return raws


def ICA_choice_comp(icas: list, epochs: list) -> list:
    """
    Plots Independent Components for each subject (calculated from Epochs),
    let the user choose the relevant components for artifact rejection
    and apply ICA on Epochs.

    Arguments:
        icas: list of Independent Components for each subject (IC are MNE
          objects).
        epochs: list of 2 Epochs objects (for each subject). Epochs_S1
          and Epochs_S2 correspond to a condition and can result from the
          concatenation of Epochs from different experimental realisations
          of the condition.
          Epochs are MNE objects: data are stored in an array of shape
          (n_epochs, n_channels, n_times) and parameters information is
          stored in a disctionnary.

    Returns:
        cleaned_epochs_ICA: list of 2 cleaned Epochs for each subject
          (the chosen IC have been removed from the signal).
    """
    # plotting Independant Components for each subject
    for ica in icas:
        ica.plot_components()

    # choosing subject and its component as a template for the other subject
    # if do not want to apply ICA on the data, do not fill the answer
    subj_numb = input("Which subject ICA do you want"
                      " to use as a template for artifact rejection?"
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


def ICA_apply(icas: int, subj_number: int, comp_number: int, epochs: list) -> list:
    """
    Applies ICA with template model from 1 subject in the dyad.
    See ICA_choice_comp for a detailed description of the parameters and output.
    """

    cleaned_epochs_ICA = []
    # selecting which ICs corresponding to the template
    template_eog_component = icas[subj_number].get_components()[:, comp_number]

    # applying corrmap with at least 1 component detected for each subj
    fig_template, fig_detected = corrmap(icas,
                                         template=template_eog_component,
                                         threshold=0.9,
                                         label='blink',
                                         ch_type='eeg')

    # labeling the ICs that capture blink artifacts
    print([ica.labels_ for ica in icas])

    # selecting ICA components after viz
    for i in icas:
        i.exclude = i.labels_['blink']

    epoch_all_ch = []
    # applying ica on clean_epochs
    # for each subject
    for i, j in zip(range(0, len(epochs)), icas):
        # taking all channels to apply ICA
        bads = epochs[i].info['bads']
        epoch_all_ch.append(mne.Epochs.copy(epochs[i]))
        epoch_all_ch[i].info['bads'] = []
        j.apply(epoch_all_ch[i])
        epoch_all_ch[i].info['bads'] = bads
        cleaned_epochs_ICA.append(epoch_all_ch[i])

    return cleaned_epochs_ICA


def ICA_fit(epochs: list, n_components: int, method: str, random_state: int) -> list:
    """
    Computes global Autorejection to fit Independent Components Analysis
    on Epochs, for each subject.

    Pre requisite : install autoreject
    https://api.github.com/repos/autoreject/autoreject/zipball/master

    Arguments:
        epochs: list of 2 Epochs objects (for each subject).
          Epochs_S1 and Epochs_S2 correspond to a condition and can result
          from the concatenation of Epochs from different experimental
          realisations of the condition (Epochs are MNE objects).
        n_components: the number of principal components that are passed to the
          ICA algorithm during fitting, int. For a first estimation,
          n_components can be set to 15.
        method: the ICA method used, str 'fastica', 'infomax' or 'picard'.
          'Fastica' is the most frequently used.
        random_state: the parameter used to compute random distributions
          for ICA calulation, int or None. It can be useful to fix
          random_state value to have reproducible results. For 15
          components, random_state can be set to 97, for 20 components to 0
          for example.

    Note:
        If Autoreject and ICA take too much time, change the decim value
        (see MNE documentation).
        Please filter the Epochs between 2 and 30 Hz before ICA fit
        (mne.Epochs.filter(epoch, 2, 30, method='fir')).

    Returns:
        icas: list of Independant Components for each subject (IC are MNE
          objects, see MNE documentation for more details).
    """
    icas = []
    for epoch in epochs:
        # per subj
        # applying AR to find global rejection threshold
        reject = get_rejection_threshold(epoch, ch_types='eeg')
        # if very long, can change decim value
        print('The rejection dictionary is %s' % reject)

        # fitting ICA on filt_raw after AR
        ica = ICA(n_components=n_components,
                  method=method,
                  random_state=random_state).fit(epoch)
        # take bad channels into account in ICA fit
        epoch_all_ch = mne.Epochs.copy(epoch)
        epoch_all_ch.info['bads'] = []
        icas.append(ica.fit(epoch_all_ch, reject=reject, tstep=1))

    return icas


def AR_local(cleaned_epochs_ICA: list, verbose: bool = False) -> list:
    """
    Applies local Autoreject to repair or reject bad epochs.

    Arguments:
        clean_epochs_ICA: list of Epochs after global Autoreject and ICA.
        verbose: option to plot data before and after AR, boolean, set to
          False by default.

    Note:
        To reject or repair epochs, parameters are more or less conservative,
        see http://autoreject.github.io/generated/autoreject.AutoReject.  

    Returns:
        cleaned_epochs_AR: list of Epochs after local Autoreject.
    """
    bad_epochs_AR = []
    AR = []

    # defaults values for n_interpolates and consensus_percs
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    # more generous values
    # n_interpolates = np.array([16, 32, 64])
    # n_interpolates = np.array([1, 4, 8, 16, 32, 64])
    # consensus_percs = np.linspace(0.5, 1.0, 11)

    for clean_epochs in cleaned_epochs_ICA:  # per subj

        picks = mne.pick_types(
            clean_epochs[0].info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[])

        if verbose:
            ar_verbose = 'progressbar'
        else:
            ar_verbose = False

        ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                        thresh_method='random_search', random_state=42,
                        verbose=ar_verbose)
        AR.append(ar)

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
    if verbose:
        print('%s percent of bad epochs' % int(len(bad)/len(list(log1.bad_epochs))*100))

    # picking good epochs for the two subj
    cleaned_epochs_AR = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        clean_epochs_ep = clean_epochs.drop(indices=bad)
        # interpolating bads or removing epochs
        ar = AR[cleaned_epochs_ICA.index(clean_epochs)]
        clean_epochs_AR = ar.transform(clean_epochs_ep)
        cleaned_epochs_AR.append(clean_epochs_AR)
    # equalizing epochs length between two subjects
    mne.epochs.equalize_epoch_counts(cleaned_epochs_AR)

    # Vizualisation before after AR
    evoked_before = []
    for clean_epochs in cleaned_epochs_ICA:  # per subj
        evoked_before.append(clean_epochs.average())

    evoked_after_AR = []
    for clean in cleaned_epochs_AR:
        evoked_after_AR.append(clean.average())

    if verbose:
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
