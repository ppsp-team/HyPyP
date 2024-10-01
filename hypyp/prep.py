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
import copy
import matplotlib.pyplot as plt
import mne
from autoreject import get_rejection_threshold, AutoReject, RejectLog
from mne.preprocessing import ICA, corrmap
from typing import List, Tuple, TypedDict, Union

class DicAR(TypedDict):
    """
    Epoch rejection info
    """
    strategy: str
    threshold: float
    S1: float
    S2: float
    dyad: float


def filt(
    raw_S: List[mne.io.Raw],
    freqs: Tuple[Union[float, None], Union[float, None]] = (2., None)
) -> List[mne.io.Raw]:
    """
    Filters list of raw data to remove slow drifts.

    Arguments:
        raw_S: list of Raw data (as an example: different occurences of
            a condition for a participant). Raws are MNE objects.

    Returns:
        raws: list of high-pass filtered raws.
    """
  
    raws = [mne.io.Raw.filter(raw, l_freq=freqs[0], h_freq=freqs[1]) for raw in raw_S]
    return raws


def ICA_choice_comp(icas: List[ICA], epochs: List[mne.Epochs]) -> List[mne.Epochs]:
    """
    Plots Independent Components for each participant (calculated from Epochs),
    let the user choose the relevant components for artifact rejection
    and apply ICA on Epochs.

    Arguments:
        icas: list of Independent Components for each participant (IC are MNE
            objects).
        epochs: list of 2 Epochs objects (for each participant). Epochs_S1
            and Epochs_S2 correspond to a condition and can result from the
            concatenation of Epochs from different experimental realisations
            of the condition.
            Epochs are MNE objects: data are stored in an array of shape
            (n_epochs, n_channels, n_times) and parameters information is
            stored in a disctionnary.

    Returns:
        cleaned_epochs_ICA: list of 2 cleaned Epochs for each participant
          (the chosen IC have been removed from the signal).
    """
    # plotting Independant Components for each participant
    for ica in icas:
        ica.plot_components()

    # choosing participant and its component as a template for the other participant
    # if do not want to apply ICA on the data, do not fill the answer
    subject_id = input("Which participant ICA do you want"
                        " to use as a template for artifact rejection?"
                        " Index begins at zero. (If you do not want to apply"
                        " ICA on your data, do not enter nothing and press enter.)")

    component_id = input("Which IC do you want to use as a template?"
                        " Index begins at zero. (If you did not choose"
                        " a participant number at first question,"
                        " then do not enter nothing and press enter again"
                        " to not apply ICA on your data)")

    if (len(subject_id) == 0 or len(component_id) == 0):
        return epochs

    return ICA_apply(icas, int(subject_id), int(component_id), epochs)


def ICA_apply(icas: List[ICA], subject_id: int, component_id: int, epochs: List[mne.Epochs], plot: bool = True) -> List[mne.Epochs]:
    """
    Applies ICA with template model from 1 participant in the dyad.
    See ICA_choice_comp for a detailed description of the parameters and output.
    """

    cleaned_epochs_ICA: List[ICA] = []

    # selecting which ICs corresponding to the template
    template_eog_component = icas[subject_id].get_components()[:, component_id]

    # applying corrmap with at least 1 component detected for each subj
    corrmap(icas,
        template=template_eog_component,
        threshold=0.9,
        label='blink',
        ch_type='eeg',
        plot=plot,
    )

    # labeling the ICs that capture blink artifacts
    print([ica.labels_ for ica in icas])

    # selecting ICA components after viz
    for ica in icas:
        ica.exclude = ica.labels_['blink']

    # applying ica on clean_epochs
    # for each participant
    for subject_id, ica in zip(range(0, len(epochs)), icas):
        # taking all channels to apply ICA
        epochs_subj = mne.Epochs.copy(epochs[subject_id])
        bads_keep = epochs_subj.info['bads']
        epochs_subj.info['bads'] = []
        ica.apply(epochs_subj)
        epochs_subj.info['bads'] = bads_keep
        cleaned_epochs_ICA.append(epochs_subj)

    return cleaned_epochs_ICA


def ICA_fit(
    epochs: List[mne.Epochs],
    n_components: int,
    method: str,
    fit_params: dict,
    random_state: int
) -> List[ICA]:
    """
    Computes global Autorejection to fit Independent Components Analysis
    on Epochs, for each participant.

    Pre requisite : install autoreject
    https://api.github.com/repos/autoreject/autoreject/zipball/master

    Arguments:
        epochs: list of 2 Epochs objects (for each participant).
            Epochs_S1 and Epochs_S2 correspond to a condition and can result
            from the concatenation of Epochs from different experimental
            realisations of the condition (Epochs are MNE objects).
        n_components: the number of principal components that are passed to the
            ICA algorithm during fitting, int. For a first estimation,
            n_components can be set to 15.
        method: the ICA method used, str 'fastica', 'infomax' or 'picard'.
            'Fastica' is the most frequently used. Use the fit_params argument to set
            additional parameters. Specifically, if you want Extended Infomax, set
            method=’infomax’ and fit_params=dict(extended=True) (this also works
            for method=’picard’). 
        fit_params: Additional parameters passed to the ICA estimator
            as specified by method. None by default.
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
        icas: list of Independant Components for each participant (IC are MNE
            objects, see MNE documentation for more details).
    """
    icas: List[ICA] = []
    for epochs_subj in epochs:
        # per subj
        # applying AR to find global rejection threshold
        reject = get_rejection_threshold(epochs_subj, ch_types='eeg')
        # if very long, can change decim value
        print(f"The rejection dictionary is {reject}")

        # fitting ICA on filt_raw after AR
        ica = ICA(n_components=n_components,
                  method=method,
                  fit_params= fit_params,
                  random_state=random_state)

        # take bad channels into account in ICA fit
        epochs_fit = mne.Epochs.copy(epochs_subj)
        epochs_fit.info['bads'] = []
        epochs_fit.drop_bad(reject=reject, flat=None)
        icas.append(ica.fit(epochs_fit))

    return icas


def AR_local(cleaned_epochs_ICA: List[mne.Epochs], strategy: str = 'union', threshold: float = 50.0, verbose: bool = False) -> Tuple[mne.Epochs, DicAR]:
    """
    Applies local Autoreject to repair or reject bad epochs.

    Arguments:
        cleaned_epochs_ICA: list of Epochs after global Autoreject and ICA.
        strategy: more or less generous strategy to reject bad epochs: 'union'
            or 'intersection'. 'union' rejects bad epochs from subject 1 and
            subject 2 immediatly, whereas 'intersection' rejects shared bad epochs
            between subjects, tries to repare remaining bad epochs per subject,
            reject the non-reparable per subject and finally equalize epochs number
            between subjects. Set to 'union' by default.
        threshold: percentage of epochs removed that is accepted. Above
            this threshold, data are considered as a too shortened sample
            for further analyses. Set to 50.0 by default.
        verbose: option to plot data before and after AR, boolean, set to
            False by default. # use verbose = false until next Autoreject update

    Note:
        To reject or repair epochs, parameters are more or less conservative,
        see http://autoreject.github.io/generated/autoreject.AutoReject.

    Returns:
        cleaned_epochs_AR: list of Epochs after local Autoreject.
        dic_AR: dictionnary with the percentage of epochs rejection
            for each subject and for the intersection of the them.
    """

    reject_logs: List[RejectLog] = []
    AR: List[AutoReject] = []
    dic_AR: DicAR = {}
    dic_AR['strategy'] = strategy
    dic_AR['threshold'] = threshold

    # defaults values for n_interpolates and consensus_percs
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    # more generous values
    # n_interpolates = np.array([16, 32, 64])
    # n_interpolates = np.array([1, 4, 8, 16, 32, 64])
    # consensus_percs = np.linspace(0.5, 1.0, 11)

    for subject_id, clean_epochs_subj in enumerate(cleaned_epochs_ICA):  # per subj
        picks = mne.pick_types(
            clean_epochs_subj[subject_id].info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude=[])

        ar = AutoReject(n_interpolates,
            consensus_percs,
            picks=picks,
            thresh_method='random_search',
            random_state=42,
            verbose='tqdm_notebook')

        # fitting AR to get bad epochs
        ar.fit(clean_epochs_subj)
        reject_log = ar.get_reject_log(clean_epochs_subj, picks=picks)

        AR.append(ar)
        reject_logs.append(reject_log)

    # taking bad epochs for min 1 subj (dyad)
    bad1_idx = np.where(reject_logs[0].bad_epochs == True)[0].tolist()
    bad2_idx = np.where(reject_logs[1].bad_epochs == True)[0].tolist()

    if strategy == 'union':
        bad_idx = list(set(bad1_idx).union(set(bad2_idx)))
    elif strategy == 'intersection':
        bad_idx = list(set(bad1_idx).intersection(set(bad2_idx)))
    else:
        raise RuntimeError('not good strategy input!')

    # storing the percentage of epochs rejection
    dic_AR['S1'] = float((len(bad1_idx) / len(cleaned_epochs_ICA[0])) * 100)
    dic_AR['S2'] = float((len(bad2_idx) / len(cleaned_epochs_ICA[1])) * 100)

    # picking good epochs for the two subj
    cleaned_epochs_AR: List[mne.Epochs] = []

    for subject_id, clean_epochs_subj in enumerate(cleaned_epochs_ICA):  # per subj
        # keep a copy of the original data
        epochs_subj = copy.deepcopy(clean_epochs_subj)
        epochs_subj.drop(indices=bad_idx)
        # interpolating bads or removing epochs
        ar = AR[subject_id]
        epochs_AR = ar.transform(epochs_subj)
        cleaned_epochs_AR.append(epochs_AR)

    if strategy == 'intersection':
        # equalizing epochs length between two participants
        mne.epochs.equalize_epoch_counts(cleaned_epochs_AR)

    n_epochs_ICA = len(cleaned_epochs_ICA[0])
    n_epochs_AR = len(cleaned_epochs_AR[0])

    dic_AR['dyad'] = float(((n_epochs_ICA - n_epochs_AR) / n_epochs_ICA) * 100)
    if dic_AR['dyad'] >= threshold:
        raise RuntimeError(f"percentage of rejected epochs ({dic_AR['dyad']}) above threshold ({threshold})! ")
    if verbose:
        print(f"{dic_AR['dyad']} percent of bad epochs")

    # Vizualisation before after AR
    evoked_before: List[mne.Evoked] = [epochs.average() for epochs in cleaned_epochs_ICA]
    evoked_after_AR: List[mne.Evoked] = [epochs.average() for epochs in cleaned_epochs_AR]

    if verbose:
        for evoked_before_subj, evoked_after_AR_subj in zip(evoked_before, evoked_after_AR):
            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            for ax in axes:
                ax.tick_params(axis='x', which='both', bottom='off', top='off')
                ax.tick_params(axis='y', which='both', left='off', right='off')

            ylim = dict(grad=(-170, 200))

            evoked_before_subj.pick_types(eeg=True, exclude=[])
            evoked_before_subj.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
            axes[0].set_title('Before autoreject')

            evoked_after_AR_subj.pick_types(eeg=True, exclude=[])
            evoked_after_AR_subj.plot(exclude=[], axes=axes[1], ylim=ylim)
            # Problème titre ne s'affiche pas pour le deuxieme axe !!!
            axes[1].set_title('After autoreject')

            plt.tight_layout()

    return cleaned_epochs_AR, dic_AR
