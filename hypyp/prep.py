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
    Dictionary type for storing epoch rejection information.
    
    This type provides structured information about the rejection process,
    including the rejection strategy, threshold, and percentage of epochs
    rejected for each participant and the dyad.
    
    Attributes
    ----------
    strategy : str
        The strategy used for epoch rejection ('union' or 'intersection')
        
    threshold : float
        The maximum allowed percentage of epochs to be rejected
        
    S1 : float
        Percentage of epochs rejected for participant 1
        
    S2 : float
        Percentage of epochs rejected for participant 2
        
    dyad : float
        Overall percentage of epochs rejected across both participants
    """

    strategy: str
    threshold: float
    S1: float
    S2: float
    dyad: float


def filt(raw_S: List[mne.io.Raw], 
        freqs: Tuple[Union[float, None], Union[float, None]] = (2., None)) -> List[mne.io.Raw]:
    """
    Filter a list of raw EEG data to remove slow drifts or other unwanted frequency components.
    
    This function applies a high-pass or band-pass filter to each Raw object in the input list.
    Filtering helps to remove low-frequency drifts, power line noise, or other frequency-specific
    artifacts from the EEG data.
    
    Parameters
    ----------
    raw_S : List[mne.io.Raw]
        List of Raw objects containing continuous EEG data
        
    freqs : Tuple[Union[float, None], Union[float, None]], optional
        Frequency range for filtering (default=(2., None)):
        - First element: Lower frequency bound (high-pass filter cutoff)
        - Second element: Upper frequency bound (low-pass filter cutoff)
        - None for either bound means no filtering in that direction
    
    Returns
    -------
    raws : List[mne.io.Raw]
        List of filtered Raw objects with the same structure as the input
    
    Notes
    -----
    By default, a 2 Hz high-pass filter is applied, which effectively removes slow
    drifts while preserving most of the EEG signal content. This is appropriate
    for most EEG analyses but may need adjustment for specific paradigms.
    
    Examples
    --------
    >>> # Filter data with a high-pass at 1 Hz
    >>> filtered_raws = filt(raw_list, freqs=(1., None))
    
    >>> # Apply a band-pass filter between 1 and 40 Hz
    >>> filtered_raws = filt(raw_list, freqs=(1., 40.))
    """
  
    raws = [mne.io.Raw.filter(raw, l_freq=freqs[0], h_freq=freqs[1]) for raw in raw_S]
    return raws


def ICA_choice_comp(icas: List[ICA], epochs: List[mne.Epochs]) -> List[mne.Epochs]:
    """
    Select ICA components for artifact rejection and apply ICA cleaning to epochs.
    
    This interactive function plots the Independent Components for each participant,
    lets the user choose relevant components for artifact rejection, and applies
    the cleaning to the epochs data.
    
    Parameters
    ----------
    icas : List[ICA]
        List of fitted ICA objects (one for each participant)
        
    epochs : List[mne.Epochs]
        List of Epochs objects to clean (one for each participant)
    
    Returns
    -------
    cleaned_epochs_ICA : List[mne.Epochs]
        List of ICA-cleaned Epochs objects (one for each participant)
    
    Notes
    -----
    The function uses an interactive approach:
    1. It plots the ICA components for each participant
    2. It prompts the user to select a participant and component to use as a template
    3. It uses corrmap to find similar components across participants
    4. It removes the identified components from all participants' data
    
    If you don't want to apply ICA cleaning, simply press Enter without typing
    anything when prompted for participant and component selection.
    
    Examples
    --------
    >>> # This function is interactive and prompts the user for input
    >>> cleaned_epochs = ICA_choice_comp(icas, epochs)
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


def ICA_apply(icas: List[ICA], subject_id: int, component_id: int, 
             epochs: List[mne.Epochs], plot: bool = True) -> List[mne.Epochs]:
    """
    Apply ICA artifact rejection using a template component.
    
    This function uses a component from one participant as a template to identify
    similar artifact components in all participants, then removes these components
    from the data.
    
    Parameters
    ----------
    icas : List[ICA]
        List of fitted ICA objects (one for each participant)
        
    subject_id : int
        Index of the participant whose component will serve as the template
        
    component_id : int
        Index of the component to use as the template
        
    epochs : List[mne.Epochs]
        List of Epochs objects to clean (one for each participant)
        
    plot : bool, optional
        Whether to plot the identified components (default=True)
    
    Returns
    -------
    cleaned_epochs_ICA : List[mne.Epochs]
        List of ICA-cleaned Epochs objects (one for each participant)
    
    Notes
    -----
    This function uses MNE's corrmap function to identify components similar to
    the template component. It then labels these components as 'blink' artifacts
    and removes them from the data.
    
    The threshold for component similarity is set to 0.9 by default.
    
    Examples
    --------
    >>> # Use the first component of the first participant as template
    >>> cleaned_epochs = ICA_apply(icas, subject_id=0, component_id=0, epochs=epochs)
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


def ICA_fit(epochs: List[mne.Epochs], n_components: int, method: str, 
           fit_params: dict, random_state: int) -> List[ICA]:
    """
    Compute Independent Component Analysis (ICA) on epochs for artifact rejection.
    
    This function applies global Autoreject to establish rejection thresholds,
    then fits ICA on the cleaned data. ICA is a commonly used technique to identify
    and remove artifacts such as eye blinks, muscle activity, and cardiac artifacts.
    
    Parameters
    ----------
    epochs : List[mne.Epochs]
        List of Epochs objects (one for each participant)
        
    n_components : int
        Number of principal components to pass to the ICA algorithm
        For a first estimation, a value around 15 is often appropriate
        
    method : str
        ICA method to use. Options:
        - 'fastica': FastICA algorithm (most commonly used)
        - 'infomax': Infomax algorithm
        - 'picard': Picard algorithm
        
    fit_params : dict
        Additional parameters passed to the ICA estimator
        For Extended Infomax, use method='infomax' and fit_params=dict(extended=True)
        
    random_state : int
        Random seed for reproducible results
        For 15 components, random_state=97 works well
        For 20 components, random_state=0 works well
    
    Returns
    -------
    icas : List[ICA]
        List of fitted ICA objects (one for each participant)
    
    Notes
    -----
    Pre-requisites:
    - Install autoreject: https://api.github.com/repos/autoreject/autoreject/zipball/master
    - Filter the Epochs between 2 and 30 Hz before ICA fitting
    
    If Autoreject and ICA take too much time, try changing the 'decim' value
    in the ICA initialization to downsample the data during fitting.
    
    Examples
    --------
    >>> # Fit ICA with 15 components using FastICA
    >>> icas = ICA_fit(epochs_list, n_components=15, method='fastica', 
    ...               fit_params=None, random_state=97)
    
    >>> # Fit ICA with Extended Infomax
    >>> icas = ICA_fit(epochs_list, n_components=15, method='infomax', 
    ...               fit_params=dict(extended=True), random_state=97)
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


def AR_local(cleaned_epochs_ICA: List[mne.Epochs], strategy: str = 'union', 
            threshold: float = 50.0, verbose: bool = False) -> Tuple[mne.Epochs, DicAR]:
    """
    Apply local Autoreject to repair or reject bad epochs.
    
    After ICA cleaning, this function identifies remaining problematic epochs and
    either repairs them (by interpolating bad channels) or rejects them entirely.
    It can use different strategies to handle epochs that are bad in only one participant.
    
    Parameters
    ----------
    cleaned_epochs_ICA : List[mne.Epochs]
        List of Epochs objects after ICA cleaning (one for each participant)
        
    strategy : str, optional
        Strategy for handling bad epochs (default='union'):
        - 'union': Reject epochs that are bad in either participant
        - 'intersection': Reject only epochs that are bad in both participants,
          attempt to repair other bad epochs individually
        
    threshold : float, optional
        Maximum acceptable percentage of rejected epochs (default=50.0)
        If more epochs would be rejected, raises an error
        
    verbose : bool, optional
        Whether to plot before/after comparisons (default=False)
    
    Returns
    -------
    cleaned_epochs_AR : List[mne.Epochs]
        List of Epochs objects after Autoreject cleaning
        
    dic_AR : DicAR
        Dictionary with information about the rejection process:
        - 'strategy': Strategy used ('union' or 'intersection')
        - 'threshold': Maximum acceptable percentage of rejected epochs
        - 'S1': Percentage of epochs rejected for participant 1
        - 'S2': Percentage of epochs rejected for participant 2
        - 'dyad': Overall percentage of epochs rejected
    
    Notes
    -----
    This function uses the autoreject package to automatically identify and handle
    problematic epochs. The identification is based on statistical properties of the
    data and is more objective than manual rejection.
    
    With the 'intersection' strategy, the function also equalizes the number of
    epochs between participants after cleaning.
    
    Raises
    ------
    RuntimeError
        If the percentage of rejected epochs exceeds the specified threshold
        
    Examples
    --------
    >>> # Apply Autoreject with default settings
    >>> cleaned_epochs, rejection_info = AR_local(ica_cleaned_epochs)
    
    >>> # Use 'intersection' strategy with lower threshold
    >>> cleaned_epochs, rejection_info = AR_local(
    ...     ica_cleaned_epochs, strategy='intersection', threshold=30.0, verbose=True
    ... )
    >>> print(f"Rejected {rejection_info['dyad']}% of epochs")
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
