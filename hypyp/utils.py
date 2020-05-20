#!/usr/bin/env python
# coding=utf-8

"""
Useful tools

| Option | Description |
| ------ | ----------- |
| title           | utils.py |
| authors         | Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


import numpy as np
import pandas as pd
import mne
from mne.io.constants import FIFF


def create_epochs(raw_S1: mne.io.Raw, raw_S2: mne.io.Raw, freq_bands: list) -> list:
    """
    Creates Epochs from Raws and vizualize Power Spectral Density (PSD)
    on average Epochs (option).

    Arguments:
        raw_S1: list of Raws for subject 1 (with the different
          experimental realizations of a condition - for example
          the baseline. The length can be 1).
        raw_S2: list of Raws for subject 2.  
          Raws are MNE objects: data are ndarray with shape
          (n_channels, n_times) and information is a dictionnary
          sampling parameters.
        freq_bands: frequency bands-of-interest, list of tuple.

    Note:
        Plots topomaps of PSD values calculated with welch FFT
        for each epoch and each subject, averaged in each
        frequency band-of-interest.
        # TODO: option with verbose

    Returns:
        epoch_S1, epoch_S2: list of Epochs for each subject.
    """
    epoch_S1 = []
    epoch_S2 = []

    for raw1, raw2 in zip(raw_S1, raw_S2):
        # creating fixed events
        fixed_events1 = mne.make_fixed_length_events(raw1,
                                                     id=1,
                                                     start=0,
                                                     stop=None,
                                                     duration=1.0,
                                                     first_samp=True,
                                                     overlap=0.0)
        fixed_events2 = mne.make_fixed_length_events(raw2,
                                                     id=1,
                                                     start=0,
                                                     stop=None,
                                                     duration=1.0,
                                                     first_samp=True,
                                                     overlap=0.0)

        # epoching the events per time window
        epoch1 = mne.Epochs(raw1, fixed_events1, event_id=1, tmin=0, tmax=1,
                            baseline=None, preload=True, reject=None, proj=True)
        # reject=reject_criteria, no baseline correction
        # preload needed after
        epoch2 = mne.Epochs(raw2, fixed_events2, event_id=1, tmin=0, tmax=1,
                            baseline=None, preload=True, reject=None, proj=True)

        # vizu topoplots of PSD for epochs
        # epoch1.plot()
        # epoch1.plot_psd_topomap(bands=freq_bands)  # welch FFT
        # epoch1.plot_psd_topomap(bands=freq_bands)  # welch FFT

        # interpolating bad channels and removing the label
        if len(epoch1.info['bads']) > 0:
            epoch1 = mne.Epochs.interpolate_bads(epoch1,
                                                 reset_bads=True,
                                                 mode='accurate',
                                                 origin='auto',
                                                 verbose=None)
        if len(epoch2.info['bads']) > 0:
            epoch2 = mne.Epochs.interpolate_bads(epoch2,
                                                 reset_bads=True,
                                                 mode='accurate',
                                                 origin='auto',
                                                 verbose=None)

        epoch_S1.append(epoch1)
        epoch_S2.append(epoch2)

    return epoch_S1, epoch_S2


def merge(epoch_S1: mne.Epochs, epoch_S2: mne.Epochs) -> mne.Epochs:
    """
    Merges Epochs from 2 subjects after interpolation of bad channels.

    Arguments:
        epoch_S1: Epochs object for subject 1.
        epoch_S2: Epochs object for subject 2.  
          epoch_S1 and epoch_S2 correspond to a condition and can result
          from the concatenation of epochs from different experimental
          realizations of a condition.  
          Epochs are MNE objects: data are stored in an array of shape
          (n_epochs, n_channels, n_times) and parameters information
          is stored in a disctionnary.

    Note:
        Bad channels labelling is removed.
        Note that average on reference can not be done anymore. Similarly,
        montage can not be set to the data and as a result topographies in MNE
        are not possible anymore. Use toolbox vizualisations instead.

    Returns:
        ep_hyper: Epochs object for the dyad (with merged data of the two
          subjects). The time alignement has been done at raw data creation.
    """
    # checking bad ch for epochs, interpolating
    # and removing them from 'bads' if needed
    if len(epoch_S1.info['bads']) > 0:
        epoch_S1 = mne.Epochs.interpolate_bads(epoch_S1,
                                               reset_bads=True,
                                               mode='accurate',
                                               origin='auto',
                                               verbose=None)
        # head-digitization-based origin fit
    if len(epoch_S2.info['bads']) > 0:
        epoch_S2 = mne.Epochs.interpolate_bads(epoch_S2,
                                               reset_bads=True,
                                               mode='accurate',
                                               origin='auto',
                                               verbose=None)

    sfreq = epoch_S1[0].info['sfreq']
    ch_names = epoch_S1[0].info['ch_names']

    # creating channels label for each subject
    ch_names1 = []
    for i in ch_names:
        ch_names1.append(i+'_S1')
    ch_names2 = []
    for i in ch_names:
        ch_names2.append(i+'_S2')

    merges = []

    # checking wether data have the same size
    assert(len(epoch_S1) == len(epoch_S2)
           ), "Epochs from S1 and S2 should have the same size!"

    # picking data per epoch
    for l in range(0, len(epoch_S1)):
        data_S1 = epoch_S1[l].get_data()
        data_S2 = epoch_S2[l].get_data()

        data_S1 = np.squeeze(data_S1, axis=0)
        data_S2 = np.squeeze(data_S2, axis=0)

        dicdata1 = {i: data_S1[:, i] for i in range(0, len(data_S1[0, :]))}
        dicdata2 = {i: data_S2[:, i] for i in range(0, len(data_S2[0, :]))}

        # creating dataframe to merge data for each time point
        dataframe1 = pd.DataFrame(dicdata1, index=ch_names1)
        dataframe2 = pd.DataFrame(dicdata2, index=ch_names2)
        merge = pd.concat([dataframe1, dataframe2])

        # reconverting to array and joining the info file
        merge_arr = merge.to_numpy()
        merges.append(merge_arr)

    merged = np.array(merges)
    ch_names_merged = ch_names1+ch_names2
    info = mne.create_info(ch_names_merged, sfreq, ch_types='eeg',
                           montage=None, verbose=None)
    ep_hyper = mne.EpochsArray(merged, info)

    # setting channels type
    EOG_ch = []
    for ch in epoch_S1.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EOG_CH:
            EOG_ch.append(ch['ch_name'])

    for ch in ep_hyper.info['chs']:
        if ch['ch_name'].split('_')[0] in EOG_ch:
            # print('emg')
            ch['kind'] = FIFF.FIFFV_EOG_CH
        else:
            ch['kind'] = FIFF.FIFFV_EEG_CH

    # info about task
    ep_hyper.info['description'] = epoch_S1[0].info['description']

    return ep_hyper


def split(raw_merge: mne.io.Raw) -> mne.io.Raw:
    """
    Splits merged Raw data into 2 subjects Raw data.

    Arguments:
        raw_merge: Raw data for the dyad with data from subject 1
          and data from subject 2 (channels name are defined with
          the suffix S1 and S2 respectively).

    Note:
        Subject's Raw data is set to the standard montage 1020
        available in MNE. An average is computed to avoid reference bias
        (see MNE documentation about set_eeg_reference).

    Returns:
        raw_1020_S1, raw_1020_S2: Raw data for each subject separately.
          Raws are MNE objects.
    """
    ch_S1 = []
    ch_S2 = []
    ch = []
    for name in raw_merge.info['ch_names']:
        if name.endswith('S1'):
            ch_S1.append(name)
            ch.append(name.split('_')[0])
        elif name.endswith('S2'):
            ch_S2.append(name)

    # picking individual subject data
    data_S1 = raw_merge.get_data(picks=ch_S1)
    data_S2 = raw_merge.get_data(picks=ch_S2)

    # creating info for raws
    info = mne.create_info(ch, raw_merge.info['sfreq'], ch_types='eeg',
                           montage=None, verbose=None)
    raw_S1 = mne.io.RawArray(data_S1, info)
    raw_S2 = mne.io.RawArray(data_S2, info)

    # setting info about channels and task
    raw_S1.info['bads'] = [
        ch.split('_')[0] for ch in ch_S1 if ch in raw_merge.info['bads']]
    raw_S2.info['bads'] = [
        ch.split('_')[0] for ch in ch_S2 if ch in raw_merge.info['bads']]
    for raws in (raw_S1, raw_S2):
        raws.info['description'] = raw_merge.info['description']
        raws.info['events'] = raw_merge.info['events']

    # setting montage 94 electrodes (ignore somes to correspond to our data)
        for ch in raws.info['chs']:
            if ch['ch_name'].startswith('MOh') or ch['ch_name'].startswith('MOb'):
                # print('emg')
                ch['kind'] = FIFF.FIFFV_EOG_CH
            else:
                ch['kind'] = FIFF.FIFFV_EEG_CH
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_1020_S1 = raw_S1.copy().set_montage(montage)
    raw_1020_S2 = raw_S2.copy().set_montage(montage)
    # raw_1020_S1.plot_sensors()

    # set reference to electrodes average
    # (instate of initial ref to avoid ref biais)
    # and storing it in raw.info['projs']: applied when Epochs
    raw_1020_S1, _ = mne.set_eeg_reference(
        raw_1020_S1, 'average', projection=True)
    raw_1020_S2, _ = mne.set_eeg_reference(
        raw_1020_S2, 'average', projection=True)

    # TODO: annotations, subj name, events
    # task description different across subj

    # raw_1020_S1.plot()
    # raw_1020_S1.plot_psd()

    return raw_1020_S1, raw_1020_S2


def concatenate_epochs(epoch_S1: mne.Epochs, epoch_S2: mne.Epochs) -> mne.Epochs:
    """
    Concatenates a list of Epochs in one Epochs object.

    Arguments:
        epoch_S1: list of Epochs for subject 1 (for example the
          list samples different experimental realizations
          of the baseline condition).
        epoch_S2: list of Epochs for subject 2.  
          Epochs are MNE objects.

    Returns:
        epoch_S1_concat, epoch_S2_concat: list of concatenate Epochs
          (for example one epoch with all the experimental realizations
          of the baseline condition) for each subject.
    """
    epoch_S1_concat = mne.concatenate_epochs(epoch_S1)
    epoch_S2_concat = mne.concatenate_epochs(epoch_S2)

    return epoch_S1_concat, epoch_S2_concat


def normalizing(baseline: np.ndarray, task: np.ndarray, type: str) -> np.ndarray:
    """
    Computes Zscore or Logratio of a value between a 'task' condition and
    a baseline for example.

    Arguments:
        baseline: PSD or CSD values for the 'baseline',
          ndarray, shape (n_epochs, n_channels, n_frequencies).
        task: PSD or CSD values for the 'task' conditions,
          ndarray, shape (n_epochs, n_channels, n_frequencies).
        type: normalization choice, str 'Zscore' or 'Logratio'.

    Returns:
        Normed_task: PSD or CSD values for the condition 'task' normed by
          values in a baseline and average across epochs, ndarray, shape
          (n_channels, n_frequencies).
    """
    m_baseline = np.mean(baseline, axis=0)
    m_task = np.mean(task, axis=0)
    std_baseline = np.std(baseline, axis=0)
    # normalizing power during task by baseline average power across events
    if type == 'Zscore':
        s = np.subtract(m_task, m_baseline)
        Normed_task = np.divide(s, std_baseline)
    if type == 'Logratio':
        d = np.divide(m_task, m_baseline)
        Normed_task = np.log10(d)

    return Normed_task
