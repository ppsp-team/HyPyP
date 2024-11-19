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


from enum import Enum
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import mne
from mne.io.constants import FIFF
from mne import create_info, EpochsArray


def create_epochs(raw_S1: mne.io.Raw, raw_S2: mne.io.Raw, duration: float) -> Tuple[mne.Epochs, mne.Epochs]:
    """
    Creates Epochs from Raws and vizualize Power Spectral Density (PSD)
    on average Epochs (option).

    Arguments:
        raw_S1: list of Raws for participant 1 (with the different
            experimental realizations of a condition - for example
            the baseline. The length can be 1).
        raw_S2: list of Raws for participant 2.  
            Raws are MNE objects: data are ndarray with shape
            (n_channels, n_times) and information is a dictionnary
            sampling parameters.

    Note:
        Plots topomaps of PSD values calculated with welch FFT
        for each epoch and each participant, averaged in each
        frequency band-of-interest.
        # TODO: option with verbose

    Returns:
        epoch_S1, epoch_S2: list of Epochs for each participant.
    """
    epoch_S1: List[mne.Epochs] = []
    epoch_S2: List[mne.Epochs] = []

    for raw1, raw2 in zip(raw_S1, raw_S2):
        # creating fixed events
        fixed_events1 = mne.make_fixed_length_events(raw1,
                                                     id=1,
                                                     start=0,
                                                     stop=None,
                                                     duration=duration,
                                                     first_samp=True,
                                                     overlap=0.0)
        fixed_events2 = mne.make_fixed_length_events(raw2,
                                                     id=1,
                                                     start=0,
                                                     stop=None,
                                                     duration=duration,
                                                     first_samp=True,
                                                     overlap=0.0)

        # epoching the events per time window
        epoch1 = mne.Epochs(raw1, fixed_events1, event_id=1, tmin=0, tmax=duration,
                            baseline=None, preload=True, reject=None, proj=True)
        # reject=reject_criteria, no baseline correction
        # preload needed after
        epoch2 = mne.Epochs(raw2, fixed_events2, event_id=1, tmin=0, tmax=duration,
                            baseline=None, preload=True, reject=None, proj=True)

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
    Merges Epochs from 2 participants after interpolation of bad channels.

    Arguments:
        epoch_S1: Epochs object for participant 1.
        epoch_S2: Epochs object for participant 2.  
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
            participants). The time alignement has been done at raw data creation.
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

    # creating channels label for each participant
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
        data_S1 = epoch_S1[l].get_data(copy=True)
        data_S2 = epoch_S2[l].get_data(copy=True)

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
                           verbose=None)
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
    Splits merged Raw data into 2 participants Raw data.

    Arguments:
        raw_merge: Raw data for the dyad with data from participant 1
            and data from participant 2 (channels name are defined with
            the suffix S1 and S2 respectively).

    Note:
        Participant's Raw data is set to the standard montage 1020
        available in MNE. An average is computed to avoid reference bias
        (see MNE documentation about set_eeg_reference).

    Returns:
        raw_1020_S1, raw_1020_S2: Raw data for each participant separately.
            Raws are MNE objects.
    """
    ch_S1 = []
    ch_S2 = []
    ch = []
    for name in raw_merge.info['ch_names']:
        if name.endswith('S1') or name.endswith('_1'):
            ch_S1.append(name)
            ch.append(name.split('_')[0])
        elif name.endswith('S2') or name.endswith('_2'):
            ch_S2.append(name)

    # picking individual participant data
    data_S1 = raw_merge.get_data(picks=ch_S1)
    data_S2 = raw_merge.get_data(picks=ch_S2)

    # creating info for raws
    info = mne.create_info(ch, raw_merge.info['sfreq'], ch_types='eeg', verbose=None)

    raw_S1 = mne.io.RawArray(data_S1, info)
    raw_S2 = mne.io.RawArray(data_S2, info)

    # setting info about channels and task
    raw_S1.info['bads'] = [
        ch.split('_')[0] for ch in ch_S1 if ch in raw_merge.info['bads']]
    raw_S2.info['bads'] = [
        ch.split('_')[0] for ch in ch_S2 if ch in raw_merge.info['bads']]
    for raws in (raw_S1, raw_S2):
        raws.info['description'] = raw_merge.info['description']

    # setting montage 94 electrodes (ignore somes to correspond to our data)
        for ch in raws.info['chs']:
            if ch['ch_name'].startswith('MOh') or ch['ch_name'].startswith('MOb') or ('EOG' in ch['ch_name']):
                # print('emg')
                ch['kind'] = FIFF.FIFFV_EOG_CH
            else:
                ch['kind'] = FIFF.FIFFV_EEG_CH
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_1020_S1 = raw_S1.copy().set_montage(montage, on_missing='ignore')
    raw_1020_S2 = raw_S2.copy().set_montage(montage, on_missing='ignore')

    # set reference to electrodes average
    # (instate of initial ref to avoid ref biais)
    # and storing it in raw.info['projs']: applied when Epochs
    raw_1020_S1, _ = mne.set_eeg_reference(raw_1020_S1, 'average', projection=True)
    raw_1020_S2, _ = mne.set_eeg_reference(raw_1020_S2, 'average', projection=True)

    return raw_1020_S1, raw_1020_S2


def concatenate_epochs(epoch_S1: mne.Epochs, epoch_S2: mne.Epochs) -> Tuple[mne.Epochs, mne.Epochs]:
    """
    Concatenates a list of Epochs in one Epochs object.

    Arguments:
        epoch_S1: list of Epochs for participant 1 (for example the
            list samples different experimental realizations
            of the baseline condition).
        epoch_S2: list of Epochs for participant 2.  
            Epochs are MNE objects.

    Returns:
        epoch_S1_concat, epoch_S2_concat: list of concatenate Epochs
            (for example one epoch with all the experimental realizations
            of the baseline condition) for each participant.
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

    Note:
        If normalization's type is 'Logratio', only positive values
        can be used as input (if it is not the case, take the absolute
        value).

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

def generate_random_epoch(epoch: mne.Epochs, mu: float=0, sigma: float=2.0)-> mne.Epochs:
    """
    Generate epochs with random data. 

    Arguments:
        epoch: mne.Epochs
            Epochs object to get epoch info structure
        mu: float
            Mean of the normal distribution
        sigma: float
            Standart deviation of the normal distribution

    Returns:
        mne.Epochs
            new epoch with random data with normal distribution
    """

    # Get epoch information 
    info = epoch.info #create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Get epochs as a 3D NumPy array of shape (n_epochs, n_channels, n_times)
    # Get the arrays’ shape
    data_shape = epoch.get_data(copy=False).shape
    i, j, k = data_shape

    # Generate a numpy.array with same shape from the normal distribution
    r_epoch = sigma * np.random.randn(i, j, k) + mu

    return EpochsArray(data=r_epoch, info=info)


def generate_virtual_epoch(epoch: mne.Epochs, W: np.ndarray, frequency_mean: float=10, frequency_std: float=0.2,
                           noise_phase_level: float=0.005, noise_amplitude_level: float=0.1)-> mne.Epochs:
    """
    Generate epochs with simulated data using Kuramoto oscillators. 

    Arguments:
        epoch: mne.Epochs
            Epochs object to get epoch info structure
        W: np.ndarray
            Coupling matrix between the oscillators
        frequency_mean: float
            Mean of the normal distribution for oscillators frequency
        frequency_std: float
            Standart deviation of the normal distribution for oscillators frequency
        noise_phase_level: float
            Amount of noise at the phase level
        noise_amplitude_level: float
            Amount of noise at the amplitude level

    Returns:
        mne.Epochs
            new epoch with simulated data
    """

    n_epo, n_chan, n_samp = epoch.get_data().shape
    sfreq = epoch.info['sfreq']

    Nt = n_samp * n_epo
    tmax = n_samp / sfreq * n_epo  # s
    tv = np.linspace(0., tmax, Nt)

    freq = frequency_mean + frequency_std * np.random.randn(n_chan)
    omega = 2. * np.pi * freq

    def fp(t, p):
        p = np.atleast_2d(p)
        coupling = np.squeeze((np.sin(p) * np.matmul(W, np.cos(p).T).T) - (np.cos(p) * np.matmul(W, np.sin(p).T).T))
        dotp = omega - coupling + noise_phase_level * np.random.randn(n_chan) / n_samp
        return dotp

    p0 = 2 * np.pi * np.block([np.zeros(n_chan/2), np.zeros(n_chan/2) + np.random.rand(n_chan/2) + 0.5])
    ans = solve_ivp(fun=fp, t_span=(tv[0], tv[-1]), y0=p0, t_eval=tv)
    phi = ans['y'].T  % (2*np.pi)

    eeg = np.sin(phi) + noise_amplitude_level * np.random.randn(*phi.shape)
    
    simulation = epoch.copy()
    simulation._data = np.transpose(np.reshape(eeg.T, [n_chan, n_epo, n_samp]), (1, 0, 2))
    
    return simulation

# typing
Task = tuple[str, int, int|None]
TaskList = list[Task]

# Constants for task description
TASK_NEXT_EVENT = None
TASK_BEGINNING = -1
TASK_END = -1

def epochs_from_tasks_time_range(raw: mne.io.Raw, tasks: TaskList) -> List[mne.Epochs]:
    all_epochs = []
    for i, task in enumerate(tasks):
        event_id = i + 1
        duration = task[2] - task[1]
        events = mne.make_fixed_length_events(raw,
                                            id=1,
                                            start=task[1],
                                            stop=task[2],
                                            duration=duration,
                                            first_samp=True,
                                            overlap=0.0)

        event_id_map = {task[0]: 1}
        epochs = mne.Epochs(raw,
                            events,
                            event_id=event_id_map, # TODO this doesn't seem to work. The event_id is always 1
                            tmin=0,
                            tmax=duration,
                            baseline=None,
                            preload=True,
                            reject=None,
                            proj=True)
        all_epochs.append(epochs)
    return all_epochs

def epochs_from_tasks_annotations(raw: mne.io.Raw, tasks: TaskList) -> List[mne.Epochs]:
    events, events_map = mne.events_from_annotations(raw)
    #print(events)

    epochs_per_tasks = []
    sfreq = raw.info['sfreq']

    for task_key, trigger_id_task_start, trigger_id_task_end in tasks:
        t_starts = []
        t_durations = []
        task_events = []

        # To handle start of raw as "event" for task
        if trigger_id_task_start == TASK_BEGINNING:
            events_loop = [[0]]
        else:
            events_loop = events[events[:, 2] == events_map[str(trigger_id_task_start)]]

        for event_start in events_loop:
            t_start = event_start[0]

            # Find end of task
            if trigger_id_task_end == TASK_END:
                t_end = raw.n_times - 1
            else:
                where_gt_start = events[:, 0] > t_start
                if trigger_id_task_end == TASK_NEXT_EVENT:
                    # until next event
                    where = where_gt_start
                else:
                    where_task_end = events[:, 2] == events_map[str(trigger_id_task_end)]
                    where = where_gt_start & where_task_end

                event_end = events[where]
                if len(event_end) == 0:
                    raise RuntimeError(f'Cannot find end of task "{task_key}" with trigger_id "{trigger_id_task_end}" (event_id "{events_map[str(trigger_id_task_end)]}")')
                t_end = event_end[0, 0] # use the first

            t_min = (t_start - raw.first_samp) / sfreq
            t_max = (t_end - raw.first_samp) / sfreq
            t_duration = t_max - t_min

            t_starts.append(t_start)
            t_durations.append(t_duration)
            task_events.append([t_start, 0, trigger_id_task_start])

        epochs_per_tasks.append(mne.Epochs(raw,
            task_events,
            event_id={ task_key: trigger_id_task_start},
            tmin=0,
            tmax=min(t_durations),
            baseline=None,
            preload=True,
            event_repeated='merge',
            ))

    return epochs_per_tasks