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


from typing import Tuple, List
import math
import random
import string

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from skimage.measure import block_reduce
import mne
from mne.io.constants import FIFF
from mne import create_info, EpochsArray


def create_epochs(raw_S1: mne.io.Raw, raw_S2: mne.io.Raw, duration: float) -> Tuple[mne.Epochs, mne.Epochs]:
    """
    Create epochs from continuous raw EEG data for two participants.
    
    This function segments continuous EEG data into fixed-length epochs
    for both participants, interpolates bad channels, and prepares the data
    for further analysis.
    
    Parameters
    ----------
    raw_S1 : mne.io.Raw
        Raw EEG data for participant 1
        
    raw_S2 : mne.io.Raw
        Raw EEG data for participant 2
        
    duration : float
        Duration of each epoch in seconds
    
    Returns
    -------
    tuple : (List[mne.Epochs], List[mne.Epochs])
        A tuple containing two lists of Epochs objects, one for each participant
    
    Notes
    -----
    The function performs the following steps:
    1. Creates fixed-length events at regular intervals
    2. Segments the continuous data into epochs based on these events
    3. Interpolates bad channels if present
    4. Removes bad channel labels after interpolation
    
    The returned epochs have no baseline correction applied (baseline=None),
    and no automatic rejection criteria (reject=None).
    
    Examples
    --------
    >>> # Create 2-second epochs from raw data
    >>> epochs_S1, epochs_S2 = create_epochs(raw_S1, raw_S2, duration=2.0)
    >>> print(f"Created {len(epochs_S1[0])} epochs for participant 1")
    >>> print(f"Created {len(epochs_S2[0])} epochs for participant 2")
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
    Merge epochs from two participants into a single hyperscanning dataset.
    
    This function combines the EEG data from two participants into a single
    Epochs object, where channels from each participant are labeled with
    suffixes "_S1" and "_S2" respectively.
    
    Parameters
    ----------
    epoch_S1 : mne.Epochs
        Epochs object for participant 1
        
    epoch_S2 : mne.Epochs
        Epochs object for participant 2
    
    Returns
    -------
    ep_hyper : mne.Epochs
        Merged Epochs object containing data from both participants
    
    Notes
    -----
    Prior to merging, any bad channels are interpolated and their labels
    are removed from the 'bads' list.
    
    The function assumes that the time alignment between participants
    has already been performed at the raw data creation stage.
    
    After merging, average referencing cannot be applied to the data,
    and standard topographic plotting is not possible. Use specialized
    hyperscanning visualization tools instead.
    
    The function preserves channel types (EEG/EOG) from the original data.
    
    Examples
    --------
    >>> # Merge epochs from two participants
    >>> epochs_merged = merge(epochs_S1, epochs_S2)
    >>> print(f"Original channels: {len(epochs_S1.ch_names)}")
    >>> print(f"Merged channels: {len(epochs_merged.ch_names)}")
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

    # Verify both epochs have same filter settings
    if (epoch_S1.info['highpass'] != epoch_S2.info['highpass'] or 
        epoch_S1.info['lowpass'] != epoch_S2.info['lowpass']):
        import warnings
        warnings.warn("Filter settings differ between participants. Using S1 settings.")
    
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

    # Create info object with filter information preserved
    info = mne.create_info(ch_names_merged, sfreq, ch_types='eeg', verbose=None)

    # Also preserve other relevant metadata
    if 'description' in epoch_S1.info:
        info['description'] = epoch_S1.info['description']
    elif 'description' in epoch_S2.info:
        info['description'] = epoch_S2.info['description']

    ep_hyper = mne.EpochsArray(merged, info)

    # Preserve filter information from source epochs
    with ep_hyper.info._unlock():
        ep_hyper.info['highpass'] = epoch_S1.info['highpass']
        ep_hyper.info['lowpass'] = epoch_S1.info['lowpass']
    
    # setting channels type
    EOG_ch = []
    for ch in epoch_S1.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EOG_CH:
            EOG_ch.append(ch['ch_name'])

    for ch in ep_hyper.info['chs']:
        if ch['ch_name'].split('_')[0] in EOG_ch:
            ch['kind'] = FIFF.FIFFV_EOG_CH
        else:
            ch['kind'] = FIFF.FIFFV_EEG_CH

    return ep_hyper


def split(raw_merge: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """
    Split merged raw data back into separate datasets for each participant.
    
    This function reverses the merging process, extracting individual participants'
    data from a merged hyperscanning dataset based on channel name suffixes.
    
    Parameters
    ----------
    raw_merge : mne.io.Raw
        Merged Raw data for both participants, with channels having
        suffixes "_S1" and "_S2" (or "_1" and "_2")
    
    Returns
    -------
    tuple : (mne.io.Raw, mne.io.Raw)
        A tuple containing two Raw objects, one for each participant,
        with standard 10-20 montage applied
    
    Notes
    -----
    The function performs the following steps:
    1. Separates channels based on their suffix
    2. Creates new Raw objects for each participant
    3. Applies the standard 10-20 montage to both datasets
    4. Sets the EEG reference to the average of all channels
    
    Channel types (EEG/EOG) are preserved from the original data.
    Bad channel labels are transferred to the appropriate participant.
    
    Examples
    --------
    >>> # Split previously merged raw data
    >>> raw_S1, raw_S2 = split(raw_merged)
    >>> print(f"Channels for participant 1: {len(raw_S1.ch_names)}")
    >>> print(f"Channels for participant 2: {len(raw_S2.ch_names)}")
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
    Concatenate multiple epochs objects into single epochs objects for each participant.
    
    This function combines multiple epoch instances (e.g., from different
    experimental blocks or sessions) into a single epochs object for each participant.
    
    Parameters
    ----------
    epoch_S1 : list
        List of Epochs objects for participant 1
        
    epoch_S2 : list
        List of Epochs objects for participant 2
    
    Returns
    -------
    tuple : (mne.Epochs, mne.Epochs)
        A tuple containing two concatenated Epochs objects, one for each participant
    
    Notes
    -----
    This function is useful when you have recorded multiple experimental blocks
    or conditions and want to combine them for analysis.
    
    The epochs to be concatenated must be compatible in terms of sampling rate,
    channel names, and other attributes.
    
    Examples
    --------
    >>> # Concatenate epochs from two experimental blocks
    >>> epochs_S1_all, epochs_S2_all = concatenate_epochs(
    ...     [epochs_S1_block1, epochs_S1_block2],
    ...     [epochs_S2_block1, epochs_S2_block2]
    ... )
    >>> print(f"Total epochs: {len(epochs_S1_all)}")
    """

    epoch_S1_concat = mne.concatenate_epochs(epoch_S1)
    epoch_S2_concat = mne.concatenate_epochs(epoch_S2)

    return epoch_S1_concat, epoch_S2_concat


def normalizing(baseline: np.ndarray, task: np.ndarray, type: str) -> np.ndarray:
    """
    Normalize data (PSD or CSD values) relative to a baseline condition.
    
    This function computes Z-scores or log-ratios between task and baseline
    conditions, which is useful for comparing spectral power or connectivity
    changes relative to a reference state.
    
    Parameters
    ----------
    baseline : np.ndarray
        Baseline condition data with shape (n_epochs, n_channels, n_frequencies)
        
    task : np.ndarray
        Task condition data with shape (n_epochs, n_channels, n_frequencies)
        
    type : str
        Normalization method to use:
        - 'Zscore': (task - baseline_mean) / baseline_std
        - 'Logratio': log10(task / baseline_mean)
    
    Returns
    -------
    Normed_task : np.ndarray
        Normalized data with shape (n_channels, n_frequencies)
    
    Notes
    -----
    For 'Logratio' normalization, only positive values can be used as input.
    If your data contains negative values, consider taking the absolute value
    before normalization.
    
    Both normalization methods average across epochs before computing the
    normalization, resulting in a single normalized value per channel and frequency.
    
    Examples
    --------
    >>> # Normalize alpha power using Z-scores
    >>> alpha_power_baseline = baseline_psd[:, :, 8:13]  # Alpha band
    >>> alpha_power_task = task_psd[:, :, 8:13]  # Alpha band
    >>> normalized_alpha = normalizing(
    ...     alpha_power_baseline, alpha_power_task, type='Zscore'
    ... )
    >>> print(f"Shape of normalized data: {normalized_alpha.shape}")
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

def generate_random_epoch(epoch: mne.Epochs, mu: float=0, sigma: float=2.0) -> mne.Epochs:
    """
    Generate epochs with random data following a normal distribution.
    
    This function creates a new Epochs object with the same structure as
    the input epochs, but with random data values drawn from a normal distribution.
    
    Parameters
    ----------
    epoch : mne.Epochs
        Template Epochs object to copy structure from
        
    mu : float, optional
        Mean of the normal distribution (default=0)
        
    sigma : float, optional
        Standard deviation of the normal distribution (default=2.0)
    
    Returns
    -------
    random_epochs : mne.Epochs
        New Epochs object with random data values
    
    Notes
    -----
    This function is useful for:
    - Creating null data for testing analysis pipelines
    - Generating surrogate data for statistical tests
    - Simulating baseline noise with known properties
    
    The random data has the same dimensions as the input epoch data:
    (n_epochs, n_channels, n_times)
    
    Examples
    --------
    >>> # Generate random epochs with the same structure as real data
    >>> random_epochs = generate_random_epoch(real_epochs, mu=0, sigma=1.0)
    >>> # Compare real and random data
    >>> real_mean = np.mean(real_epochs.get_data())
    >>> random_mean = np.mean(random_epochs.get_data())
    >>> print(f"Real data mean: {real_mean}, Random data mean: {random_mean}")
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


def generate_virtual_epoch(epoch: mne.Epochs, W: np.ndarray, frequency_mean: float=10, 
                          frequency_std: float=0.2, noise_phase_level: float=0.005, 
                          noise_amplitude_level: float=0.1) -> mne.Epochs:
    """
    Generate epochs with simulated data using Kuramoto oscillators.
    
    This function creates a new Epochs object with simulated EEG data
    based on a network of coupled Kuramoto oscillators, which can be
    used to model brain oscillations and synchronization.
    
    Parameters
    ----------
    epoch : mne.Epochs
        Template Epochs object to copy structure from
        
    W : np.ndarray
        Coupling matrix between oscillators, with shape (n_channels, n_channels)
        
    frequency_mean : float, optional
        Mean frequency of oscillators in Hz (default=10)
        
    frequency_std : float, optional
        Standard deviation of oscillator frequencies in Hz (default=0.2)
        
    noise_phase_level : float, optional
        Amount of noise added to the phase (default=0.005)
        
    noise_amplitude_level : float, optional
        Amount of noise added to the amplitude (default=0.1)
    
    Returns
    -------
    simulated_epochs : mne.Epochs
        New Epochs object with simulated data values
    
    Notes
    -----
    The Kuramoto model is a mathematical model used to describe synchronization
    in a network of coupled oscillators. Each oscillator has its own natural
    frequency drawn from a normal distribution with mean `frequency_mean` and
    standard deviation `frequency_std`.
    
    The coupling between oscillators is defined by the matrix W, where W[i,j]
    represents the strength of the connection from oscillator j to oscillator i.
    
    The simulation uses the scipy.integrate.solve_ivp function to solve the
    differential equations of the Kuramoto model.
    
    This function is useful for:
    - Testing connectivity measures with known ground truth
    - Simulating synchronization phenomena
    - Generating data with controlled properties for method validation
    
    Examples
    --------
    >>> # Create a simple coupling matrix (3 oscillators)
    >>> W = np.array([
    ...     [0, 0.2, 0],
    ...     [0.2, 0, 0.2],
    ...     [0, 0.2, 0]
    ... ])
    >>> # Generate simulated epochs in the alpha band
    >>> sim_epochs = generate_virtual_epoch(
    ...     real_epochs, W, frequency_mean=10, frequency_std=0.1
    ... )
    >>> # Analyze the simulated data
    >>> from mne.time_frequency import psd_welch
    >>> psds, freqs = psd_welch(sim_epochs, fmin=8, fmax=12)
    >>> print(f"Peak frequency: {freqs[np.argmax(np.mean(psds, axis=0))]}")
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

class Task:
    name: str
    onset_event_id: int | None
    onset_time: float | None

    offset_event_id: int | None
    duration: float | None
    
    def __init__(
        self,
        name:str,
        onset_event_id:int|None=None,
        onset_time:float|None=None,
        offset_event_id:int|None=None,
        duration:float|None=None,
    ):
        self.name = name
        self.onset_event_id = None
        self.onset_time = None
        self.offset_event_id = None
        self.duration = None

        if onset_event_id is not None:
            self.onset_event_id = onset_event_id

        if onset_time is not None:
            self.onset_time = onset_time

        if offset_event_id is not None:
            self.offset_event_id = offset_event_id

        if duration is not None:
            self.duration = duration
        
        # Need one start
        if self.onset_event_id is None and self.onset_time is None:
            raise RuntimeError('Must set either onset_event_id or onset_time')

        if self.onset_event_id is not None and self.onset_time is not None:
            raise RuntimeError('Cannot set both onset_event_id and onset_time')

        # Need one end
        if self.offset_event_id is None and self.duration is None:
            raise RuntimeError('Must set either offset_event_id or duration')

        if self.offset_event_id is not None and self.duration is not None:
            raise RuntimeError('Cannot set both offset_event_id and duration')

        # invalid
        if self.onset_time is not None and self.offset_event_id is not None:
            raise RuntimeError('Cannot use offset_event_id with onset_time')

        #if self.onset_event_id is not None and self.onset

    @property
    def is_event_based(self):
        return self.onset_event_id is not None

    @property
    def is_time_based(self):
        return self.onset_time is not None

# typing
TaskTuple = tuple[str, int, int|None]
TaskList = list[Task]

# Constants for task description
TASK_NEXT_EVENT = -2
TASK_BEGINNING = -1
TASK_END = -1
TASK_NAME_WHOLE_RECORD = 'whole_record'

def epochs_from_tasks(raw: mne.io.Raw, tasks: TaskList, verbose: bool = False) -> List[mne.Epochs]:
    events, events_map = mne.events_from_annotations(raw)
    #print(events)

    all_epochs = []
    sfreq = raw.info['sfreq']

    for task in [t for t in tasks if t.is_event_based]:
        task_key = task.name
        onset_event_id = task.onset_event_id
        offset_event_id = task.offset_event_id
        t_starts = []
        t_durations = []
        task_events = []

        if onset_event_id is not None and not isinstance(onset_event_id, int):
            raise ValueError(f'onset_event_id must be an integer. Received {type(onset_event_id)}')

        if offset_event_id is not None and not isinstance(offset_event_id, int):
            raise ValueError(f'offset_event_id must be an integer. Received {type(offset_event_id)}')

        # To handle start of raw as "event" for task
        if onset_event_id == TASK_BEGINNING:
            events_loop = [[0]]
        else:
            events_loop = events[events[:, 2] == onset_event_id]

        for event_start in events_loop:
            t_start = event_start[0]
            if task.duration is not None: # will not use offset_event_id
                t_duration = task.duration
            else:
                # Find end of task
                if offset_event_id == TASK_END:
                    t_end = raw.n_times - 1
                else:
                    where_gt_start = events[:, 0] > t_start
                    if offset_event_id == TASK_NEXT_EVENT:
                        # until next event
                        where = where_gt_start
                    else:
                        where_task_end = events[:, 2] == offset_event_id
                        where = where_gt_start & where_task_end

                    event_end = events[where]
                    if len(event_end) == 0:
                        raise RuntimeError(f'Cannot find end of task "{task_key}" with trigger_id "{offset_event_id}" (event_id "{offset_event_id}")')
                    t_end = event_end[0, 0] # use the first

                t_min = (t_start - raw.first_samp) / sfreq
                t_max = (t_end - raw.first_samp) / sfreq
                t_duration = t_max - t_min

            t_starts.append(t_start)
            t_durations.append(t_duration)
            task_events.append([t_start, 0, onset_event_id])

        all_epochs.append(mne.Epochs(raw,
            task_events,
            event_id={ task_key: onset_event_id},
            tmin=0,
            tmax=min(t_durations),
            baseline=None,
            preload=True,
            event_repeated='merge',
            verbose=verbose,
            ))

    # time based
    events_per_task = dict()
    duration_per_task = dict()
    event_id_per_task = dict()
    next_event_id = 1000 # start our event_id at 1000 for time based

    for i, task in enumerate([t for t in tasks if t.is_time_based]):
        task_name = task.name
        task_start = task.onset_time
        duration = task.duration
        task_end = task_start + duration

        if task_name not in event_id_per_task.keys():
            event_id_per_task[task_name] = next_event_id
            next_event_id += 1
        event_id = event_id_per_task[task_name]

        events = mne.make_fixed_length_events(raw,
                                            id=event_id,
                                            start=task_start,
                                            stop=task_end,
                                            duration=duration,
                                            first_samp=True,
                                            overlap=0.0)

        if not task_name in events_per_task.keys():
            events_per_task[task_name] = events
            duration_per_task[task_name] = duration
        else:
            events_per_task[task_name] = np.vstack([events_per_task[task_name], events])
            duration_per_task[task_name] = min(duration_per_task[task_name], duration)


    for task_name in events_per_task.keys():
        event_id_map = {task_name: event_id_per_task[task_name]}
        epochs = mne.Epochs(raw,
                            events_per_task[task_name],
                            event_id=event_id_map,
                            tmin=0,
                            tmax=duration_per_task[task_name],
                            baseline=None,
                            preload=True,
                            reject=None,
                            proj=True,
                            verbose=verbose)
        all_epochs.append(epochs)

    return all_epochs

def downsample_in_time(times, *args, bins=500):
    ret = []
    # We assume time is always the last column
    factor = math.ceil(times.shape[0] / bins)

    if factor == 1:
        return [times, *args, 1]
    
    # First deal with times. Need to pad (cval) with max value, we don't want to "go back in time" for the last values
    ret.append(block_reduce(times, block_size=factor, func=np.min, cval=np.max(times)))
    
    for item in args:
        if len(item.shape) == 1:
            ret.append(block_reduce(item, block_size=factor, func=np.mean, cval=np.mean(item)))
        elif len(item.shape) == 2:
            ret.append(block_reduce(item, block_size=(1,factor), func=np.mean, cval=np.mean(item)))
        else:
            raise RuntimeError(f'Unsupported number of column for downsampling: {len(item)}')
    
    ret.append(factor)

    return ret
        
def generate_random_label(length:int) -> str:
    """
    Generate a random label of a specific length

    Args:
        length (int): length of the string

    Returns:
        str: a unique label
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
