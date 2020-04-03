#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : analyses.py
# description     : intra- and inter-brain measures functions
# author          : Phoebe Chen, Florence Brun, Guillaume Dumas
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

import copy
import numpy as np
import scipy.signal as signal
from astropy.stats import circcorrcoef
import mne
from mne.time_frequency import psd_welch
from mne.io.constants import FIFF


def PSD(epochs_baseline, epochs_task, fmin, fmax):
    """
    Compute the Power Spectral Density (PSD) on Epochs for a condition and
    normalize by the PSD of the baseline.

    Parameters
    -----
    epochs_baseline, epochs_task : Epochs for the baseline and the condition
    ('task' for example), for a subject. epochs_baseline, epochs_task result
    from the concatenation of epochs from different occurences of the condition
    across experiments. Epochs are MNE objects (data are stored in arrays of
    shape (n_epochs, n_channels, n_times) and info are into a dictionnary).

    Note that the function can be iterated on the group and/or on conditions:
    for epochs_baseline, epochs_task in zip(
        epochs['epochs_%s_%s_%s_baseline' % (subj, group, cond_name)],
        epochs['epochs_%s_%s_%s_task' % (subj, group, cond_name)]).

    You can then visualize PSD distribution on the group with the toolbox
    vizualisation to check normality for statistics for example.

    fmin, fmax : minimum and maximum frequencies for PSD (in Hz).

    Returns
    -----
    freqs_mean : list of frequencies in frequency-band-of-interest used by MNE
    for power spectral density calculation.

    m_baseline, psds_welch_task_m : ndarray
    PSD average across epochs for each channel and each frequency,
    for the baseline and the 'task' condition respectively.

    psd_mean_task_normZ, psd_mean_task_normLog : ndarray
    Zscore and Logratio of the average PSD during 'task' condition
    """
    # dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs_baseline.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs_baseline.drop_channels([ch['ch_name']])
    for ch in epochs_task.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs_task.drop_channels([ch['ch_name']])

    # computing power spectral density on epochs signal
    # average in the 1second window around event (mean but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=1)
    psds_welch_baseline, freqs_mean = psd_welch(
        epochs_baseline, **kwargs, average='mean', picks='all')  # or median
    psds_welch_task, freqs_mean = psd_welch(
        epochs_task, **kwargs, average='mean', picks='all')  # or median

    # averaging power across epochs for each ch and each f
    m_baseline = np.mean(psds_welch_baseline, axis=0)
    std_baseline = np.std(psds_welch_baseline, axis=0)
    psds_welch_task_m = np.mean(psds_welch_task, axis=0)

    # normalizing power during task by baseline average power across events
    # Z score
    s = np.subtract(psds_welch_task_m, m_baseline)
    psd_mean_task_normZ = np.divide(s, std_baseline)
    # Log ratio
    d = np.divide(psds_welch_task_m, m_baseline)
    psd_mean_task_normLog = np.log10(d)

    return freqs_mean, m_baseline, psds_welch_task_m, psd_mean_task_normZ, psd_mean_task_normLog


def indexes_connectivity_intrabrain(epochs):
    """ Compute indexes for connectivity analysis between
    all EEG sensors for one subject.
    To use instead of (n_channels, n_channels) connections.

    Parameters
    -----
    epochs : one subject Epochs object to get channels info.

    Returns
    -----
    electrodes : electrodes pairs for which connectivity
    indices will be computed, list of tuples with channels
    indexes.

    """
    names = copy.deepcopy(epochs.info['ch_names'])
    for ch in epochs.info['chs']:
            if ch['kind'] == FIFF.FIFFV_EOG_CH:
                names.remove(ch['ch_name'])

    n = len(names)
    # n = 64
    bin = 0
    idx = []
    electrodes = []
    for e1 in range(n):
        for e2 in range(n):
            if e2 > e1:
                idx.append(bin)
                electrodes.append((e1, e2))
            bin = bin + 1

    return electrodes


def indexes_connectivity_interbrains(epoch_hyper):
    """ Compute indexes for interbrains connectivity analyses
    between all EEG sensors for 2 subjects (merge data).

    Note that only interbrains connectivity will be computed.

    Parameters
    -----
    epoch_hyper : one dyad Epochs object to get channels info.

    Returns
    -----
    electrodes : electrodes pairs for which connectivity
    indices will be computed, list of tuples with channels
    indexes.

    """
    electrodes = []
    names = copy.deepcopy(epoch_hyper.info['ch_names'])
    for ch in epoch_hyper.info['chs']:
            if ch['kind'] == FIFF.FIFFV_EOG_CH:
                names.remove(ch['ch_name'])

    l = list(range(0, int(len(names)/2)))
    # l = list(range(0,62))
    L = []
    M = len(l)*list(range(len(l), len(l)*2))
    for i in range(0, len(l)):
        for p in range(0, len(l)):
            L.append(l[i])
    for i in range(0, len(L)):
        electrodes.append((L[i], M[i]))

    return electrodes



def simple_corr(data, frequencies, mode, epoch_wise=True, time_resolved=True):
    """Compute frequency- and time-frequency-domain connectivity measures.

    Note that it is computed for all possible electrode pairs between the dyad, but doesn't include intrabrain synchrony

    Parameters
    ----------
    data : array-like, shape is (2, n_epochs, n_channels, n_times)
        The data from which to compute connectivity between two subjects
    frequencies : dict | list
        frequencies of interest to compute connectivity with.
        If a dict, different frequency bands are used.
        e.g. {'alpha':[8,12],'beta':[12,20]}
        If a list, every integer frequency within the range is used.
        e.g. [5,30]
    mode : string
        Connectivity measure to compute.
        'envelope': envelope correlation
        'power': power correlation
        'plv': phase locking value
        'ccorr': circular correlation coefficient
        'coh': coherence
        'imagcoh': imaginary coherence
        'proj': projected power correlation
    epoch_wise : boolean
        whether to compute epoch-to-epoch synchrony. default is True.
        if False, complex values from epochs will be concatenated before computing synchrony
        if True, synchrony is computed from matched epochs
    time_resolved : boolean
        whether to collapse the time course, only effective when epoch_wise==True
        if False, synchrony won't be averaged over epochs, and the time course is maintained.
        if True, synchrony is averaged over epochs.

    Returns
    -------
    result : array
        Computed connectivity measure(s). The shape of each array is either (n_freq, n_epochs, n_channels, n_channels)
        if epoch_wise is True and time_resolved is False, or (n_freq, n_channels, n_channels) in other conditions.
    """
    # Data consists of two lists of np.array (n_epochs, n_channels, epoch_size)
    assert data[0].shape[0] == data[1].shape[0], "Two streams much have the same lengths."

    # compute correlation coefficient for all symmetrical channel pairs
    if type(frequencies) == list:
        values = compute_single_freq(data, frequencies)
    # generate a list of per-epoch end values
    elif type(frequencies) == dict:
        values = compute_freq_bands(data, frequencies)

    result = compute_sync(values, mode, epoch_wise, time_resolved)

    return result


def compute_sync(complex_signal, mode, epoch_wise, time_resolved):
    """Compute synchrony from analytic signals.

    Parameters
    ----------
    complex_signal : array-like, shape is (2, n_epochs, n_channels, n_frequencies, n_times)
        complex array from which to compute synchrony for the two subjects.
    mode: str
        Connectivity measure to compute.
    epoch_wise : boolean
        whether to compute epoch-to-epoch synchrony. default is True.
    time_resolved : boolean
        whether to collapse the time course, only effective when epoch_wise==True

    Returns
    -------
    result : array
        Computed connectivity measure(s). The shape of each array is either (n_freq, n_epochs, n_channels, n_channels)
        if epoch_wise is True and time_resolved is False, or (n_freq, n_channels, n_channels) in other conditions.
    """
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
        complex_signal.shape[3], complex_signal.shape[4]

    # epoch wise synchrony
    if epoch_wise:
        if mode is 'envelope':
            values = np.abs(complex_signal)
            result = np.array([[[[_corrcoef(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)
        elif mode is 'power':
            values = np.abs(complex_signal)**2
            result = np.array([[[[_corrcoef(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)

        elif mode is 'plv':
            values = complex_signal / np.abs(complex_signal)
            result = np.array([[[[_plv(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)
        elif mode is 'ccorr':
            values = np.angle(complex_signal)
            result = np.array([[[[circcorrcoef(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)
        elif mode is 'proj':
            values = complex_signal
            result = np.array([[[[_proj_power_corr(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])

        elif mode is 'imagcoh':
            values = complex_signal
            result = np.array([[[[_icoh(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])
        elif mode is 'coh':
            values = complex_signal
            result = np.array([[[[_coh(values[0, epoch, ch_i, freq, :], values[1, epoch, ch_j, freq, :]) for ch_i in range(n_ch)]
                                 for ch_j in range(n_ch)]
                                for epoch in range(n_epoch)]
                               for freq in range(n_freq)])
        else:
            raise NameError('Sychrony metric ' + mode + ' not supported.')

        # whether averaging across epochs
        if time_resolved:
            result = np.nanmean(result, axis=1)

    # generate a single connectivity value from two concatenated time series
    else:
        if mode is 'envelope':
            values = np.abs(complex_signal)
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_corrcoef(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)
        elif mode is 'power':
            values = np.abs(complex_signal)**2
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_corrcoef(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])  # shape = (n_freq, n_epoch, n_ch, n_ch)
        elif mode is 'plv':
            # should be np.angle
            values = complex_signal / np.abs(complex_signal)  # phase
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_plv(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])
        elif mode is 'ccorr':
            values = np.angle(complex_signal)
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[circcorrcoef(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])
        elif mode is 'proj':
            values = complex_signal
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_proj_power_corr(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])
        elif mode is 'imagcoh':
            values = complex_signal
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_icoh(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])
        elif mode is 'coh':
            values = complex_signal
            strands = np.array(
                [np.concatenate(values[n], axis=2) for n in range(2)])  # concatenate values from all epochs
            result = np.array([[[_coh(strands[0, ch_i, freq, :], strands[1, ch_j, freq, :]) for ch_i in range(n_ch)]
                                for ch_j in range(n_ch)]
                               for freq in range(n_freq)])
        else:
            raise NameError('Sychrony metric '+mode+' not supported.')

    return result


def compute_single_freq(data, freq_range):
    """Compute analytic signal per frequency bin using a multitaper method implemented in mne

    Parameters
    ----------
    data : array-like, shape is (2, n_epochs, n_channels, n_times)
        real-valued data to compute analytic signal from.
    freq_range : list
        a list of two specifying the frequency range

    Returns
    -------
    complex_signal : array, shape is (2, n_epochs, n_channels, n_frequencies, n_times)
    """
    n_samp = data[0].shape[2]

    complex_signal = np.array([mne.time_frequency.tfr_array_multitaper(data[subject], sfreq=n_samp,
                                                                       freqs=np.arange(
                                                                           freq_range[0], freq_range[1], 1),
                                                                       n_cycles=4,
                                                                       zero_mean=False, use_fft=True, decim=1, output='complex')
                               for subject in range(2)])

    return complex_signal


def compute_freq_bands(data, freq_bands):
    """Compute analytic signal per frequency band using filtering and hilbert transform

    Parameters
    ----------
    data : array-like, shape is (2, n_epochs, n_channels, n_times)
        real-valued data to compute analytic signal from.
    freq_bands : dict
        a dict specifying names and corresponding frequency ranges

    Returns
    -------
    complex_signal : array, shape is (2, n_epochs, n_channels, n_freq_bands, n_times)
    """
    assert data[0].shape[0] == data[1].shape[0]
    n_epoch = data[0].shape[0]
    n_ch = data[0].shape[1]
    n_samp = data[0].shape[2]
    data = np.array(data)

    # filtering and hilbert transform
    complex_signal = []
    for freq_band in freq_bands.values():
        filtered = np.array([mne.filter.filter_data(data[subject], n_samp, freq_band[0], freq_band[1], verbose=False)
                             for subject in range(2)  # for each subject
                             ])
        hilb = signal.hilbert(filtered)
        complex_signal.append(hilb)

    complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])
    assert complex_signal.shape == (2, n_epoch, n_ch, len(freq_bands), n_samp)

    return complex_signal


#  Synchrony metrics


def _plv(X, Y):
    """Phase Locking Value
    takes two vectors (phase) and compute their plv
    """
    return np.abs(np.sum(np.exp(1j * (X - Y)))) / len(X)


def _coh(X, Y):
    """Coherence
    instantaneous coherence computed from hilbert transformed signal, then averaged across time points

            |A1·A2·e^(i*delta_phase)|
    Coh = -----------------------------
               sqrt(A1^2 * A2^2)

    A1: envelope of X
    A2: envelope of Y
    reference: Kida, Tetsuo, Emi Tanaka, and Ryusuke Kakigi. “Multi-Dimensional Dynamics of Human Electromagnetic Brain Activity.” Frontiers in Human Neuroscience 9 (January 19, 2016). https://doi.org/10.3389/fnhum.2015.00713.
    """
    X_phase = np.angle(X)
    Y_phase = np.angle(Y)

    Sxy = np.abs(X) * np.abs(Y) * np.exp(1j * (X_phase - Y_phase))
    Sxx = np.abs(X)**2
    Syy = np.abs(Y)**2

    coh = np.abs(Sxy/(np.sqrt(Sxx*Syy)))
    return np.nanmean(coh)


def _icoh(X, Y):
    """Coherence
    instantaneous imaginary coherence computed from hilbert transformed signal, then averaged across time points

            |A1·A2·sin(delta_phase)|
    iCoh = -----------------------------
               sqrt(A1^2 * A2^2)
    """
    X_phase = np.angle(X)
    Y_phase = np.angle(Y)

    iSxy = np.abs(X) * np.abs(Y) * np.sin(X_phase - Y_phase)
    Sxx = np.abs(X)**2
    Syy = np.abs(Y)**2

    icoh = np.abs(iSxy/(np.sqrt(Sxx*Syy)))
    return np.nanmean(icoh)


def _corrcoef(X, Y):
    """
    just pearson correlation coefficient
    """
    return np.corrcoef([X, Y])[0][1]


def _proj_power_corr(X, Y):
    # compute power proj corr using two complex signals
    # adapted from Georgios Michalareas' MATLAB script
    X_abs = np.abs(X)
    Y_abs = np.abs(Y)

    X_unit = X / X_abs
    Y_unit = Y / Y_abs

    X_abs_norm = (X_abs - np.nanmean(X_abs)) / np.nanstd(X_abs)
    Y_abs_norm = (Y_abs - np.nanmean(Y_abs)) / np.nanstd(Y_abs)

    X_ = X_abs / np.nanstd(X_abs)
    Y_ = Y_abs / np.nanstd(Y_abs)

    X_z = X_ * X_unit
    Y_z = Y_ * Y_unit
    projX = np.imag(X_z * np.conjugate(Y_unit))
    projY = np.imag(Y_z * np.conjugate(X_unit))

    projX_norm = (projX - np.nanmean(projX)) / np.nanstd(projX)
    projY_norm = (projY - np.nanmean(projY)) / np.nanstd(projY)

    proj_corr = (np.nanmean(projX_norm * Y_abs_norm) +
                 np.nanmean(projY_norm * X_abs_norm)) / 2

    return proj_corr
