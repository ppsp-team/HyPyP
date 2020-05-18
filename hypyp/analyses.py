#!/usr/bin/env python
# coding=utf-8

"""
PSD, intra- and inter-brain measures functions

| Option | Description |
| ------ | ----------- |
| title           | analyses.py |
| authors         | Phoebe Chen, Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


from collections import namedtuple
import copy
import numpy as np
import scipy.signal as signal
from astropy.stats import circmean
import mne
from mne.time_frequency import psd_welch
from mne.io.constants import FIFF


def PSD(epochs: mne.Epochs, fmin: float, fmax: float, n_fft: int, n_per_seg: int, time_resolved: bool)-> tuple:
    """
    Computes the Power Spectral Density (PSD) on Epochs.

    Arguments:
        epochs: A subject's Epochs object, for a condition (can result from the
          concatenation of Epochs from different experimental realisations
          of the condition).
                Epochs are MNE objects: data are stored in arrays of shape
          (n_epochs, n_channels, n_times) and parameters information are stored
          in a dictionnary.
        fmin, fmax: minimum and maximum frequencies-of-interest for PSD calculation,
          floats in Hz.
        n_fft: The length of FFT used, must be ``>= n_per_seg`` (default: 256).
          The segments will be zero-padded if ``n_fft > n_per_seg``.
          If n_per_seg is None, n_fft must be <= number of time points
          in the data.
        n_per_seg : int | None
          Length of each Welch segment (windowed with a Hamming window). Defaults
          to None, which sets n_per_seg equal to n_fft.
        time_resolved: option to collapse the time course or not, boolean.
          If False, PSD won't be averaged over epochs (the time
          course is maintained).
          If True, PSD values are averaged over epochs.

    Note:
        The function can be iterated on the group and/or on conditions
      (for epochs in epochs['epochs_%s_%s_%s' % (subj, group, cond_name)]).
      You can visualize PSD distribution on the group to check normality
      for statistics.

    Returns:
        freqs_mean, PSD_welch:

          - freqs_mean: list of frequencies in frequency-band-of-interest
          actually used for PSD calculation.
          - PSD_welch: PSD value in epochs for each channel and each frequency,
          ndarray (n_epochs, n_channels, n_frequencies).
          Note that if time_resolved == True, PSD values are averaged
          across epochs.
    """
    # dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs.drop_channels([ch['ch_name']])

    # computing power spectral density on epochs signal
    # average in the 1second window around event (mean but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg, n_jobs=1)
    psds_welch, freqs_mean = psd_welch(
        epochs, **kwargs, average='mean', picks='all')  # or median

    if time_resolved is True:
        # averaging power across epochs for each ch and each f
        PSD_welch = np.mean(psds_welch, axis=0)
    else:
        PSD_welch = psds_welch

    PSDTuple = namedtuple('PSD', ['freqs_mean', 'PSD_welch'])

    return PSDTuple(freqs_mean=freqs_mean,
                    PSD_welch=PSD_welch)


def indexes_connectivity_intrabrain(epochs: mne.Epochs) -> list:
    """
    Computes indexes for connectivity analysis between all EEG
    sensors for one subject. Can be used instead of
    (n_channels, n_channels) that takes into account intra electrodes
    connectivity.

    Arguments:
        epochs: one subject Epochs object to get channels information
          (Epochs are MNE objects).

    Returns:
        electrodes: electrodes pairs for which connectivity indices will be
          computed, list of tuples with channels indexes.
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


def indexes_connectivity_interbrains(epoch_hyper: mne.Epochs) -> list:
    """
    Computes indexes for interbrains connectivity analyses between all EEG
    sensors for 2 subjects (merge data).

    Arguments:
        epoch_hyper: one dyad Epochs object to get channels information (Epochs
          are MNE objects).

    Note:
        Only interbrains connectivity will be computed.

    Returns:
        electrodes: electrodes pairs for which connectivity indices will be
          computed, list of tuples with channels indexes.
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


def simple_corr(data, frequencies, mode, time_resolved) -> np.ndarray:
    """
    Computes frequency- and time-frequency-domain connectivity measures.

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          The data from which to compute connectivity between two subjects
        frequencies : dict | list
          frequencies of interest to compute connectivity with.
          If a dict, different frequency bands are used.
          e.g. {'alpha':[8,12],'beta':[12,20]}
          If a list, every integer frequency within the range is used.
          e.g. [5,30]
        mode: string
          Connectivity measure to compute.
          'envelope': envelope correlation
          'power': power correlation
          'plv': phase locking value
          'ccorr': circular correlation coefficient
          'coh': coherence
          'imagcoh': imaginary coherence
          'proj': projected power correlation
        time_resolved: boolean
          whether to collapse the time course.
          if False, synchrony won't be averaged over epochs, and the time
          course is maintained.
          if True, synchrony is averaged over epochs.

    Note:
        Connectivity is computed for all possible electrode pairs between
        the dyad, but doesn't include intrabrain synchrony.

    Returns:
        result: array
          Computed connectivity measure(s). The shape of each array is either
          (n_freq, n_epochs, n_channels, n_channels) if epoch_wise is True
          and time_resolved is False, or (n_freq, n_channels, n_channels)
          in other conditions.
    """
    # Data consists of two lists of np.array (n_epochs, n_channels, epoch_size)
    assert data[0].shape[0] == data[1].shape[0], "Two streams much have the same lengths."

    # compute correlation coefficient for all symmetrical channel pairs
    if type(frequencies) == list:
        values = compute_single_freq(data, frequencies)
    # generate a list of per-epoch end values
    elif type(frequencies) == dict:
        values = compute_freq_bands(data, frequencies)
    else:
        TypeError("Please use a list or a dictionary for specifying frequencies.")

    result = compute_sync(values, mode, time_resolved)

    return result


# helper function
def _multiply_conjugate(real, imag, transpose_axes):
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
           np.einsum(formula, imag, imag.transpose(transpose_axes)) + 1j * \
           (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
            np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def compute_sync(complex_signal, mode, time_resolved=True):
    """
      (improved) Computes synchrony from analytic signals.

    """
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
        complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)
    if mode.lower() is 'plv':
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = abs(dphi) / n_samp

    elif mode.lower() is 'envelope':
        env = np.abs(complex_signal)
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
               np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() is 'powercorr':
        env = np.abs(complex_signal)**2
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
               np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() is 'coh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                    np.nansum(amp, axis=3)))

    elif mode.lower() is 'imagcoh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                    np.nansum(amp, axis=3)))

    elif mode.lower() is 'ccorr':
        angle = np.angle(complex_signal)
        mu_angle = circmean(angle, axis=3).reshape(n_epoch, n_freq, 2*n_ch, 1)
        angle = np.sin(angle - mu_angle)

        formula = 'nilm,nimk->nilk'
        con = np.einsum(formula, angle, angle.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), np.sum(angle ** 2, axis=3)))

    else:
        ValueError('Metric type not supported.')

    con = con.swapaxes(0, 1)  # n_freq x n_epoch x n_ch x n_ch
    if time_resolved:
        con = np.nanmean(con, axis=1)

    return con


def compute_single_freq(data: np.ndarray, freq_range: list) -> np.ndarray:
    """
    Computes analytic signal per frequency bin using a multitaper method
    implemented in MNE.

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          real-valued data to compute analytic signal from.
        freq_range: list
          a list of two specifying the frequency range

    Returns:
        complex_signal: array, shape is
          (2, n_epochs, n_channels, n_frequencies, n_times)
    """
    n_samp = data[0].shape[2]

    complex_signal = np.array([mne.time_frequency.tfr_array_multitaper(data[subject], sfreq=n_samp,
                                                                       freqs=np.arange(
                                                                           freq_range[0], freq_range[1], 1),
                                                                       n_cycles=4,
                                                                       zero_mean=False, use_fft=True, decim=1, output='complex')
                               for subject in range(2)])

    return complex_signal


def compute_freq_bands(data: np.ndarray, freq_bands: dict) -> np.ndarray:
    """
    Computes analytic signal per frequency band using filtering
    and hilbert transform

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          real-valued data to compute analytic signal from.
        freq_bands: dict
          a dict specifying names and corresponding frequency ranges

    Returns:
        complex_signal: array, shape is
          (2, n_epochs, n_channels, n_freq_bands, n_times)
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