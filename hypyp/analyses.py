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

import numpy as np
import scipy
import scipy.signal as signal
import scipy.stats
import statsmodels.stats.multitest
import copy
from collections import namedtuple
from typing import Union
from astropy.stats import circmean
import matplotlib.pyplot as plt

plt.ion()

import mne
from mne.io.constants import FIFF
from mne.time_frequency import EpochsSpectrum

from .mvarica import MVAR, connectivity_mvarica


def pow(epochs: mne.Epochs, fmin: float, fmax: float, n_fft: int, n_per_seg: int, epochs_average: bool) -> tuple:
    """
    Computes the Power Spectral Density (PSD) on Epochs.

    Arguments:

        epochs : mne.Epochs
            A participant's Epochs object, for a condition (can result from the
            concatenation of Epochs from different files with the same condition).
            Epochs are MNE objects: data are stored in arrays of shape
            (n_epochs, n_channels, n_times) and parameter information is stored
            in a dictionary.

        fmin : float
            Minimum frequency in Hz of interest for PSD calculation.

        fmax : float
            Maximum frequency in Hz of interest for PSD calculation.

        n_fft: The length of FFT used, must be ``>= n_per_seg`` (default: 256).
            The segments will be zero-padded if ``n_fft > n_per_seg``.
            If n_per_seg is None, n_fft must be <= number of time points
            in the data.

        n_per_seg : int | None
            Length of each Welch segment (windowed with a Hamming window). Defaults
            to None, which sets n_per_seg equal to n_fft.

        epochs_average : bool
            Option to collapse the time course or not, boolean.
            If False, PSD won't be averaged over epochs (the time
            course is maintained).
            If True, PSD values are averaged over epochs.

    Note:
        The function can be iterated on the group and/or on conditions
        (for epochs in epochs['epochs_%s_%s_%s' % (subj, group, cond_name)]).
        The PSD distribution on the group can be visualized to check normality
        for statistics.

    Returns:
        freq_list, psd:

      - freq_list: list of frequencies in the actual frequency band of interest
        (frequency bin) used for PSD calculation.
      - psd: PSD value for each epoch, each channel, and each frequency,
        ndarray (n_epochs, n_channels, n_frequencies).
        Note that if time_resolved == True, PSD values are averaged
        across epochs.
    """

    # dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs.drop_channels([ch['ch_name']])

    # computing power spectral density on epochs signal
    # average in the 1-second window around event (mean, but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                  tmin=None, tmax=None, method='welch', picks='all', proj=False,
                  n_jobs=1)
    spectrum = EpochsSpectrum(epochs, **kwargs)
    psds = spectrum.get_data()
    freq_list = spectrum.freqs
    
    if epochs_average is True:
        # averaging power across epochs for each channel ch and each frequency f
        psd = np.mean(psds, axis=0)
    else:
        psd = psds

    psd_tuple = namedtuple('PSD', ['freq_list', 'psd'])

    return psd_tuple(freq_list=freq_list,
                     psd=psd)


def behav_corr(data: np.ndarray, behav: np.ndarray, data_name: str, behav_name: str, p_thresh: float,
               multiple_corr: bool = True, verbose: bool = False) -> tuple:
    """
    Correlates data with a discontinuous behavioral parameter,
    uses different linear correlations after checking for
    normality of the data.

    Arguments:
        data: data to correlate with behavior. For now, inputs can be raw data
            or psd vectors for example (from n_dyads length), or con values
            without frequency dimension, numpy array of shape
            (n_dyads, n_channels, n_channels).
        behav: behavioral values for a parameter (ex: timing to control
            for learning), one dimensional array from same shape as data.
        data_name: nature of the data (used for the legend of the figure,
            if verbose=True), str.
        behav_name: nature of the behavior values (used for the legend
            of the figure, if verbose=True), str.
        p_thresh: threshold to consider p values as significant for correlation
            tests, can be set to 0.05 with multiple_corr to True or lower when
            multiple_corr set to False, float.
        multiple_corr: for connectivity correlation tests for example,
            a correction for multiple comparison can be applied
            (multiple_corr set to True by default), bool.
            The method used is fdr_bh.
        verbose: option to plot the correlation, boolean.
            Set to False by default.

    Returns:
        r, pvalue, strat:
          - r: pearson’s correlation coefficient, float if data is a vector,
            array of floats if data is an array of connectivity values. In this
            last case, the array can be used as input in viz.plot_links_2d.
          - pvalue: two-tailed p-value (probability of an uncorrelated
            system producing datasets that have a Pearson correlation
            at least as extreme as the one computed from the dataset tested),
            float if data is a vector, array of floats if data is an array
            of connectivity values. If multiple_corr is set to True, return
            corrected pvalue.
          - strat: normality of the datasets, 'non-normal' or 'normal' if data
            is a vector, corection set for multiple comparisons or not if data
            is an array of connectivity values, str.
    """

    # storage for results
    corr_tuple = namedtuple('corr_tuple', ['r', 'pvalue', 'strat'])

    # simple correlation between vectors (data can be averaged PSD for example)
    if data.shape == behav.shape:
        # test for normality on the first axis
        _, pvalue1 = scipy.stats.normaltest(data, axis=0)
        _, pvalue2 = scipy.stats.normaltest(behav, axis=0)
        if min(pvalue1, pvalue2) < 0.05:
            # reject null hypothesis
            # (H0: data come from a normal distribution)
            strat = 'non_normal'
            r, pvalue = scipy.stats.spearmanr(behav, data, axis=0)
        else:
            strat = 'normal'
            r, pvalue = scipy.stats.pearsonr(behav, data)
        # can also use np.convolve, np.correlate, np.corrcoeff
        # vizualisation
        if verbose:
            plt.figure()
            plt.scatter(behav, data, label=str(r) + str(pvalue))
            plt.legend(loc='upper right')
            plt.title('Linear correlation between ' + behav_name + ' and ' + data_name)
            plt.xlabel(behav_name)
            plt.ylabel(data_name)
            plt.show()
        return corr_tuple(r=r, pvalue=pvalue, strat=strat)

    # simple correlation between connectivity data and behavioral vector
    elif len(data.shape) == 3:
        rs = np.zeros(shape=(data.shape[1], data.shape[2]))
        pvals = np.zeros(shape=(data.shape[1], data.shape[2]))
        significant_corr = np.zeros(shape=(data.shape[1], data.shape[2]))
        # correlate across subjects for each pair of sensors, the connectivity value
        # with a behavioral value
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):
                r_i, pvalue_i = scipy.stats.pearsonr(behav, data[:, i, j])
                rs[i, j] = r_i
                pvals[i, j] = pvalue_i
        # correction for multiple comparisons
        if multiple_corr is True:
            # note: we reshape pvals to be able to use fdr_bh
            pvals_corrected = statsmodels.stats.multitest.multipletests(pvals.flatten(),
                                                                        alpha=0.05,
                                                                        method='fdr_bh',
                                                                        is_sorted=False,
                                                                        returnsorted=False)
            # put pval in original shape
            pvals_corrected = np.reshape(np.atleast_1d(pvals_corrected[1]), pvals.shape)
        # get r value for significant correlation only
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):
                # with pvalues non corrected for multiple comparisons
                if multiple_corr is False:
                    pvalue = pvals
                # or corrected for multiple comparisons
                else:
                    pvalue = pvals_corrected
                if pvalue[i, j] < p_thresh:
                    significant_corr[i, j] = rs[i, j]
        r = np.nan_to_num(significant_corr)
        strat = 'correction for multiple comaprison ' + str(multiple_corr)
        return corr_tuple(r=r, pvalue=pvalue, strat=strat)


def indices_connectivity_intrabrain(epochs: mne.Epochs) -> list:
    """
    Computes indices for connectivity analysis between all EEG
    channels for one participant. Can be used instead of
    (n_channels, n_channels) that takes into account intrabrain channel
    connectivity.

    Arguments:
        epochs: one participant's Epochs object, to retrieve channel information.
            (Epochs are MNE objects).

    Returns:
        channels: channel pairs for which connectivity indices will be
            computed, a list of tuples with channels indices.
    """
    names = copy.deepcopy(epochs.info['ch_names'])
    for ch in epochs.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EOG_CH:
            names.remove(ch['ch_name'])

    n = len(names)
    bin = 0
    idx = []
    channels = []
    for e1 in range(n):
        for e2 in range(n):
            if e2 > e1:
                idx.append(bin)
                channels.append((e1, e2))
            bin = bin + 1

    return channels


def indices_connectivity_interbrain(epoch_hyper: mne.Epochs) -> list:
    """
    Computes indices for interbrain connectivity analyses between all EEG
    sensors for 2 participants (merge data).

    Arguments:
        epoch_hyper: a pair's Epochs object; contains channel information (Epochs
            are MNE objects).

    Returns:
        channels: channel pairs for which connectivity indices will be
            computed, a list of tuples with channels indices.
    """
    channels = []
    names = copy.deepcopy(epoch_hyper.info['ch_names'])
    for ch in epoch_hyper.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EOG_CH:
            names.remove(ch['ch_name'])

    l = list(range(0, int(len(names) / 2)))
    # l = list(range(0,62))
    L = []
    M = len(l) * list(range(len(l), len(l) * 2))
    for i in range(0, len(l)):
        for p in range(0, len(l)):
            L.append(l[i])
    for i in range(0, len(L)):
        channels.append((L[i], M[i]))

    return channels


def pair_connectivity(data: Union[list, np.ndarray], sampling_rate: int, frequencies: Union[dict, list], mode: str,
                      epochs_average: bool = True) -> np.ndarray:
    """
    Computes frequency- or time-frequency-domain connectivity measures from preprocessed EEG data.
    This function aggregates compute_single_freq/compute_freq_bands and compute_sync.

    Arguments:

        data:
            shape = (2, n_epochs, n_channels, n_times). data input for computing connectivity between two participants
        sampling_rate:
            sampling rate.
        frequencies :
            frequencies of interest for which connectivity will be computed.
            If a dictionary, different frequency bands are used.
            - e.g. {'alpha':[8,12],'beta':[12,20]}
            If a list, every integer frequency within the range is used.
            - e.g. [5,30] dictates that connectivity will be computed over every integer in the frequency bin between 5 Hz and 30 Hz.

        mode:
            connectivity measure. Options are in the notes.

        epochs_average:
            option to either return the average connectivity across epochs (collapse across time) or preserve epoch-by-epoch connectivity, boolean.
            If False, PSD won't be averaged over epochs (the time
            course is maintained).
            If True, PSD values are averaged over epochs.


    Returns:
        result:
            Connectivity matrix. The shape is either
            (n_freq, n_epochs, 2*n_channels, 2*n_channels) if time_resolved is False,
            or (n_freq, 2*n_channels, 2*n_channels) if time_resolved is True.

            To extract inter-brain connectivity values, slice the last two dimensions of con with [0:n_channels, n_channels: 2*n_channels].


    Note:
        Connectivity is computed for all possible electrode pairs between
        the dyad, including inter- and intra-brain connectivities.

      **supported connectivity measures**
          - 'envelope_corr': envelope correlation
          - 'pow_corr': power correlation
          - 'plv': phase locking value
          - 'ccorr': circular correlation coefficient
          - 'coh': coherence
          - 'imaginary_coh': imaginary coherence
          - 'pli': phase lag index
          - 'wpli': weighted phase lag index
    """

    # Data consists of two lists of np.array (n_epochs, n_channels, epoch_size)
    assert data[0].shape[0] == data[1].shape[0], "Two streams much have the same lengths."

    # compute instantaneous analytic signal from EEG data
    if type(frequencies) == list:
        # average over tapers
        values = np.mean(compute_single_freq(data, sampling_rate, frequencies), 3).squeeze()
    elif type(frequencies) == dict:
        values = compute_freq_bands(data, sampling_rate, frequencies)
    else:
        TypeError("Please use a list or a dictionary to specify frequencies.")

    # compute connectivity values
    result = compute_sync(values, mode, epochs_average)

    return result


# helper function
def _multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    It is designed specifically to collapse the last dimension of a four-dimensional array.

    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.

    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


# helper function
def _multiply_conjugate_time(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    Unlike _multiply_conjugate, this doenst collapse the last dimension of a 
    four-dimensional array. Useful when computing some connectivity metrics 
    (e.g., wpli), since it preserves the product values across e.g., time.
    
    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.
    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))
    
    return product


def compute_sync(complex_signal: np.ndarray, mode: str, epochs_average: bool = True) -> np.ndarray:
    """
    Computes frequency- or time-frequency-domain connectivity measures from analytic signals.

    Arguments:

        complex_signal:
            shape = (2, n_epochs, n_channels, n_freq_bins, n_times).
            Analytic signals for computing connectivity between two participants.

        mode:
            Connectivity measure. Options in the notes.

        epochs_average:
            option to either return the average connectivity across epochs (collapse across time) or preserve epoch-by-epoch connectivity, boolean.
            If False, PSD won't be averaged over epochs (the time course is maintained).
            If True, PSD values are averaged over epochs.


    Returns:
        con:
            Connectivity matrix. The shape is either
            (n_freq, n_epochs, 2*n_channels, 2*n_channels) if time_resolved is False,
            or (n_freq, 2*n_channels, 2*n_channels) if time_resolved is True.

            To extract inter-brain connectivity values, slice the last two dimensions of con with [0:n_channels, n_channels: 2*n_channels].

    Note:
        **supported connectivity measures**
          - 'envelope_corr': envelope correlation
          - 'pow_corr': power correlation
          - 'plv': phase locking value
          - 'ccorr': circular correlation coefficient
          - 'coh': coherence
          - 'imaginary_coh': imaginary coherence
          - 'pli': phase lag index
          - 'wpli': weighted phase lag index

    """

    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)
    if mode.lower() == 'plv':
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = abs(dphi) / n_samp

    elif mode.lower() == 'envelope_corr':
        env = np.abs(complex_signal)
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
              np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() == 'pow_corr':
        env = np.abs(complex_signal) ** 2
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
              np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() == 'coh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                               np.nansum(amp, axis=3)))

    elif mode.lower() == 'imaginary_coh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                        np.nansum(amp, axis=3)))

    elif mode.lower() == 'ccorr':
        angle = np.angle(complex_signal)
        mu_angle = circmean(angle, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        angle = np.sin(angle - mu_angle)

        formula = 'nilm,nimk->nilk'
        con = np.abs(np.einsum(formula, angle, angle.transpose(transpose_axes)) /
                     np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), 
                                       np.sum(angle ** 2, axis=3))))
        
    elif mode.lower() == 'pli':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        con = abs(np.mean(np.sign(np.imag(dphi)), axis=4))
        
    elif mode.lower() == 'wpli':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
        con_num = abs(np.mean(abs(np.imag(dphi)) * np.sign(np.imag(dphi)), axis=4))
        con_den = np.mean(abs(np.imag(dphi)), axis=4)      
        con_den[con_den == 0] = 1 
        con = con_num / con_den        

    else:
        ValueError('Metric type not supported.')

    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    if epochs_average:
        con = np.nanmean(con, axis=1)

    return con


def compute_conn_mvar(complex_signal: np.ndarray, mvar_params: dict, ica_params: dict, measure_params: dict, check_stability: bool = True) -> np.ndarray:
    """
    Computes connectivity measures based on MVAR coefficients.

    Arguments:
      complex_signal:
        shape = (2, n_epochs, n_channels, n_freq_bins, n_times).
        Analytic signals for computing connectivity between two participants.

      mvar_params:
        python dictionary for defining the MVAR model parameters.
        it should contain three variables:
        { "mvar_order": ,
        "fitting_method: ,
        "delta: ,
        }

      ica_params:
        python dictionary for choosing the ica method.
        it should contain two variables:
        { "method": ,
        "random_state":
        }

        measure_params:
          python dictionary for defining connectivity measure attributes.
          it should contain two variables:
          { "name": ,
          "n_fft":
          }

        check_stability:
          bool, whether to check stability of mvar model or not.
          note that mvar models need adequate sample number to be stable.
          (default: True)

    Returns:
      connectivity measure matrix.
      ndarray with shape = (epochs, frequency, channels, channels, n_fft)
                          or (1, frequency,channels, channels, n_fft) if epochs are merged
    """
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    real_signal = np.real(complex_signal)
    aug_signal = real_signal[np.newaxis, ...]

    mvar = MVAR(mvar_params["mvar_order"], mvar_params["fitting_method"], mvar_params["delta"])

    if check_stability:
        c = True
        fit_mvar = mvar.fit(aug_signal[:, 0, 0, :, :])
        is_stable = fit_mvar.stability()

        while c:

            if is_stable:
                print("MVAR model is stable.")
                inp = input("Do you want to continue? ")

                if inp.lower() == "yes":
                    aux_3 = np.zeros((aug_signal.shape[1], aug_signal.shape[2], real_signal.shape[2],
                                      real_signal.shape[2], measure_params["n_fft"]), dtype=np.complex128)
                    for e in range(aug_signal.shape[1]):
                        for f in range(aug_signal.shape[2]):
                            aux_sig = aug_signal[:, e, f, :, :]
                            conn_freq_epoch = connectivity_mvarica(real_signal=aux_sig, ica_params=ica_params,
                                                                   measure_name=measure_params["name"],
                                                                   n_fft=measure_params["n_fft"], var_model=mvar)
                            d_type = conn_freq_epoch.dtype
                            aux_3[e, f, :, :, :] = conn_freq_epoch

                    return np.asarray(aux_3, dtype=d_type)

                else:

                    return None
            else:

                counter = 0

                if counter == 0:

                    print("MVAR model is not stable: number of time samples may be too small!")
                    print("\n")
                    nes_sample = mvar_params["mvar_order"] * real_signal.shape[2] * real_signal.shape[2]
                    print("At least " + str(nes_sample) + " samples are required for fitting MVAR model.")
                    print("\n")
                    inp = input("Do you want to merge the epochs?")

                    if inp.lower() == "yes":

                        merged_signal = aug_signal.reshape(1, real_signal.shape[1], real_signal.shape[2],
                                                           real_signal.shape[3] * real_signal.shape[0])
                        fit_mvar = mvar.fit(merged_signal[:, 0, 0, :][np.newaxis, ...])
                        is_stable = fit_mvar.stability()
                        aug_signal = merged_signal[np.newaxis, ...]
                        counter += counter

                    else:

                        return None
                else:

                    return "epochs are already merged"
    else:

        aux_3 = np.zeros((aug_signal.shape[1], aug_signal.shape[2], real_signal.shape[2], real_signal.shape[2],
                          measure_params["n_fft"]), dtype=np.complex128)
        for e in range(aug_signal.shape[1]):
            for f in range(aug_signal.shape[2]):
                aux_sig = aug_signal[:, e, f, :, :]
                conn_freq_epoch = connectivity_mvarica(real_signal=aux_sig, ica_params=ica_params,
                                                       measure_name=measure_params["name"],
                                                       n_fft=measure_params["n_fft"], var_model=mvar)
                d_type = conn_freq_epoch.dtype
                aux_3[e, f, :, :, :] = conn_freq_epoch

        return np.asarray(aux_3, dtype=d_type)


def compute_single_freq(data: np.ndarray, sampling_rate: int, freq_range: list) -> np.ndarray:
    """
    Computes analytic signal per frequency bin using the multitaper method.

    Arguments:
        data:
            shape is (2, n_epochs, n_channels, n_times)
            real-valued data used to compute analytic signal.
        sampling_rate:
            sampling rate.
        freq_range:
            a list of two specifying the frequency range.
            e.g. [5,30] refers to every integer in the frequency bin from 5 Hz to 30 Hz.
    Returns:
        complex_signal:
          shape is (2, n_epochs, n_channels, n_tapers, n_frequencies, n_times)
    """

    complex_signal = np.array([mne.time_frequency.tfr_array_multitaper(data[participant], sfreq=sampling_rate,
                                                                       freqs=np.arange(
                                                                           freq_range[0], freq_range[1], 1),
                                                                       n_cycles=4, zero_mean=False, use_fft=True,
                                                                       decim=1,
                                                                       output='complex')
                               for participant in range(2)])

    return complex_signal


def compute_freq_bands(data: np.ndarray, sampling_rate: int, freq_bands: dict, filter_signal: bool = True , **filter_options) -> np.ndarray:
    """
    Computes analytic signal per frequency band using FIR filtering
    and Hilbert transform.

    Arguments:
        data:
            shape is (2, n_epochs, n_channels, n_times)
            real-valued data to compute analytic signal from.
        sampling_rate:
            sampling rate.
        filter_signal:
            bool: whether to apply a filter on the signal,
            default, true
        freq_bands:
            a dictionary specifying frequency band labels and corresponding frequency ranges
            e.g. {'alpha':[8,12], 'beta':[12,20]} indicates that computations are performed over two frequency bands: 8-12 Hz for the alpha band and 12-20 Hz for the beta band.
        **filter_options:
            additional arguments for mne.filter.filter_data, such as filter_length, l_trans_bandwidth, h_trans_bandwidth
    Returns:
        complex_signal: array, shape is
            (2, n_epochs, n_channels, n_freq_bands, n_times)
    """
    assert data[0].shape[0] == data[1].shape[0], "Two data streams should have the same number of trials."
    data = np.array(data)

    # filtering and Hilbert transform
    complex_signal = []
    for freq_band in freq_bands.values():
        if filter_signal:
            filtered = np.array([mne.filter.filter_data(data[participant],
                                                    sampling_rate, l_freq=freq_band[0], h_freq=freq_band[1],
                                                    **filter_options,
                                                    verbose=False)
                             for participant in range(2)
                             # for each participant
                             ])
        else:
            filtered=np.array([data[participant] for participant in range(2)])
        hilb = signal.hilbert(filtered)
        complex_signal.append(hilb)

    complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])

    return complex_signal


def compute_nmPLV(data: np.ndarray, sampling_rate: int, freq_range1: list, freq_range2: list, **filter_options) -> np.ndarray:
    """
    Computes the n:m PLV for a dyad with two different frequency ranges.

    Arguments:
        data : np.ndarray
            shape is (2, n_epochs, n_channels, n_times)
            real-valued data to compute analytic signal from.
        sampling_rate: int
            sampling rate.
        freq_range1 : list
            a list of two specifying the frequency range for participant 1
            e.g. [5,30] refers to every integer in the frequency bin from 5 Hz to 30 Hz.
        freq_range2 : list
            a list of two specifying the frequency range for participant 1
            e.g. [5,30] refers to every integer in the frequency bin from 5 Hz to 30 Hz.
        **filter_options:
            additional arguments for mne.filter.filter_data, such as filter_length, l_trans_bandwidth, h_trans_bandwidth
    Returns:
        con:
            Connectivity matrix. The shape is either (n_freq, 2*n_channels, 2*n_channels)

            To extract inter-brain connectivity values, slice the last two dimensions of con with [0:n_channels, n_channels: 2*n_channels].
    """
    r = np.mean(freq_range2)/np.mean(freq_range1)
    freq_range = [np.min(freq_range1), np.max(freq_range2)]
    complex_signal = np.mean(compute_single_freq(data, sampling_rate, freq_range, **filter_options),3).squeeze()

    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)
    phase = complex_signal / np.abs(complex_signal)

    freqsn = freq_range
    freqsm = [f * r for f in freqsn]
    n_mult = (freqsn[0] + freqsm[0]) / (2 * freqsn[0])
    m_mult = (freqsm[0] + freqsn[0]) / (2 * freqsm[0])

    phase[:, :, :, :n_ch] = n_mult * phase[:, :, :, :n_ch]
    phase[:, :, :, n_ch:] = m_mult * phase[:, :, :, n_ch:]

    c = np.real(phase)
    s = np.imag(phase)
    dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    con = abs(dphi) / n_samp
    con = np.nanmean(con, axis=1)
    return con


def xwt(sig1: mne.Epochs, sig2: mne.Epochs,
        freqs: Union[int, np.ndarray], n_cycles=5.0, mode: str = "xwt") -> np.ndarray:
    """
    Performs a cross wavelet transform on two signals.

    Arguments:

        sig1 : mne.Epochs
            Signal (eg. EEG data) of first participant.

        sig2 : mne.Epochs
            Signal (eg. EEG data) of second participant.

        freqs: int | float
            Range of frequencies of interest in Hz.

        mode: str
            Sets the type of analyses.

    Note:
        This function relies on MNE's mne.time_frequency.morlet
        and mne.time_frequency.tfr.cwt functions.
    
    Returns:
        data:
            Wavelet results. The shape is (n_chans1, n_chans2, n_epochs, n_freqs, n_samples).
            Wavelet transform coherence calculated according to Maraun & Kurths (2004)
    """
    
    # Set parameters for the output
    n_freqs = len(freqs)
    sfreq = sig1.info['sfreq']
    assert sig1.info['sfreq'] == sig2.info['sfreq'], "Sig1 et sig2 should have the same sfreq value."

    n_epochs1, n_chans1, n_samples1 = sig1.get_data().shape
    n_epochs2, n_chans2, n_samples2 = sig2.get_data().shape

    assert n_epochs1 == n_epochs2, "n_epochs1 and n_epochs2 should have the same number of epochs."
    assert n_chans1 == n_chans2, "n_chans1 and n_chans2 should have the same number of channels."
    assert n_samples1 == n_samples2, "n_samples1 and n_samples2 should have the same number of samples."

    cross_sigs = np.zeros((n_chans1, n_chans2, n_epochs1, n_freqs, n_samples1), dtype=complex) * np.nan
    wcts = np.zeros((n_chans1, n_chans2, n_epochs1, n_freqs, n_samples1), dtype=complex) * np.nan

    # Set the mother wavelet
    Ws = mne.time_frequency.tfr.morlet(sfreq, freqs, 
                                       n_cycles=n_cycles, sigma=None, zero_mean=True)

    # Perform a continuous wavelet transform on all epochs of each signal
    for ind1, ch_label1 in enumerate(sig1.ch_names):
        for ind2, ch_label2 in enumerate(sig2.ch_names):
            # Extract the channel's data for both participants and apply cwt
            cur_sig1 = np.squeeze(sig1.get_data(mne.pick_channels(sig1.ch_names, [ch_label1])))
            out1 = mne.time_frequency.tfr.cwt(cur_sig1, Ws, use_fft=True,
                                              mode='same', decim=1)
            cur_sig2 = np.squeeze(sig2.get_data(mne.pick_channels(sig2.ch_names, [ch_label2])))
            out2 = mne.time_frequency.tfr.cwt(cur_sig2, Ws, use_fft=True,
                                              mode='same', decim=1)
            
            # Compute cross-spectrum
            wps1 = out1 * out1.conj()
            wps2 = out2 * out2.conj()
            cross_sig = out1 * out2.conj()
            cross_sigs[ind1, ind2, :, :, :] = cross_sig
            coh = (cross_sig) / (np.sqrt(wps1*wps2))
            abs_coh = np.abs(coh)
            wct = (abs_coh - np.min(abs_coh)) / (np.max(abs_coh) - np.min(abs_coh))
            wcts[ind1, ind2, :, :, :] = wct

    if mode == 'power':
        data = np.abs(cross_sigs)
    elif mode == 'phase':
        data = np.angle(cross_sigs)
    elif mode == 'xwt':
        data = cross_sigs
    elif mode == 'wtc':
        data = wcts 
    else:
        data = 'Please specify a valid mode: power, phase, xwt, or wtc.'
        print(data)
    return data