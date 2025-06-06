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
from scipy.stats import circmean
import statsmodels.stats.multitest
import copy
from collections import namedtuple
from typing import Union, List, Tuple
import matplotlib.pyplot as plt

plt.ion()

import mne
from mne.io.constants import FIFF
from mne.time_frequency import EpochsSpectrum

from .mvarica import MVAR, connectivity_mvarica


def pow(epochs: mne.Epochs, fmin: float, fmax: float, n_fft: int, n_per_seg: int, epochs_average: bool) -> namedtuple:
    """
    Computes the Power Spectral Density (PSD) on Epochs using Welch's method.
    
    This function calculates the power spectrum for each channel across the specified 
    frequency range. EOG channels are automatically dropped before computation.
    
    Parameters
    ----------
    epochs : mne.Epochs
        A participant's Epochs object containing EEG data of shape (n_epochs, n_channels, n_times).
        
    fmin : float
        Minimum frequency in Hz to include in PSD calculation.
        
    fmax : float
        Maximum frequency in Hz to include in PSD calculation.
        
    n_fft : int
        Length of FFT used. Must be >= n_per_seg. If larger, the segments will be 
        zero-padded. If n_per_seg is None, n_fft must be <= number of time points.
        
    n_per_seg : int or None
        Length of each Welch segment (windowed with a Hamming window).
        If None, n_per_seg is set equal to n_fft.
        
    epochs_average : bool
        If True, PSD values are averaged over epochs.
        If False, PSD won't be averaged (the time course is maintained).
    
    Returns
    -------
    psd_tuple : namedtuple
        A named tuple containing:
        - freq_list: ndarray of frequencies (frequency bins) in Hz
        - psd: ndarray of PSD values with shape:
          - If epochs_average=True: (n_channels, n_frequencies)
          - If epochs_average=False: (n_epochs, n_channels, n_frequencies)
          
    Notes
    -----
    This function can be iterated on groups and/or conditions:
    (for epochs in epochs['epochs_%s_%s_%s' % (subj, group, cond_name)]).
    
    The PSD values are computed using Welch's method with Hamming windows and 
    expressed in µV²/Hz.
    
    Examples
    --------
    >>> from mne import Epochs
    >>> result = pow(epochs, fmin=1, fmax=40, n_fft=512, n_per_seg=256, epochs_average=True)
    >>> frequencies = result.freq_list
    >>> power_values = result.psd
    """

    # dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs.drop_channels([ch['ch_name']])

    # computing power spectral density on epochs signal
    # average in the 1-second window around event (mean, but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                  tmin=None, tmax=None, method='welch', picks='all', exclude=[],
                  proj=False, remove_dc=True, n_jobs=1)
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


def behav_corr(data: np.ndarray, behav: np.ndarray, data_name: str, behav_name: str, 
               p_thresh: float, multiple_corr: bool = True, verbose: bool = False) -> namedtuple:
    """
    Correlates neural data with behavioral parameters using appropriate linear correlation methods.
    
    This function first checks for data normality, then applies either Pearson's or 
    Spearman's correlation. For connectivity data, it can apply multiple comparison correction.
    
    Parameters
    ----------
    data : np.ndarray
        Data to correlate with behavior. Can be either:
        - 1D array of values (same shape as behav)
        - 3D array of connectivity values with shape (n_dyads, n_channels, n_channels)
        
    behav : np.ndarray
        1D array of behavioral values (e.g., timing, performance scores)
        
    data_name : str
        Description of the data (used for plot labeling if verbose=True)
        
    behav_name : str
        Description of the behavioral parameter (used for plot labeling if verbose=True)
        
    p_thresh : float
        Significance threshold for correlation tests (typically 0.05)
        
    multiple_corr : bool, optional
        Whether to apply multiple comparison correction (default=True).
        Uses the FDR-BH method when True.
        
    verbose : bool, optional
        Whether to generate visualization of the correlation (default=False)
    
    Returns
    -------
    corr_tuple : namedtuple
        A named tuple containing:
        - r: correlation coefficient(s)
          - For 1D data: a single float
          - For connectivity data: array of shape (n_channels, n_channels)
        - pvalue: p-value(s) (original or corrected if multiple_corr=True)
        - strat: string indicating which strategy was used ('normal', 'non_normal',
          or information about multiple comparison correction)
    
    Notes
    -----
    When correlating connectivity matrices with behavioral data, only the significant 
    correlations are returned in the r matrix, with non-significant values set to zero.
    
    Examples
    --------
    >>> # Correlating average alpha power with reaction times
    >>> result = behav_corr(alpha_power, reaction_times, 'Alpha Power', 'RT (ms)', 0.05)
    >>> r_value = result.r
    >>> p_value = result.pvalue
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


def indices_connectivity_intrabrain(epochs: mne.Epochs) -> List[Tuple[int, int]]:
    """
    Computes indices for intrabrain connectivity analysis between all EEG channels.
    
    This function generates all possible pairs of EEG channel indices for a single
    participant, excluding EOG channels.
    
    Parameters
    ----------
    epochs : mne.Epochs
        A participant's Epochs object containing channel information
        
    Returns
    -------
    channels : List[Tuple[int, int]]
        List of tuples, each containing a pair of channel indices (i, j) where i < j.
        These indices correspond to all possible pairs of EEG channels for the participant.
    
    Notes
    -----
    This function automatically removes EOG channels before generating the pairs.
    The resulting indices can be used as input for connectivity analyses within
    a single brain.
    
    Examples
    --------
    >>> channel_pairs = indices_connectivity_intrabrain(participant_epochs)
    >>> print(f"Number of channel pairs: {len(channel_pairs)}")
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


def indices_connectivity_interbrain(epoch_hyper: mne.Epochs) -> List[Tuple[int, int]]:
    """
    Computes indices for interbrain connectivity analysis between EEG channels of two participants.
    
    This function generates all possible pairs of channel indices where the first channel
    belongs to participant 1 and the second to participant 2, based on a merged epochs object.
    
    Parameters
    ----------
    epoch_hyper : mne.Epochs
        A merged Epochs object containing data from both participants, with channels
        ordered such that the first half belongs to participant 1 and the second half
        to participant 2.
        
    Returns
    -------
    channels : List[Tuple[int, int]]
        List of tuples, each containing a pair of channel indices (i, j) where:
        - i is an index of a channel from participant 1 (in range 0 to n_channels/2 - 1)
        - j is an index of a channel from participant 2 (in range n_channels/2 to n_channels - 1)
    
    Notes
    -----
    This function assumes that the channels in epoch_hyper are organized as:
    [participant1_ch1, ..., participant1_chN, participant2_ch1, ..., participant2_chN]
    
    EOG channels are automatically excluded before generating the pairs.
    
    Examples
    --------
    >>> interbrain_pairs = indices_connectivity_interbrain(merged_epochs)
    >>> print(f"Number of interbrain channel pairs: {len(interbrain_pairs)}")
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


def pair_connectivity(data: Union[list, np.ndarray], sampling_rate: int, 
                     frequencies: Union[dict, list], mode: str,
                     epochs_average: bool = True) -> np.ndarray:
    """
    Computes frequency-domain connectivity measures between two participants.
    
    This function is a high-level interface that processes EEG data, computes analytic 
    signals, and calculates connectivity metrics between all possible channel pairs.
    
    Parameters
    ----------
    data : Union[list, np.ndarray]
        EEG data from two participants with shape (2, n_epochs, n_channels, n_times)
        
    sampling_rate : int
        Sampling rate of the EEG data in Hz
        
    frequencies : Union[dict, list]
        Specification of frequency bands of interest:
        - If dict: {'band_name': [fmin, fmax], ...} defining frequency bands
          e.g., {'alpha': [8, 12], 'beta': [12, 20]}
        - If list: [fmin, fmax] for each integer frequency in the range
          e.g., [5, 30] for all integer frequencies from 5 to 30 Hz
          
    mode : str
        Connectivity measure to compute. Options:
        - 'envelope_corr': envelope correlation
        - 'pow_corr': power correlation
        - 'plv': phase locking value
        - 'ccorr': circular correlation coefficient
        - 'coh': coherence
        - 'imaginary_coh': imaginary coherence
        - 'pli': phase lag index
        - 'wpli': weighted phase lag index
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
    
    Returns
    -------
    result : np.ndarray
        Connectivity matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
        
        The channels are ordered as:
        [participant1_ch1, ..., participant1_chN, participant2_ch1, ..., participant2_chN]
        
        To extract only interbrain connectivity values, slice the last two dimensions:
        result[:, :, 0:n_channels, n_channels:2*n_channels]
    
    Notes
    -----
    This function handles the complete process from raw EEG data to connectivity metrics:
    1. Computes analytic signals for the specified frequencies
    2. Applies the selected connectivity measure
    
    Each connectivity metric has different mathematical properties and interpretations:
    - PLV measures consistency in phase differences regardless of amplitude
    - Coherence measures linear relationship between signals in frequency domain
    - Imaginary coherence is less susceptible to volume conduction
    - PLI and wPLI are robust against common sources and amplitude effects
    
    Time complexity scales with O(n_channels^2 × n_epochs × n_frequencies × n_times)
    
    Examples
    --------
    >>> # Computing alpha band PLV
    >>> alpha_plv = pair_connectivity(
    ...     [subj1_data, subj2_data], 
    ...     sampling_rate=256, 
    ...     frequencies={'alpha': [8, 12]}, 
    ...     mode='plv'
    ... )
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
        raise TypeError("Please use a list or a dictionary to specify frequencies.")

    # compute connectivity values
    result = compute_sync(values, mode, epochs_average)

    return result


def compute_sync(complex_signal: np.ndarray, mode: str, epochs_average: bool = True) -> np.ndarray:
    """
    Computes frequency-domain connectivity measures from analytic signals.
    
    This function calculates various connectivity metrics between all possible
    channel pairs based on the input complex-valued signals.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (2, n_epochs, n_channels, n_freq_bins, n_times)
        
    mode : str
        Connectivity measure to compute. Options:
        - 'envelope_corr': envelope correlation - correlation between signal envelopes
        - 'pow_corr': power correlation - correlation between signal power
        - 'plv': phase locking value - consistency of phase differences
        - 'ccorr': circular correlation coefficient - circular statistic for phase coupling
        - 'coh': coherence - normalized cross-spectrum
        - 'imaginary_coh': imaginary coherence - imaginary part of coherence (volume conduction resistant)
        - 'pli': phase lag index - asymmetry of phase difference distribution
        - 'wpli': weighted phase lag index - weighted version of PLI with improved properties
        
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved
    
    Returns
    -------
    con : np.ndarray
        Connectivity matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    Mathematical formulations for each connectivity measure:
    
    - PLV: |⟨e^(i(φₓ-φᵧ))⟩|
      Measures consistency of phase differences across time
      
    - Envelope correlation: corr(env(x), env(y))
      Pearson correlation between signal envelopes
      
    - Coherence: |⟨XY*⟩|²/(⟨|X|²⟩⟨|Y|²⟩)
      Normalized cross-spectrum
      
    - Imaginary coherence: |Im(⟨XY*⟩)|/√(⟨|X|²⟩⟨|Y|²⟩)
      Takes only imaginary part which is less affected by volume conduction
      
    - PLI: |⟨sign(Im(XY*))⟩|
      Quantifies asymmetry in phase difference distribution
      
    - wPLI: |⟨|Im(XY*)|sign(Im(XY*))⟩|/⟨|Im(XY*)|⟩
      Weighted version that downweights phase differences near 0 or π
    
    Raises
    ------
    ValueError
        If an unsupported connectivity metric is specified
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
        mu_angle = circmean(angle, high=np.pi, low=-np.pi, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
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
        raise ValueError('Metric type not supported.')

    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    if epochs_average:
        con = np.nanmean(con, axis=1)

    return con


def compute_conn_mvar(complex_signal: np.ndarray, mvar_params: dict, 
                     ica_params: dict, measure_params: dict, 
                     check_stability: bool = True) -> np.ndarray:
    """
    Computes connectivity measures based on multivariate autoregressive (MVAR) modeling.
    
    This function fits MVAR models to the data and computes directed connectivity 
    measures based on those models, with optional ICA for source separation.
    
    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (2, n_epochs, n_channels, n_freq_bins, n_times)
        
    mvar_params : dict
        Dictionary of MVAR model parameters with keys:
        - 'mvar_order': int, order of the MVAR model
        - 'fitting_method': str, method for fitting the MVAR model
        - 'delta': float, regularization parameter
        
    ica_params : dict
        Dictionary of ICA parameters with keys:
        - 'method': str, ICA algorithm to use
        - 'random_state': int, random seed for reproducibility
        
    measure_params : dict
        Dictionary defining the connectivity measure to compute with keys:
        - 'name': str, name of the connectivity measure (e.g., 'PDC', 'DTF')
        - 'n_fft': int, number of FFT points
        
    check_stability : bool, optional
        Whether to verify the stability of the MVAR model (default=True)
        If True, the function will check if the model is stable and prompt for 
        continuation or epoch merging if necessary.
    
    Returns
    -------
    connectivity : np.ndarray
        Connectivity measure matrix with shape:
        (n_epochs, n_freq, n_channels, n_channels, n_fft)
        or (1, n_freq, n_channels, n_channels, n_fft) if epochs are merged
    
    Notes
    -----
    MVAR-based connectivity measures provide information about directed (causal) 
    relationships between signals, unlike most of the other connectivity metrics 
    implemented in this module.
    
    The function relies on the MVARICA approach:
    1. Apply ICA to separate independent sources
    2. Fit an MVAR model to the sources
    3. Compute connectivity measures in the source space
    4. Back-project to the original signal space
    
    For stability, the number of time samples should be substantially larger than:
    mvar_order × n_channels^2
    
    Warnings
    --------
    This function is computationally intensive, especially for high model orders
    and large numbers of channels.
    
    If the MVAR model is unstable, the function will offer to merge epochs to 
    increase the sample size for more stable estimation.
    
    References
    ----------
    Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a new 
    concept in neural structure determination. Biological cybernetics, 84(6), 463-474.
    
    Examples
    --------
    >>> mvar_params = {'mvar_order': 5, 'fitting_method': 'least_squares', 'delta': 0.0}
    >>> ica_params = {'method': 'extended-infomax', 'random_state': 42}
    >>> measure_params = {'name': 'PDC', 'n_fft': 512}
    >>> conn = compute_conn_mvar(complex_signal, mvar_params, ica_params, measure_params)
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

                    return np.asarray(np.real(aux_3), dtype=d_type)

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


def compute_single_freq(data: np.ndarray, sampling_rate: int, freq_range: List[float]) -> np.ndarray:
    """
    Computes analytic signals for each frequency in the specified range using multitaper method.
    
    This function calculates complex-valued time-frequency representations for each 
    integer frequency in the given range.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data from two participants with shape (2, n_epochs, n_channels, n_times)
        
    sampling_rate : int
        Sampling rate of the EEG data in Hz
        
    freq_range : List[float]
        A list [fmin, fmax] specifying the frequency range to analyze.
        Every integer frequency from fmin to fmax will be included.
    
    Returns
    -------
    complex_signal : np.ndarray
        Complex-valued analytic signals with shape:
        (2, n_epochs, n_channels, n_tapers, n_frequencies, n_times)
    
    Notes
    -----
    This function uses MNE's implementation of the multitaper method with 4 cycles
    per frequency, which provides a good balance between time and frequency resolution.
    
    Time-frequency decomposition is calculated for each participant separately.
    
    Examples
    --------
    >>> complex_tf = compute_single_freq(data, sampling_rate=256, freq_range=[8, 30])
    >>> print(f"Shape: {complex_tf.shape}")  # (2, n_epochs, n_channels, n_tapers, 23, n_times)
    """

    complex_signal = np.array([mne.time_frequency.tfr_array_multitaper(data[participant], sfreq=sampling_rate,
                                                                       freqs=np.arange(
                                                                           freq_range[0], freq_range[1], 1),
                                                                       n_cycles=4, zero_mean=False, use_fft=True,
                                                                       decim=1,
                                                                       output='complex')
                               for participant in range(2)])

    return complex_signal


def compute_freq_bands(data: np.ndarray, sampling_rate: int, freq_bands: dict, 
                      filter_signal: bool = True, **filter_options) -> np.ndarray:
    """
    Computes analytic signals for specified frequency bands using FIR filtering and Hilbert transform.
    
    This function performs bandpass filtering for each defined frequency band, then
    applies the Hilbert transform to obtain complex-valued analytic signals.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data from two participants with shape (2, n_epochs, n_channels, n_times)
        
    sampling_rate : int
        Sampling rate of the EEG data in Hz
        
    freq_bands : dict
        Dictionary defining frequency bands: {'band_name': [fmin, fmax], ...}
        e.g., {'alpha': [8, 12], 'beta': [12, 20]}
        
    filter_signal : bool, optional
        Whether to apply bandpass filtering (default=True)
        If False, the Hilbert transform is applied directly to the input signals
        
    **filter_options
        Additional arguments for mne.filter.filter_data, such as:
        - filter_length: Length of the FIR filter in samples or 's' for seconds
        - l_trans_bandwidth: Width of the transition band at the low cut-off frequency
        - h_trans_bandwidth: Width of the transition band at the high cut-off frequency
    
    Returns
    -------
    complex_signal : np.ndarray
        Complex-valued analytic signals with shape:
        (2, n_epochs, n_channels, n_freq_bands, n_times)
        where n_freq_bands corresponds to the number of frequency bands in freq_bands
    
    Notes
    -----
    Unlike compute_single_freq which uses the multitaper method, this function:
    1. Applies a bandpass filter to isolate the specified frequency band
    2. Uses the Hilbert transform to compute the analytic signal
    
    This approach may be computationally more efficient when analyzing broader 
    frequency bands rather than individual frequencies.
    
    The order of frequency bands in the output corresponds to their order in the 
    freq_bands dictionary.
    
    Examples
    --------
    >>> bands = {'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30]}
    >>> complex_signals = compute_freq_bands(data, 256, bands, 
    ...                                     filter_length='auto', l_trans_bandwidth=1)
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


def compute_nmPLV(data: np.ndarray, sampling_rate: int, 
                 freq_range1: List[float], freq_range2: List[float], 
                 **filter_options) -> np.ndarray:
    """
    Computes n:m Phase-Locking Value for cross-frequency coupling between participants.
    
    This function calculates the phase synchronization between two participants
    when their brain oscillations operate at different frequency ranges with
    an integer ratio relationship (n:m).
    
    Parameters
    ----------
    data : np.ndarray
        EEG data from two participants with shape (2, n_epochs, n_channels, n_times)
        
    sampling_rate : int
        Sampling rate of the EEG data in Hz
        
    freq_range1 : List[float]
        Frequency range [fmin, fmax] for participant 1
        
    freq_range2 : List[float]
        Frequency range [fmin, fmax] for participant 2
        
    **filter_options
        Additional arguments for the underlying filtering functions
    
    Returns
    -------
    con : np.ndarray
        n:m PLV connectivity matrix with shape (n_freq, 2*n_channels, 2*n_channels)
    
    Notes
    -----
    n:m Phase Locking Value measures synchronization between oscillations at 
    different frequencies when there exists an n:m ratio between them. For example, 
    if participant 1 shows activity at 10 Hz and participant 2 at 20 Hz, this would
    be a 1:2 relationship.
    
    The function:
    1. Computes the ratio between the mean frequencies of the two ranges
    2. Applies appropriate phase multiplication to account for the frequency ratio
    3. Calculates the phase locking value between the adjusted phases
    
    This measure is useful for studying cross-frequency coupling between brains,
    where different participants might exhibit coordination at different frequency bands.
    
    References
    ----------
    Palva, J. M., Palva, S., & Kaila, K. (2005). Phase synchrony among neuronal
    oscillations in the human cortex. Journal of Neuroscience, 25(15), 3962-3972.
    
    Examples
    --------
    >>> # Computing 1:2 PLV between alpha (participant 1) and beta (participant 2)
    >>> nm_plv = compute_nmPLV(data, sampling_rate=256, 
    ...                        freq_range1=[8, 12], freq_range2=[16, 24])
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


def xwt(sig1: mne.Epochs, sig2: mne.Epochs, freqs: Union[int, np.ndarray], 
       n_cycles=5.0, mode: str = "xwt") -> np.ndarray:
    """
    Performs cross-wavelet transform or wavelet coherence analysis between two signals.
    
    This function computes time-frequency representations of two signals and their
    cross-spectrum using continuous wavelet transform.
    
    Parameters
    ----------
    sig1 : mne.Epochs
        EEG data of first participant
        
    sig2 : mne.Epochs
        EEG data of second participant
        
    freqs : Union[int, np.ndarray]
        Frequencies of interest in Hz
        
    n_cycles : float, optional
        Number of cycles in the Morlet wavelet (default=5.0)
        Controls the time-frequency resolution trade-off
        
    mode : str, optional
        Type of analysis to perform (default="xwt"). Options:
        - 'power': absolute value of cross-wavelet transform
        - 'phase': phase angles of cross-wavelet transform
        - 'xwt': raw cross-wavelet transform (complex-valued)
        - 'wtc': wavelet coherence
    
    Returns
    -------
    data : np.ndarray
        Result of the wavelet analysis with shape:
        (n_chans1, n_chans2, n_epochs, n_freqs, n_samples)
    
    Notes
    -----
    This function provides different types of wavelet-based analyses:
    
    - Cross-wavelet transform (XWT): Reveals common power and relative phase between
      two time series in time-frequency space
      
    - Wavelet coherence (WTC): Measures the correlation between two signals in the
      time-frequency domain normalized between 0 and 1
    
    Unlike Fourier-based methods, wavelet analysis maintains time resolution,
    allowing for the study of non-stationary signals and transient relationships.
    
    The function automatically checks that both signals have the same sampling rate,
    number of epochs, channels, and samples.
    
    References
    ----------
    Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross 
    wavelet transform and wavelet coherence to geophysical time series. 
    Nonlinear processes in geophysics, 11(5/6), 561-566.
    
    Maraun, D., & Kurths, J. (2004). Cross wavelet analysis: significance testing 
    and pitfalls. Nonlinear Processes in Geophysics, 11(4), 505-514.
    
    Examples
    --------
    >>> # Computing wavelet coherence between two EEG signals
    >>> coherence = xwt(subj1_epochs, subj2_epochs, freqs=np.arange(4, 40, 1), 
    ...                 n_cycles=7, mode='wtc')
    """
    
    # Set parameters for the output
    n_freqs = len(freqs)
    sfreq = sig1.info['sfreq']
    assert sig1.info['sfreq'] == sig2.info['sfreq'], "Sig1 et sig2 should have the same sfreq value."

    n_epochs1, n_chans1, n_samples1 = sig1.get_data(copy=False).shape
    n_epochs2, n_chans2, n_samples2 = sig2.get_data(copy=False).shape

    assert n_epochs1 == n_epochs2, "n_epochs1 and n_epochs2 should have the same number of epochs."
    assert n_chans1 == n_chans2, "n_chans1 and n_chans2 should have the same number of channels."
    assert n_samples1 == n_samples2, "n_samples1 and n_samples2 should have the same number of samples."

    cross_sigs = np.zeros((n_chans1, n_chans2, n_epochs1, n_freqs, n_samples1), dtype=complex) * np.nan
    wtcs = np.zeros((n_chans1, n_chans2, n_epochs1, n_freqs, n_samples1), dtype=complex) * np.nan

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
            wtc = (abs_coh - np.min(abs_coh)) / (np.max(abs_coh) - np.min(abs_coh))
            wtcs[ind1, ind2, :, :, :] = wtc

    if mode == 'power':
        data = np.abs(cross_sigs)
    elif mode == 'phase':
        data = np.angle(cross_sigs)
    elif mode == 'xwt':
        data = cross_sigs
    elif mode == 'wtc':
        data = wtcs 
    else:
        data = 'Please specify a valid mode: power, phase, xwt, or wtc.'
        print(data)
    return data


# helper function
def _multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate efficiently.
    
    This helper function performs matrix multiplication between complex arrays
    represented by their real and imaginary parts, collapsing the last dimension.
    
    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
        
    imag : np.ndarray
        Imaginary part of the complex array
        
    transpose_axes : tuple
        Axes to transpose for matrix multiplication
    
    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate
    
    Notes
    -----
    This function implements the formula:
    product = (real × real.T + imag × imag.T) - i(real × imag.T - imag × real.T)
    
    Using einsum for efficient computation without explicitly creating complex arrays.
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
    Computes the product of a complex array and its conjugate without collapsing time dimension.
    
    Similar to _multiply_conjugate, but preserves the time dimension, which is
    needed for certain connectivity metrics like wPLI.
    
    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
        
    imag : np.ndarray
        Imaginary part of the complex array
        
    transpose_axes : tuple
        Axes to transpose for matrix multiplication
    
    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate with time dimension preserved
    
    Notes
    -----
    This function uses a different einsum formula than _multiply_conjugate:
    'jilm,jimk->jilkm' instead of 'jilm,jimk->jilk'
    
    This preserves the time dimension (m) in the output, which is necessary for 
    computing metrics that require individual time point values rather than 
    time-averaged products.
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))
    
    return product