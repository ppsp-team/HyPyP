#!/usr/bin/env python
# coding: utf-8

# # Test for simulated EEG
#
# Authors          : Guillaume Dumas
#
# Date            : 2020-07-09

from pathlib import Path
from copy import copy
from collections import OrderedDict
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import mne
from scipy.integrate import solve_ivp
from hypyp import utils, analyses
from mne.channels import find_ch_connectivity


def generate_virtual_epoch(epochs: mne.Epochs, W: np.ndarray, frequency_mean: float = 10, frequency_std: float = 0.2,
                           noise_phase_level: float = 0.005, noise_amplitude_level: float = 0.1) -> mne.Epochs:
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

    n_epo, n_chan, n_samp = epochs.get_data().shape
    sfreq = epochs.info['sfreq']
    N = int(n_chan / 2)

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

    p0 = 2 * np.pi * np.block([np.zeros(N) + np.random.rand(N) + 0.5, np.zeros(N) + np.random.rand(N) + 0.5])  # initialization
    ans = solve_ivp(fun=fp, t_span=(tv[0], tv[-1]), y0=p0, t_eval=tv)
    phi = ans['y'].T % (2 * np.pi)

    eeg = np.sin(phi) + noise_amplitude_level * np.random.randn(*phi.shape)

    simulation = epochs.copy()
    simulation._data = np.transpose(np.reshape(eeg.T, [n_chan, n_epo, n_samp]), (1, 0, 2))

    return simulation


def virtual_dyad(epochs,W, frequency_mean=10., frequency_std=0.2, noise_phase_level=0.005,
                 noise_amplitude_level=0.1):
    n_epo, n_chan, n_samp = epochs.get_data().shape
    sfreq = epochs.info['sfreq']

    Nt = n_samp * n_epo
    tmax = n_samp / sfreq * n_epo  # s
    tv = np.linspace(0., tmax, Nt)

    freq = frequency_mean + frequency_std * np.random.randn(n_chan)
    omega = 2. * np.pi * freq

    def fp(p, t):
        p = np.atleast_2d(p)
        coupling = np.squeeze((np.sin(p) * np.matmul(W, np.cos(p).T).T) - (np.cos(p) * np.matmul(W, np.sin(p).T).T))
        dotp = omega - coupling + noise_phase_level * np.random.randn(n_chan) / n_samp
        return dotp

    p0 = 2 * np.pi * np.block([np.zeros(N), np.zeros(N) + np.random.rand(N) + 0.5])

    phi = odeint(fp, p0, tv) % (2 * np.pi)
    eeg = np.sin(phi) + noise_amplitude_level * np.random.randn(*phi.shape)

    simulation = epo_real.copy()
    simulation._data = np.transpose(np.reshape(eeg.T, [n_chan, n_epo, n_samp]), (1, 0, 2))

    return simulation


# Load data
montage = mne.channels.read_custom_montage('../syncpipeline/FINS_data/enobio32.locs')
info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')
epo1 = mne.EpochsArray(data=np.empty((36, 32, 501)), info=info)
epo1.set_montage(montage)

epo2 = epo1.copy()
mne.epochs.equalize_epoch_counts([epo1, epo2])
sampling_rate = epo1.info['sfreq'] #Hz


# concatenate two datasets
epo_real = utils.merge(epoch_S1=epo1, epoch_S2=epo2)

# setting up parameters
n_chan = len(epo_real.ch_names)
# get channel locations
con, _ = find_ch_connectivity(epo1.info, 'eeg')
con = con.toarray()

N = int(n_chan/2)
A11 = 1 * np.ones((N, N))
A12 = 0 * np.ones((N, N))
A21 = 0 * np.ones((N, N))
A22 = 1 * np.ones((N, N))

# A11 = con
# A22 = con
W = np.block([[A11, A12], [A21, A22]])
W = 0.2 * W

# simulation params
frequency_mean = 10.
frequency_std = 0.2
noise_phase_level = 0.005
noise_amplitude_level = 0.1

# check simulated set
sim = generate_virtual_epoch(epochs=epo_real, frequency_mean=frequency_mean, frequency_std=frequency_std,
                             noise_phase_level=noise_phase_level, noise_amplitude_level=noise_amplitude_level, W=W)
# sim.plot(scalings=5, n_epochs=3, n_channels=62)
# plt.show()


"""
PLV
"""
modes = ['plv', 'ccorr', 'coh', 'imaginary_coh', 'envelope_corr', 'pow_corr']

# generate 20 simulated datasets, and average the results

for mode in modes:
    cons = []
    for i in range(20):
        sim = generate_virtual_epoch(epochs=epo_real, frequency_mean=frequency_mean, frequency_std=frequency_std,
                                     noise_phase_level=noise_phase_level,
                                     noise_amplitude_level=noise_amplitude_level, W=W)
        freq_bands = {'Alpha-Low': [8, 12]}
        connectivity = analyses.pair_connectivity(data=[sim.get_data()[:,0:32,:], sim.get_data()[:,32:,:]],
                                   sampling_rate=sampling_rate, frequencies=freq_bands, mode=mode)  # data.shape = (2, n_epochs, n_channels, n_times).
        cons.append(connectivity[0])
    plt.figure()
    plt.imshow(np.nanmean(np.array(cons), axis=0))
    plt.title(mode)

