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
from mne.channels import find_ch_adjacency


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
#Loading datasets (see MNE functions mne.io.read_raw_format), convert them to MNE Epochs.
#In our example, we load Epochs directly from EEG dataset in the fiff format

epo1 = mne.read_epochs(
    Path('../data/participant1-epo.fif').resolve(),
    preload=True,
)

epo2 = mne.read_epochs(
    Path('../data/participant2-epo.fif').resolve(),
    preload=True,
)

#Since our example dataset was not initially dedicated to hyperscanning, 
# we need to equalize the number of epochs between our two participants.

mne.epochs.equalize_epoch_counts([epo1, epo2])

#Specify sampling frequency
sampling_rate = epo1.info['sfreq'] #Hz

# Generate random epochs
epo_real = utils.merge(epoch_S1=epo1, epoch_S2=epo2)
epo_rnd = utils.generate_random_epoch(epoch=epo_real, mu=0.0, sigma=2.0)
n_epo, n_chan, n_samp = epo_real.get_data().shape
sfreq = epo_real.info['sfreq']

# Generate coupled oscillators

frequency_mean = 10.  # Hz
frequency_std = 0.2 # Hz

noise_phase_level = 0.005 / n_samp
noise_amplitude_level = 0.

N = int(n_chan/2)
A11 = 1 * np.ones((N, N))
A12 = 0 * np.ones((N, N))
A21 = 0 * np.ones((N, N))
A22 = 1 * np.ones((N, N))
W = np.block([[A11, A12], [A21, A22]])
W = 0.2 * W
plt.matshow(W)

Nt = n_samp * n_epo
tmax = n_samp / sfreq * n_epo  # s
tv = np.linspace(0., tmax, Nt)

freq = frequency_mean + frequency_std * np.random.randn(n_chan)
omega = 2. * np.pi * freq

def fp(p, t):
    p = np.atleast_2d(p)
    coupling = np.squeeze((np.sin(p) * np.matmul(W, np.cos(p).T).T) - (np.cos(p) * np.matmul(W, np.sin(p).T).T))
    dotp = omega - coupling + noise_phase_level * np.random.randn(n_chan)
    return dotp

%%time
p0 = 2 * np.pi * np.block([np.zeros(N), np.zeros(N) + np.random.rand(N) + 0.5])
Phi = odeint(fp, p0, tv) % (2*np.pi)

plt.figure(figsize=(20, 10))
plt.subplot(2,1,1)
plt.imshow(Phi[:n_samp, :].T,interpolation='none', cmap='hsv')
plt.subplot(2,1,2)
plt.imshow(Phi[-n_samp:, :].T,interpolation='none', cmap='hsv')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(2,2,1)
plt.plot(tv[:n_samp], np.squeeze(epo_real[0]._data.T), "-")
plt.xlabel("Time, $t$ [s]")
plt.ylabel("Amplitude, $x$ [m]")
plt.title('Original EEG')
plt.subplot(2,2,2)
plt.plot(tv[:n_samp], np.squeeze(epo_rnd[0]._data).T, "-")
plt.xlabel("Time, $t$ [s]")
plt.ylabel("Amplitude, $x$ [m]")
plt.title('Random signal')
plt.subplot(2,2,3)
plt.plot(tv[:n_samp], np.sin(Phi[:n_samp, :]), "-")
plt.xlabel("Time, $t$ [s]")
plt.ylabel("Amplitude, $x$ [m]")
plt.title('Start of the simulation')
plt.subplot(2,2,4)
plt.plot(tv[-n_samp:], np.sin(Phi[-n_samp:, :]), "-")
plt.xlabel("Time, $t$ [s]")
plt.ylabel("Amplitude, $x$ [m]")
plt.title('End of the simulation')
plt.show()


N = int(n_chan/2)
A11 = 1 * np.ones((N, N))
A12 = 0 * np.ones((N, N))
A21 = 0 * np.ones((N, N))
A22 = 1 * np.ones((N, N))
W = np.block([[A11, A12], [A21, A22]])
W = 0.2 * W

sim = virtual_dyad(epochs = epo_real, frequency_mean = 10., frequency_std = 0.2, noise_phase_level = 0.005, noise_amplitude_level = 0.1, W = W)
sim.plot(scalings=5, n_epochs=3, n_channels=62)
plt.show()




