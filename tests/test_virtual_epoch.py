"""The Kuramoto simulator does what its coupling matrix says.

A strong one-way coupling with a small frequency mismatch must lock the two
oscillators: the phase difference settles to a constant. A free oscillator
with no frequency spread must run at exactly the mean frequency on the
sample grid, which checks the time axis as well as the integration. Every
expected value follows from the model, not from HyPyP.
"""

import mne
import numpy as np
from numpy.testing import assert_array_equal
from scipy.signal import hilbert

from hypyp import utils

SFREQ = 250.0
N_EPOCHS = 4
N_TIMES = 500


def template(n_channels=4):
    info = mne.create_info([f"C{i}" for i in range(n_channels)], SFREQ, "eeg")
    data = np.zeros((N_EPOCHS, n_channels, N_TIMES))
    return mne.EpochsArray(data, info, verbose=False)


def unwrapped_phase(signal):
    return np.unwrap(np.angle(hilbert(signal)))


def test_coupled_oscillators_lock():
    coupling = np.zeros((4, 4))
    coupling[2, 0] = 20.0  # oscillator 0 drives oscillator 2
    simulated = utils.generate_virtual_epoch(
        template(),
        coupling,
        frequency_mean=10,
        frequency_std=0.5,
        noise_phase_level=0.0,
        noise_amplitude_level=0.0,
        rng=0,
    )
    last_epoch = simulated.get_data(copy=False)[-1]
    difference = unwrapped_phase(last_epoch[0]) - unwrapped_phase(last_epoch[2])
    inner = slice(N_TIMES // 10, -N_TIMES // 10)  # away from the Hilbert edges
    assert np.abs(np.mean(np.exp(1j * difference[inner]))) > 0.99
    assert np.std(difference[inner]) < 0.05


def test_free_oscillator_runs_at_the_mean_frequency_on_the_sample_grid():
    simulated = utils.generate_virtual_epoch(
        template(),
        np.zeros((4, 4)),
        frequency_mean=10,
        frequency_std=0.0,
        noise_phase_level=0.0,
        noise_amplitude_level=0.0,
        rng=0,
    )
    whole = simulated.get_data(copy=False).transpose(1, 0, 2).reshape(4, -1)
    inner = slice(whole.shape[1] // 10, -whole.shape[1] // 10)
    t = np.arange(whole.shape[1])[inner] / SFREQ
    for channel in whole:
        slope = np.polyfit(t, unwrapped_phase(channel)[inner], 1)[0]
        assert abs(slope / (2 * np.pi) - 10.0) < 1e-3


def test_rng_makes_the_simulation_reproducible():
    coupling = np.zeros((4, 4))
    first = utils.generate_virtual_epoch(template(), coupling, rng=1).get_data()
    second = utils.generate_virtual_epoch(template(), coupling, rng=1).get_data()
    other = utils.generate_virtual_epoch(template(), coupling, rng=2).get_data()
    assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_random_epoch_takes_an_rng():
    first = utils.generate_random_epoch(template(), mu=0, sigma=1.0, rng=3).get_data()
    second = utils.generate_random_epoch(template(), mu=0, sigma=1.0, rng=3).get_data()
    assert_array_equal(first, second)
    assert abs(first.mean()) < 0.05 and abs(first.std() - 1.0) < 0.05
