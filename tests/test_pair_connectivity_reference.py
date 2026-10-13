"""compute_sync end to end on narrowband signals whose coupling is known.

Ten epochs of two channels per participant, 8 to 12 Hz noise, analytic signal
by compute_freq_bands with the filter off. Identical participants must give
1 on the amplitude and phase measures and 0 on the lag-sensitive ones; a
quarter-turn shift (the Hilbert transform) must give 1 on the lag-sensitive
ones; independent participants must stay near the chance level of ten
4-second epochs; and a single coupled pair must land in the inter-brain
block at [i, j] = (participant 1 channel i, participant 2 channel j).
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import hilbert

from hypyp import analyses

SFREQ = 256
N_EPOCHS = 10
N_CHANNELS = 2
TIMES = np.arange(int(SFREQ * 4.0)) / SFREQ
MODES = [
    "plv",
    "ccorr",
    "accorr",
    "coh",
    "imcoh",
    "pli",
    "wpli",
    "envelope_corr",
    "pow_corr",
]


def narrowband(rng, n):
    noise = rng.standard_normal((n, TIMES.size))
    return mne.filter.filter_data(noise, SFREQ, 8, 12, verbose=False)


def build(case, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((2, N_EPOCHS, N_CHANNELS, TIMES.size))
    for epoch in range(N_EPOCHS):
        base = narrowband(rng, N_CHANNELS)
        other = narrowband(rng, N_CHANNELS)
        if case == "identical":
            data[0, epoch], data[1, epoch] = base, base
        elif case == "shift90":
            data[0, epoch], data[1, epoch] = base, np.imag(hilbert(base))
        elif case == "independent":
            data[0, epoch], data[1, epoch] = base, other
        elif case == "coupled pair":
            data[0, epoch], data[1, epoch] = base, other
            data[1, epoch, 1] = base[0]
    return data


def inter_brain(case, mode):
    complex_signal = analyses.compute_freq_bands(
        build(case), SFREQ, {"alpha": [8, 12]}, filter_signal=False
    )
    return analyses.compute_sync(complex_signal, mode)[0, :N_CHANNELS, N_CHANNELS:]


@pytest.mark.parametrize("mode", MODES)
def test_identical_participants(mode):
    block = inter_brain("identical", mode)
    expected = 0.0 if mode in ("imcoh", "pli", "wpli") else 1.0
    assert_allclose(np.diag(block), expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["plv", "coh", "imcoh", "pli", "wpli"])
def test_quarter_turn_shift(mode):
    block = inter_brain("shift90", mode)
    assert_allclose(np.diag(block), 1.0, atol=2e-3)


@pytest.mark.parametrize("mode", MODES)
def test_independent_participants_stay_near_chance(mode):
    assert inter_brain("independent", mode).max() < 0.5


def test_a_coupled_pair_lands_in_its_cell_of_the_inter_brain_block():
    block = inter_brain("coupled pair", "plv")
    assert block[0, 1] > 0.999  # participant 1 channel 0 with participant 2 channel 1
    assert np.delete(block.ravel(), 1).max() < 0.5
