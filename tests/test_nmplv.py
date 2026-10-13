"""n:m phase locking checked against its definition and against pure tones.

The first test rebuilds the n:m PLV from the definition on the same
multitaper phases that ``compute_nmPLV`` uses, so it pins which axis gets
the powers and which axis is averaged. The others use tones whose locking is
known by construction: participant 1 at 10 Hz and participant 2 at exactly
20 Hz with twice the phase have a constant 1:2 phase difference within every
epoch, so the n:m PLV must be close to 1 at the two tone bins; at 21 Hz the
difference drifts through one cycle per second and the n:m PLV must be close
to 0 everywhere. Away from its own tone a participant reaches a bin only
through the sidelobes of the tapers, so no value is asserted there.
"""

import numpy as np
from numpy.testing import assert_allclose

from hypyp import analyses

SFREQ = 256
DURATION = 16.0
N_CHANNELS = 2
FREQ_RANGE_1 = [8, 12]
FREQ_RANGE_2 = [16, 24]
FREQS = np.arange(FREQ_RANGE_1[0], FREQ_RANGE_2[1])
TONE_BINS = [int(np.flatnonzero(FREQS == f)[0]) for f in (10, 20)]


def tone_pair(n_epochs, f2, phase_locked, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    data = np.zeros((2, n_epochs, N_CHANNELS, t.size))
    for epoch in range(n_epochs):
        phase = rng.uniform(0, 2 * np.pi)
        phase2 = 2 * phase + 0.3 if phase_locked else rng.uniform(0, 2 * np.pi)
        data[0, epoch] = np.cos(2 * np.pi * 10 * t + phase)
        data[1, epoch] = np.cos(2 * np.pi * f2 * t + phase2)
    return data + noise * rng.standard_normal(data.shape)


def nm_plv_from_definition(data, n, m):
    """Participant 1's phase to the power n against participant 2's to the power m,
    locked within each epoch, averaged over epochs."""
    tf = np.mean(
        analyses.compute_single_freq(data, SFREQ, [FREQS[0], FREQS[-1] + 1]), 3
    )
    phase = tf / np.abs(tf)  # (2, n_epochs, n_channels, n_freq, n_times)
    z = np.concatenate([phase[0] ** n, phase[1] ** m], axis=1)
    per_epoch = np.abs(np.einsum("ecft,edft->efcd", z, np.conj(z))) / z.shape[-1]
    return per_epoch.mean(axis=0)


def test_matches_the_definition_on_the_same_phases():
    data = tone_pair(3, f2=20.0, phase_locked=True, noise=0.5)
    result = analyses.compute_nmPLV(data, SFREQ, FREQ_RANGE_1, FREQ_RANGE_2)
    assert_allclose(result, nm_plv_from_definition(data, 2, 1), rtol=0, atol=1e-12)


def test_locked_tones_give_unit_inter_brain_nm_plv():
    data = tone_pair(3, f2=20.0, phase_locked=True)
    result = analyses.compute_nmPLV(data, SFREQ, FREQ_RANGE_1, FREQ_RANGE_2)
    assert result.shape == (len(FREQS), 2 * N_CHANNELS, 2 * N_CHANNELS)
    assert result[TONE_BINS][:, :N_CHANNELS, N_CHANNELS:].min() > 0.95


def test_tones_off_the_ratio_give_low_inter_brain_nm_plv():
    data = tone_pair(3, f2=21.0, phase_locked=False)
    result = analyses.compute_nmPLV(data, SFREQ, FREQ_RANGE_1, FREQ_RANGE_2)
    assert result[:, :N_CHANNELS, N_CHANNELS:].max() < 0.2


def test_single_epoch_and_single_channel_are_accepted():
    data = tone_pair(1, f2=20.0, phase_locked=True)[:, :, :1]
    result = analyses.compute_nmPLV(data, SFREQ, FREQ_RANGE_1, FREQ_RANGE_2)
    assert result.shape == (len(FREQS), 2, 2)
    assert result[TONE_BINS, 0, 1].min() > 0.95
