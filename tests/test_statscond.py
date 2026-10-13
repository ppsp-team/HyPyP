"""statsCond is a one-sample sign-flip test with max-T correction, nothing more.

The reference enumerates every sign flip of ten observations by hand and
corrects by the maximum absolute t over channels, which is what MNE's
permutation_t_test does with n_permutations="all". The p-values must agree
exactly, and no second correction may be applied on top.
"""

import itertools

import mne
import numpy as np
import pytest
import scipy.stats
from numpy.testing import assert_allclose, assert_array_equal

from hypyp import stats

NAMES = ["A", "B", "C", "D"]
N_SUBJECTS = 10


@pytest.fixture
def epochs():
    info = mne.create_info(NAMES, 100.0, "eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    return mne.EpochsArray(np.zeros((2, len(NAMES), 10)), info, verbose=False)


def contrast(seed=0, lift=1.5):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_SUBJECTS, len(NAMES), 3))
    data[:, 0, :] += lift
    return data


def one_sample_t(power):
    return power.mean(0) / (power.std(0, ddof=1) / np.sqrt(len(power)))


def exact_max_t_p_values(power):
    """MNE's exact two-tailed convention: sign patterns are counted up to a
    global flip (a pattern and its negation give the same |t|), the observed one
    included, over 2 ** (n - 1) - 1."""
    n = len(power)
    t_obs = one_sample_t(power)
    max_t = []
    for signs in itertools.product([-1, 1], repeat=n):
        max_t.append(np.abs(one_sample_t(np.array(signs)[:, None] * power)).max())
    max_t = np.array(max_t)
    p_values = [(max_t >= abs(t)).sum() / 2 / (2 ** (n - 1) - 1) for t in t_obs]
    return np.minimum(p_values, 1.0)


def test_p_values_are_the_exact_max_t_sign_flip_p_values(epochs):
    data = contrast()
    result = stats.statsCond(data, epochs, "all", 0.05)
    power = data.mean(axis=2)
    assert_allclose(result.T_obs, scipy.stats.ttest_1samp(power, 0.0)[0])
    assert_allclose(result.p_values, exact_max_t_p_values(power))


def test_no_second_correction_is_applied(epochs):
    data = contrast()
    result = stats.statsCond(data, epochs, "all", 0.05)
    assert_array_equal(result.adj_p[1], result.p_values)
    assert_array_equal(result.adj_p[0], result.p_values <= 0.05)
    assert_array_equal(
        result.T_obs_plot, np.where(result.p_values <= 0.05, result.T_obs, 0)
    )
    assert result.adj_p[0][0] and not result.adj_p[0][1:].any()


def test_all_positive_input_is_flagged(epochs):
    power = np.random.default_rng(1).uniform(1.0, 2.0, (N_SUBJECTS, len(NAMES), 3))
    with pytest.warns(UserWarning, match="contrast"):
        stats.statsCond(power, epochs, 200, 0.05)


def test_seed_makes_the_permutation_reproducible(epochs):
    data = contrast(seed=2)
    first = stats.statsCond(data, epochs, 200, 0.05, seed=0)
    second = stats.statsCond(data, epochs, 200, 0.05, seed=0)
    assert_array_equal(first.H0, second.H0)
    assert_array_equal(first.p_values, second.p_values)
