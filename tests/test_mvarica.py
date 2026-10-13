"""MVAR fitting and stability on a known VAR(2), and the epoch merge.

The reference coefficients are the ones the signal was generated with; the
expected merged series follows from the construction of the input.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from hypyp import analyses
from hypyp.mvarica import MVAR

B1 = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.4], [0.0, 0.0, 0.5]])
B2 = -0.2 * np.eye(3)


def var2_signal(n_epochs=5, n_times=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_epochs, 3, n_times))
    for epoch in range(n_epochs):
        for t in range(2, n_times):
            x[epoch, :, t] = B1 @ x[epoch, :, t - 1] + B2 @ x[epoch, :, t - 2]
            x[epoch, :, t] += rng.standard_normal(3)
    return x


def explosive_signal(n_times=300, seed=1):
    rng = np.random.default_rng(seed)
    y = np.zeros((1, 2, n_times))
    for t in range(1, n_times):
        y[0, :, t] = np.array([1.2, 0.3]) * y[0, :, t - 1]
        y[0, :, t] += 1e-3 * rng.standard_normal(2)
    return y


def test_fit_recovers_the_var2_coefficients():
    model = MVAR(model_order=2).fit(var2_signal())
    assert_allclose(model.coeff[:, 0::2], B1, atol=0.05)
    assert_allclose(model.coeff[:, 1::2], B2, atol=0.05)
    assert model.stability()


def test_ridge_penalty_is_accepted():
    signal = var2_signal()
    plain = MVAR(model_order=2).fit(signal).coeff
    ridge = MVAR(model_order=2, delta=0.1).fit(signal).coeff
    assert_allclose(ridge, plain, atol=0.05)


def test_stability_runs_for_model_order_one():
    assert MVAR(model_order=1).fit(var2_signal()).stability()
    assert not MVAR(model_order=1).fit(explosive_signal()).stability()


def test_stability_detects_an_explosive_model():
    assert not MVAR(model_order=2).fit(explosive_signal()).stability()


def test_merge_epochs_concatenates_each_channel_in_time():
    # (n_epochs, n_freq, n_channels, n_times): epoch 0 holds 0..5, epoch 1 holds 6..11
    signal = np.arange(12, dtype=float).reshape(2, 1, 2, 3)
    merged = analyses._merge_epochs(signal)
    assert merged.shape == (1, 2, 6)
    assert_array_equal(merged[0, 0], [0, 1, 2, 6, 7, 8])
    assert_array_equal(merged[0, 1], [3, 4, 5, 9, 10, 11])
