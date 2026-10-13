"""Every metric against a scalar written from its definition, on two channels.

The references use np.angle, np.corrcoef, scipy.stats.circmean and plain
sums; none of them calls hypyp.sync. Case A is a random pair, case B a
constant lag of pi/4 on the unit circle, case C an envelope relation, case D
an identical pair; the expected values of B, C and D are closed forms. The
conventions these pin are the ones the code has: coherence is |C|, not
|C|**2; ImCoh and CCorr are returned as absolute values; wPLI is the
estimator of Vinck et al. 2011 without the debiasing term.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import circmean

from hypyp.sync import get_metric

N_TIMES = 4000
TRANSPOSE_AXES = (0, 1, 3, 2)
MODES = ["plv", "ccorr", "accorr", "coh", "imcoh", "pli", "wpli", "envcorr", "powcorr"]


def metric_value(mode, x, y):
    signal = np.stack([x, y])[None, None]  # (n_epochs, n_freq, 2 channels, n_times)
    metric = get_metric(mode)
    if mode == "accorr":
        metric.show_progress = False
    return metric.compute(signal, N_TIMES, TRANSPOSE_AXES)[0, 0, 0, 1]


def ref_plv(x, y):
    return abs(np.mean(np.exp(1j * (np.angle(x) - np.angle(y)))))


def ref_ccorr(x, y):  # Jammalamadaka & SenGupta 2001, eq. 8.2.2, absolute value
    a, b = np.angle(x), np.angle(y)
    am = circmean(a, high=np.pi, low=-np.pi)
    bm = circmean(b, high=np.pi, low=-np.pi)
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    return abs(num / np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2)))


def ref_accorr(x, y):  # Zimmermann et al. 2024, eq. 1.2, from the phases only
    a, b = np.angle(x), np.angle(y)
    md = np.angle(np.mean(np.exp(1j * (a - b))))
    ms = np.angle(np.mean(np.exp(1j * (a + b))))
    m, n = (md + ms) / 2, (ms - md) / 2
    num = np.sum(np.sin(a - m) * np.sin(b - n))
    return num / np.sqrt(np.sum(np.sin(a - m) ** 2) * np.sum(np.sin(b - n) ** 2))


def ref_coherency(x, y):  # Nolte et al. 2004, eq. 1
    return np.mean(x * np.conj(y)) / np.sqrt(
        np.mean(abs(x) ** 2) * np.mean(abs(y) ** 2)
    )


def ref_pli(x, y):  # Stam et al. 2007
    return abs(np.mean(np.sign(np.imag(x * np.conj(y)))))


def ref_wpli(x, y):  # Vinck et al. 2011, eq. 8
    im = np.imag(x * np.conj(y))
    return abs(np.mean(im)) / np.mean(abs(im))


def ref_envcorr(x, y):
    return np.corrcoef(abs(x), abs(y))[0, 1]


def ref_powcorr(x, y):
    return np.corrcoef(abs(x) ** 2, abs(y) ** 2)[0, 1]


REFERENCES = {
    "plv": ref_plv,
    "ccorr": ref_ccorr,
    "accorr": ref_accorr,
    "coh": lambda x, y: abs(ref_coherency(x, y)),
    "imcoh": lambda x, y: abs(ref_coherency(x, y).imag),
    "pli": ref_pli,
    "wpli": ref_wpli,
    "envcorr": ref_envcorr,
    "powcorr": ref_powcorr,
}


@pytest.fixture(scope="module")
def random_pair():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N_TIMES) + 1j * rng.standard_normal(N_TIMES)
    noise = rng.standard_normal(N_TIMES) + 1j * rng.standard_normal(N_TIMES)
    y = x + noise * 1j * 0.7 + 0.3 * x * 1j
    return x, y


@pytest.mark.parametrize("mode", MODES)
def test_random_pair_matches_the_definition(mode, random_pair):
    x, y = random_pair
    assert_allclose(
        metric_value(mode, x, y), REFERENCES[mode](x, y), rtol=0, atol=1e-10
    )


def test_constant_lag_of_a_quarter_turn():
    phase = np.random.default_rng(1).uniform(-np.pi, np.pi, N_TIMES)
    x = np.exp(1j * phase)
    y = np.exp(1j * (phase - np.pi / 4))
    for mode in ["plv", "ccorr", "accorr", "coh", "pli", "wpli"]:
        assert_allclose(metric_value(mode, x, y), 1.0, atol=1e-10)
    assert_allclose(metric_value("imcoh", x, y), np.sin(np.pi / 4), atol=1e-10)


def test_envelope_relation():
    rng = np.random.default_rng(2)
    phase = rng.uniform(-np.pi, np.pi, N_TIMES)
    amplitude = rng.uniform(0.5, 2.0, N_TIMES)
    x = amplitude * np.exp(1j * phase)
    y = (2 * amplitude + 1) * np.exp(1j * phase)
    assert_allclose(metric_value("envcorr", x, y), 1.0, atol=1e-10)
    assert_allclose(metric_value("powcorr", x, y), ref_powcorr(x, y), atol=1e-10)


def test_identical_channels():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N_TIMES) + 1j * rng.standard_normal(N_TIMES)
    for mode in ["plv", "ccorr", "accorr", "coh", "envcorr", "powcorr"]:
        assert_allclose(metric_value(mode, x, x), 1.0, atol=1e-10)
    for mode in ["imcoh", "pli", "wpli"]:
        assert_allclose(metric_value(mode, x, x), 0.0, atol=1e-10)


def test_coherence_is_the_modulus_not_the_square():
    rng = np.random.default_rng(4)
    n = 200_000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    signal = np.stack([x, x + noise])[None, None]
    value = get_metric("coh").compute(signal, n, TRANSPOSE_AXES)[0, 0, 0, 1]
    assert abs(value - 1 / np.sqrt(2)) < 0.01  # |C|; the magnitude squared would be 0.5
