"""AR_local keeps every participant on the same epochs, whatever the strategy.

Synthetic recordings of twelve channels, forty one-second epochs, with one
epoch of fifty times the noise planted in one participant only. Which epochs
must survive follows from the strategy: the union drops every planted epoch,
the intersection drops none and repairs them in place. The kept events are
compared sample for sample across participants.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hypyp import prep

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz"]
SFREQ = 250
N_EPOCHS = 40
N_TIMES = 250


def make_epochs(seed, bad_epochs):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_EPOCHS, len(CHANNELS), N_TIMES)) * 1e-5
    for k in bad_epochs:
        data[k] += 5e-4 * rng.standard_normal((len(CHANNELS), N_TIMES))
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    events = np.column_stack(
        [np.arange(N_EPOCHS) * N_TIMES, np.zeros(N_EPOCHS, int), np.ones(N_EPOCHS, int)]
    )
    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, verbose=False)
    epochs.set_montage("standard_1020")
    return epochs


def kept_epochs(cleaned):
    return [epochs.events[:, 0] // N_TIMES for epochs in cleaned]


@pytest.mark.parametrize("strategy", ["union", "intersection"])
def test_participants_keep_the_same_epochs(strategy):
    cleaned, dic = prep.AR_local(
        [make_epochs(0, [5]), make_epochs(1, [9])], strategy=strategy, threshold=50.0
    )
    first, second = kept_epochs(cleaned)
    assert_array_equal(first, second)
    dropped = set(range(N_EPOCHS)) - set(first)
    assert dropped == ({5, 9} if strategy == "union" else set())
    assert dic["dyad"] == pytest.approx(len(dropped) / N_EPOCHS * 100)
    assert dic["S1"] == pytest.approx(100 / N_EPOCHS)
    assert dic["S2"] == pytest.approx(100 / N_EPOCHS)


def test_three_participants_are_handled():
    cleaned, dic = prep.AR_local(
        [make_epochs(0, []), make_epochs(1, []), make_epochs(2, [7])],
        strategy="union",
        threshold=50.0,
    )
    kept = kept_epochs(cleaned)
    for other in kept[1:]:
        assert_array_equal(kept[0], other)
    assert 7 not in set(kept[0])
    assert dic["S3"] == pytest.approx(100 / N_EPOCHS)
    assert dic["dyad"] == pytest.approx(100 / N_EPOCHS)
