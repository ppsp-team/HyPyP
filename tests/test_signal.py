import pytest

import numpy as np

from hypyp.signal import SynteticSignal

def test_instanciate():
    signal = SynteticSignal(tmax=300)
    assert signal is not None
    assert max(signal.x) == 300
    assert len(signal.y) == len(signal.x)
    assert np.sum(signal.y) == 0

def add_noise():
    signal = SynteticSignal(tmax=300)
    signal.add_noise()
    assert np.sum(np.abs(signal.y)) > 0

def test_sin():
    signal = SynteticSignal(tmax=300)
    signal.add_sin(freq=1)
    assert len(signal.y) == len(signal.x)
    assert signal.y[0] == 0

def test_chirp():
    signal = SynteticSignal(tmax=300)
    signal.add_chirp(1, 5)
    assert len(signal.y) == len(signal.x)
    # TODO should test the frequencies

def test_custom():
    n = 300
    signal = SynteticSignal(tmax=n, n_points=n)
    signal.add_custom(np.ones((n, )))
    assert np.sum(signal.y) == n

    

