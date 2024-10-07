import pytest

import matplotlib.pyplot as plt

from hypyp.signal import SynteticSignal

def test_instanciate():
    signal = SynteticSignal(tmax=300)
    assert signal is not None
    assert max(signal.x) == 300

    signal.gen_sin(freq=1)
    assert len(signal.y) == len(signal.x)
    assert signal.y[0] == 0
