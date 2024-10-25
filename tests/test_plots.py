import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
import hypyp.plots

def test_downsampling():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=300).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    assert res.W.shape[0] == len(res.scales)
    assert res.W.shape[1] == len(signal.y)

    t = 1000
    times, coif, W, factor = hypyp.plots.downsample_in_time(res.times, res.coif, res.W)
    assert factor == len(signal.y) // t
    assert len(times) == t
    assert len(coif) == t
    assert W.shape[0] == res.W.shape[0] # no change
    assert W.shape[1] == t

def test_downsampling_low_values():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=10, n_points=500).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    t = 1000
    times, coif, W, factor = hypyp.plots.downsample_in_time(res.times, res.coif, res.W)
    assert factor == 1
    assert len(times) == len(res.times)
    assert len(coif) == len(res.coif)
    assert W.shape[0] == res.W.shape[0]
    assert W.shape[1] == res.W.shape[1]
    
def test_downsampling_threshold():
    t = 1000
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=10, n_points=t+1).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    times, coif, W, factor = hypyp.plots.downsample_in_time(res.times, res.coif, res.W)
    assert factor == 2
    expected_len = t / 2 + 1
    assert len(times) == expected_len
    assert len(coif) == expected_len
    assert W.shape[1] == expected_len
    
    
