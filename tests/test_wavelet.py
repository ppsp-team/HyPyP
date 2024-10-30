import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

def test_instanciate():
    wavelet_name = 'cgau1'
    wavelet = PywaveletsWavelet(wavelet_name=wavelet_name)
    assert wavelet.wtc_smoothing_smooth_factor != 0
    assert wavelet.wtc_smoothing_boxcar_size > 0
    assert wavelet.wavelet_name == wavelet_name
    assert len(wavelet.psi) == 2 ** wavelet.precision
    assert isinstance(wavelet.cwt_params, dict)

def test_resolution():
    wavelet = PywaveletsWavelet(evaluate=False)
    # need to be evaluated before calling .dt
    with pytest.raises(Exception):
        wavelet.psi_dx
    
    wavelet.evaluate_psi()
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1

def test_default_domain():
    wavelet = PywaveletsWavelet()
    assert min(wavelet.psi_x) == wavelet.domain[0]
    assert max(wavelet.psi_x) == wavelet.domain[1]
    
def test_domain():
    lower_bound = -1
    upper_bound = 1
    wavelet = PywaveletsWavelet(lower_bound=lower_bound, upper_bound=upper_bound)

    assert min(wavelet.psi_x) == lower_bound
    assert max(wavelet.psi_x) == upper_bound
    assert wavelet.psi_dx > 0
    assert wavelet.psi_dx < 1
    
def test_precision():
    precision = 12
    wavelet = PywaveletsWavelet(precision=precision)
    assert len(wavelet.psi_x) == 2 ** precision
    assert len(wavelet.psi) == 2 ** precision
    
def test_psi():
    wavelet = PywaveletsWavelet(wavelet_name='cmor2,1')
    assert len(wavelet.psi_x) == len(wavelet.psi)
    assert len(wavelet.psi_x) == 2 ** wavelet.precision

    assert np.sum(np.real(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.imag(wavelet.psi) * wavelet.psi_dx) != 0
    assert np.sum(np.abs(wavelet.psi) * wavelet.psi_dx) == pytest.approx(1)

def test_cwt():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal().add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)
    assert len(res.scales) > 0
    assert len(res.scales) == len(res.frequencies)
    max_id = np.argmax(np.sum(np.abs(res.W), axis=1))
    import matplotlib.pyplot as plt
    assert res.frequencies[max_id] == pytest.approx(1, rel=0.05)

def test_wtc():
    wavelet = PywaveletsWavelet()
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(signal1.y, signal2.y, signal1.period)

    