import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet import WaveletAnalyser

def test_instanciate():
    wavelet_name = 'cgaus1'
    analyser = WaveletAnalyser(wavelet_name=wavelet_name)
    assert analyser.wct_smoothing_smooth_factor != 0
    assert analyser.wct_smoothing_boxcar_size > 0
    assert analyser.wavelet_name == wavelet_name
    assert isinstance(analyser.cwt_params, dict)

def test_resolution():
    analyser = WaveletAnalyser()
    # need to be evaluated before calling .dt
    with pytest.raises(Exception):
        analyser.resolution
    
    analyser.evaluate_psi()
    assert analyser.resolution > 0
    assert analyser.resolution < 1

def test_default_domain():
    analyser = WaveletAnalyser()

    with pytest.raises(Exception):
        analyser.resolution

    psi, x = analyser.evaluate_psi()
    assert min(x) == analyser.domain[0]
    assert max(x) == analyser.domain[1]
    
def test_domain():
    lower_bound = -1
    upper_bound = 1
    analyser = WaveletAnalyser(lower_bound=lower_bound, upper_bound=upper_bound)

    psi, x = analyser.evaluate_psi()
    assert min(x) == lower_bound
    assert max(x) == upper_bound
    
def test_precision():
    precision = 12
    analyser = WaveletAnalyser(precision=precision)
    _, x = analyser.evaluate_psi()
    assert len(x) == 2 ** precision
    
def test_psi():
    analyser = WaveletAnalyser(wavelet_name='cmor2,1')
    psi, x = analyser.evaluate_psi()
    assert len(x) == len(psi)
    assert len(x) == 2 ** analyser.precision

    assert np.sum(np.real(psi) * analyser.resolution) != 0
    assert np.sum(np.imag(psi) * analyser.resolution) != 0
    assert np.sum(np.abs(psi) * analyser.resolution) == pytest.approx(1)

def test_wct():
    analyser = WaveletAnalyser()
    signal1 = SynteticSignal(tmax=300).add_noise()
    signal2 = SynteticSignal(tmax=300).add_noise()
    res = analyser.wct(signal1.y, signal2.y, signal1.period)

def test_pycwt_implementation():
    analyser = WaveletAnalyser('cmor_pycwt')
    psi, x = analyser.evaluate_psi()
    assert min(x) == analyser.lower_bound
    assert max(x) == analyser.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal(tmax=300).add_noise()
    signal2 = SynteticSignal(tmax=300).add_noise()
    res = analyser.wct(signal1.y, signal2.y, signal1.period)


    