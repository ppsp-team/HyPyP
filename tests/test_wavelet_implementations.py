import pytest
import warnings

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pycwt_wavelet import PycwtWavelet
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
from hypyp.wavelet.scipy_wavelet import ScipyWavelet


# TODO: test values with sinusoid signal

def test_pywavelets():
    wavelet = PywaveletsWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wct(signal1.y, signal2.y, signal1.period)

def test_pycwt():
    wavelet = PycwtWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wct(signal1.y, signal2.y, signal1.period)

def test_scipy():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    wavelet = ScipyWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    # TODO test something

    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wct(signal1.y, signal2.y, signal1.period)




