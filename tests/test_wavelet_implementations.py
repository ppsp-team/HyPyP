import pytest

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pycwt_wavelet import PycwtWavelet
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

def test_pywavelets():
    wavelet = PywaveletsWavelet()
    psi, x = wavelet.evaluate_psi()
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal(tmax=300).add_noise()
    signal2 = SynteticSignal(tmax=300).add_noise()
    res = wavelet.wct(signal1.y, signal2.y, signal1.period)

def test_pycwt():
    wavelet = PycwtWavelet()
    psi, x = wavelet.evaluate_psi()
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal(tmax=300).add_noise()
    signal2 = SynteticSignal(tmax=300).add_noise()
    res = wavelet.wct(signal1.y, signal2.y, signal1.period)