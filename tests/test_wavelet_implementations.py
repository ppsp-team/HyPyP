import pytest
import warnings

from hypyp.signal import SynteticSignal
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
from hypyp.wavelet.scipy_wavelet import ScipyWavelet

warnings.filterwarnings("ignore", category=DeprecationWarning)
from hypyp.wavelet.pycwt_wavelet import PycwtWavelet


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
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))

def test_pycwt():

    wavelet = PycwtWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))

def test_scipy():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    wavelet = ScipyWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    # TODO test something

    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))




