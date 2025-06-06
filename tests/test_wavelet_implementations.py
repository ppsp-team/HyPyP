import pytest
import warnings

from hypyp.signal import SyntheticSignal
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet

warnings.filterwarnings("ignore", category=DeprecationWarning)
from hypyp.wavelet.implementations.pycwt_wavelet import PycwtWavelet


# TODO: test values with sinusoid signal

def test_pywavelets():
    wavelet = ComplexMorletWavelet(disable_caching=True)
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))

def test_pycwt():
    if PycwtWavelet is None:
        pytest.skip("Optional dependency Pycwt is not installed")
        
    wavelet = PycwtWavelet()
    psi, x = wavelet.evaluate_psi()
    assert psi.dtype.kind == 'c'
    assert min(x) == wavelet.lower_bound
    assert max(x) == wavelet.upper_bound
    assert len(x) == len(psi)
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y))

    # make sure we do not exceed the period range
    assert res.periods[0] >= wavelet.period_range[0]
    assert res.periods[-1] <= wavelet.period_range[1]

