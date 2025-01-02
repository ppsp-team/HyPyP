import pytest

import numpy as np

from hypyp.signal import SynteticSignal
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
from hypyp.wavelet.wtc import WTC

def test_instanciate():
    wavelet = PywaveletsWavelet(disable_caching=True)
    signal1 = SynteticSignal().add_noise()
    signal2 = SynteticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y, label_ch1='foo', label_ch2='bar'))
    df = CoherenceDataFrame.from_wtcs([res.as_frame_row])
    assert np.all(df['channel1'] == 'foo')
    assert np.all(df['channel2'] == 'bar')

