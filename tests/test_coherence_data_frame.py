import pytest
import os
import tempfile

import numpy as np
import pandas as pd

from hypyp.signal import SyntheticSignal
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.wavelet.wtc import WTC

def test_instanciate():
    wavelet = ComplexMorletWavelet(disable_caching=True)
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y, label_ch1='foo', label_ch2='bar'))
    df = CoherenceDataFrame.from_wtc_frame_rows(res.as_frame_rows)
    assert np.all(df['channel1'] == 'foo')
    assert np.all(df['channel2'] == 'bar')

    # Make sure categorical is correct when merging data frames
    res2 = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y, label_ch1='baz', label_ch2='bar'))
    df2 = CoherenceDataFrame.from_wtc_frame_rows(res2.as_frame_rows)

    df_concat = CoherenceDataFrame.concat([df, df2])

    assert len(df_concat['channel1'].unique()) == 2
    assert len(df_concat['channel2'].unique()) == 1
    # All the 'bar' should be set to the same category
    #assert np.all(df_concat['channel2'].cat.codes == 0)


def test_save_load_feather():
    wavelet = ComplexMorletWavelet(disable_caching=True)
    signal1 = SyntheticSignal().add_noise()
    signal2 = SyntheticSignal().add_noise()
    res = wavelet.wtc(PairSignals(signal1.x, signal1.y, signal2.y, label_ch1='foo', label_ch2='bar'))
    df_before = CoherenceDataFrame.from_wtc_frame_rows(res.as_frame_rows)

    with tempfile.TemporaryDirectory(prefix='hypyp-') as tmp_path:
        feather_path = os.path.join(tmp_path, 'test.feather')
        CoherenceDataFrame.save_feather(df_before, feather_path)
        df_after = CoherenceDataFrame.from_feather(feather_path)

    assert np.all(df_after['channel1'] == 'foo')

    
