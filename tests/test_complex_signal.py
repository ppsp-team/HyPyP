import pytest

import mne
import numpy as np

from hypyp.signal.complex_signal import ComplexSignal, DEFAULT_FREQ_BANDS
from hypyp.signal.synthetic_signal import SyntheticSignal

def test_complex_signal():
    duration = 10
    sfreq = 1000
    ch_names = ['ch1', 'ch2']
    n_ch = len(ch_names)
    signals = [SyntheticSignal(duration, duration * sfreq).add_noise().y for _ in range(n_ch * 2)]

    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=None)
    raw1 = mne.io.RawArray(signals[:n_ch], info)
    raw2 = mne.io.RawArray(signals[n_ch:], info)

    epos = [mne.make_fixed_length_epochs(raw, 2, preload=True) for raw in [raw1, raw2]]

    n_epochs = len(epos[0])
    n_freq_bands = len(DEFAULT_FREQ_BANDS)
    n_times = len(epos[0].times)

    complex_signal = ComplexSignal(epos, sfreq)
    assert complex_signal.data.shape == (2, n_epochs, n_ch, n_freq_bands, n_times)

