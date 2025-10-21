from collections import OrderedDict

import mne
import numpy as np

from ..analyses import (
    compute_freq_bands,
)

FREQ_BANDS_ALPHAS = {
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
}

class ComplexSignal():
    def __init__(self, epos: list[mne.Epochs], sfreq: float, freq_bands: OrderedDict = FREQ_BANDS_ALPHAS, **compute_freq_bands_kwargs):
        self.data = compute_freq_bands(
            np.array(epos),
            sfreq,
            freq_bands,
            **compute_freq_bands_kwargs,
        )
        self.freq_bands = freq_bands
