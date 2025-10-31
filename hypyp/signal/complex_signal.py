from collections import OrderedDict

import mne
import numpy as np

from ..dataclasses.freq_band import FreqBands

from ..analyses import (
    compute_freq_bands,
)

DEFAULT_FREQ_BANDS = FreqBands({
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
})

class ComplexSignal():
    def __init__(
            self,
            epos: list[mne.Epochs],
            sfreq: float,
            freq_bands: FreqBands = DEFAULT_FREQ_BANDS,
            **compute_freq_bands_kwargs,
        ):
        self.data = compute_freq_bands(
            np.array(epos),
            sfreq,
            freq_bands.as_dict,
            **compute_freq_bands_kwargs,
        )
        self.freq_bands = freq_bands
