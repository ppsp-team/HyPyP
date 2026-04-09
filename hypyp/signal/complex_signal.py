from collections import OrderedDict

import mne
import numpy as np

from ..dataclasses.freq_band import FreqBands, FREQ_BANDS_ALPHA_LOW_HIGH

from ..analyses import (
    compute_freq_bands,
)

class ComplexSignal():
    def __init__(
            self,
            epos: list[mne.Epochs],
            sfreq: float,
            freq_bands: FreqBands = FREQ_BANDS_ALPHA_LOW_HIGH,
            **compute_freq_bands_kwargs,
        ):

        self.data = compute_freq_bands(
            np.array(epos),
            sfreq,
            freq_bands.as_dict,
            **compute_freq_bands_kwargs,
        )
        self.freq_bands = freq_bands
