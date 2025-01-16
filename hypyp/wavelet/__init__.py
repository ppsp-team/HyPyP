from .base_wavelet import BaseWavelet
from .implementations.pywavelets_wavelet import PywaveletsWavelet
from .coherence_data_frame import CoherenceDataFrame
from .cwt import CWT
from .pair_signals import PairSignals
from .wtc import WTC

__all__ = [
    'BaseWavelet',
    # Only expose PywaveletsWavelet to avoid confusion. This is the one that should always be used
    'PywaveletsWavelet',
    'CoherenceDataFrame',
    'CWT',
    'PairSignals',
    'WTC',
]
