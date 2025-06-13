from .base_wavelet import BaseWavelet
from .implementations.pywavelets_wavelet import ComplexMorletWavelet, ComplexGaussianWavelet
from .coherence_data_frame import CoherenceDataFrame
from .cwt import CWT
from .pair_signals import PairSignals
from .wtc import WTC

__all__ = [
    'BaseWavelet',
    # Only expose wavelets from pywavelets_wavelet to avoid confusion. This is the default implementation.
    'ComplexMorletWavelet',
    'ComplexGaussianWavelet',
    'CoherenceDataFrame',
    'CWT',
    'PairSignals',
    'WTC',
]
