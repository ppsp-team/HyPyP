from hypyp.profiling import TimeTracker
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.signal import SyntheticSignal

foo = CoherenceDataFrame()

signal1 = SyntheticSignal()
signal2 = SyntheticSignal()
wavelet = ComplexMorletWavelet()
pair = PairSignals(signal1.x, signal1.y, signal2.y, label_s1='foo', label_s2='bar', label_task='my_task')
wtc1 = wavelet.wtc(pair)

with TimeTracker('foo'):
    wtc2 = wavelet.wtc(pair)


import matplotlib
matplotlib.use('Agg')
signal1.plot()
