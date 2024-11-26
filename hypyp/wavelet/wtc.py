import numpy as np
import pandas as pd

from .pair_signals import PairSignals
from ..plots import plot_wavelet_coherence
from ..utils import downsample_in_time

FRAME_COLUMNS = ['dyad',
                 'is_intra',
                 'task',
                 'subject1',
                 'subject2',
                 'roi1',
                 'roi2',
                 'channel1',
                 'channel2',
                 'coherence']

class WTC:
    def __init__(self, wtc, times, scales, frequencies, coif, pair: PairSignals, sig=None, tracer=None):
        self.wtc = wtc
        # TODO: compute Region of Interest, something like this:
        #roi = wtc * (wtc > coif[np.newaxis, :]).astype(int)
        self.times = times
        self.scales = scales
        self.frequencies = frequencies
        self.coi = 1 / coif
        self.coif = coif
        self.task = pair.task
        self.label = pair.label
        self.label_subject1 = pair.label_s1
        self.label_subject2 = pair.label_s2
        self.roi1 = pair.roi1
        self.roi2 = pair.roi2
        self.ch_name1 = pair.ch_name1
        self.ch_name2 = pair.ch_name2
        self.label_dyad = pair.label_dyad

        self.tracer = tracer

        self.wtc_roi: np.ma.MaskedArray
        self.coherence_metric: float

        self.coherence_p_value = None
        self.coherence_t_stat = None
        self.sig = sig

        self.compute_roi()
    
    def compute_roi(self):
        mask = self.frequencies[:, np.newaxis] < self.coif
        self.wtc_roi = np.ma.masked_array(self.wtc, mask)
        self.coherence_metric = np.mean(self.wtc_roi) # TODO this is just a PoC
    
    def downsample_in_time(self, bins):
        self.times, self.wtc, self.coi, self.coif, _factor = downsample_in_time(self.times, self.wtc, self.coi, self.coif, bins=bins)
        # must recompute region of interest
        self.compute_roi()
    
    @property
    def as_frame_row(self) -> list:
        return [
            self.label_dyad,
            self.label_subject1 == self.label_subject2,
            self.task,
            self.label_subject1,
            self.label_subject2,
            self.roi1,
            self.roi2,
            self.ch_name1,
            self.ch_name2,
            self.coherence_metric,
        ]
    
    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([self.as_frame_row], columns=FRAME_COLUMNS)
        return df


    def plot(self, **kwargs):
        return plot_wavelet_coherence(self.wtc_roi, self.times, self.frequencies, self.coif, self.sig, **kwargs)

