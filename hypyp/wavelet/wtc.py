import numpy as np
import pandas as pd


from .pair_signals import PairSignals
from .coherence_data_frame import CoherenceDataFrame
from ..plots import plot_wtc
from ..utils import downsample_in_time

class WTC:
    def __init__(self, wtc, times, scales, periods, coi, pair: PairSignals, sig=None):
        self.wtc = wtc
        self.times = times
        self.scales = scales
        self.periods = periods
        self.frequencies = 1 / periods
        self.coi = coi
        self.coif = 1 / coi
        self.task = pair.task
        self.epoch = pair.epoch
        self.section = pair.section
        self.label = pair.label
        self.label_subject1 = pair.label_s1
        self.label_subject2 = pair.label_s2
        self.roi1 = pair.roi1
        self.roi2 = pair.roi2
        self.ch_name1 = pair.ch_name1
        self.ch_name2 = pair.ch_name2
        self.label_dyad = pair.label_dyad

        # These will not change when we downsample
        dt = (times[1] - times[0])
        self.sfreq = 1 / dt
        self.nyquist = self.sfreq / 2

        self.wtc_masked: np.ma.MaskedArray
        self.coherence_metric: float

        self.coherence_p_value = None
        self.coherence_t_stat = None
        self.sig = sig

        self.compute_coherence_in_coi()
    
    def compute_coherence_in_coi(self):
        mask = self.frequencies[:, np.newaxis] < self.coif
        self.wtc_masked = np.ma.masked_array(self.wtc, mask)

        # TODO maybe we should weight our average because we have more values at higher frequencies than lower frequencies, due to the coi
        coherence = np.mean(self.wtc_masked)
        if np.ma.is_masked(coherence):
            coherence = np.nan
        elif not np.isfinite(coherence):
            coherence = np.nan
        self.coherence_metric = coherence
    
    def downsample_in_time(self, bins):
        self.times, self.wtc, self.coi, self.coif, _factor = downsample_in_time(self.times, self.wtc, self.coi, self.coif, bins=bins)
        # must recompute region of interest
        self.compute_coherence_in_coi()
    
    @property
    def as_frame_row(self) -> list:
        # IMPORTANT: must match the ordering of COHERENCE_FRAME_COLUMNS
        return [
            self.label_dyad,
            self.label_subject1 == self.label_subject2,
            self.task,
            self.epoch,
            self.section,
            self.label_subject1,
            self.label_subject2,
            self.roi1,
            self.roi2,
            self.ch_name1,
            self.ch_name2,
            self.coherence_metric,
        ]
    
    def to_frame(self) -> CoherenceDataFrame:
        df = CoherenceDataFrame.from_wtcs([self.as_frame_row])
        return df

    #
    # Plots
    #
    def plot(self, **kwargs):
        return plot_wtc(self.wtc, self.times, self.frequencies, self.coi, self.sfreq, self.sig, **kwargs)

