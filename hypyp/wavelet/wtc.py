import numpy as np
import pandas as pd


from .pair_signals import PairSignals
from .coherence_data_frame import CoherenceDataFrame
from ..plots import plot_wtc
from ..utils import downsample_in_time

class WTC:
    def __init__(self, wtc, times, scales, periods, coi, pair: PairSignals):
        self.wtc = wtc

        self.times = times
        self.scales = scales
        self.periods = periods
        self.frequencies = 1 / periods

        self.coi = coi
        self.coif = 1 / coi

        self.is_intra = pair.is_intra
        self.is_shuffle = pair.is_shuffle

        self.task = pair.task
        self.epoch = pair.epoch
        self.section = pair.section

        self.label_dyad = pair.label_dyad
        self.label_pair = pair.label
        self.label_s1 = pair.label_s1
        self.label_s2 = pair.label_s2
        self.label_ch1 = pair.label_ch1
        self.label_ch2 = pair.label_ch2
        self.label_roi1 = pair.label_roi1
        self.label_roi2 = pair.label_roi2

        # These will not change when we downsample
        dt = (times[1] - times[0])
        self.dt = dt
        self.sfreq = 1 / dt
        self.nyquist = self.sfreq / 2

        self.wtc_masked: np.ma.MaskedArray
        self.coherence_metric: float

        self.compute_coherence_in_coi()
    
    # TODO split in time segments here and make sure we weight our values correctly given time and frequencies
    def compute_coherence_in_coi(self):
        mask = self.periods[:, np.newaxis] > self.coi
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
            self.is_intra,
            self.is_shuffle,
            self.label_s1,
            self.label_s2,
            self.label_roi1,
            self.label_roi2,
            self.label_ch1,
            self.label_ch2,
            self.task,
            self.epoch,
            self.section,
            self.coherence_metric,
        ]
    
    def to_frame(self) -> CoherenceDataFrame:
        df = CoherenceDataFrame.from_wtcs([self.as_frame_row])
        return df

    #
    # Plots
    #
    def plot(self, **kwargs):
        return plot_wtc(self.wtc, self.times, self.frequencies, self.coi, self.sfreq, **kwargs)

