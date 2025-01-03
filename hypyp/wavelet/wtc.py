from typing import List
import numpy as np
import pandas as pd


from .pair_signals import PairSignals
from .coherence_data_frame import CoherenceDataFrame
from ..plots import plot_wtc
from ..utils import downsample_in_time

MASK_THRESHOLD = 0.5

class WTC:
    def __init__(self, wtc, times, scales, periods, coi, pair: PairSignals, bin_seconds:float|None=None):
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

        self.bin_seconds = bin_seconds
        self.coherence_metric_bins: List[float] = []
        self.coherence_masked_bins: List[float] = []

        self.compute_coherence_in_coi()
    
    # TODO split in time segments here and make sure we weight our values correctly given time and frequencies
    def compute_coherence_in_coi(self):
        mask = self.periods[:, np.newaxis] > self.coi
        self.wtc_masked = np.ma.masked_array(self.wtc, mask)

        coherence = np.mean(self.wtc_masked)
        if np.ma.is_masked(coherence) or not np.isfinite(coherence):
            coherence = np.nan
        self.coherence_metric = coherence

        # Bins
        self.coherence_metric_bins = []
        self.coherence_masked_bins = []

        if self.bin_seconds is None:
            self.coherence_metric_bins = [self.coherence_metric]
            self.coherence_masked_bins = [np.mean(self.wtc_masked.mask)]
        else:
            duration = len(self.times) * self.dt
            steps = int(duration / self.bin_seconds)
            size = int(self.bin_seconds / self.dt)
            for step in range(steps):
                start = step * size
                stop = start + size
                wtc_bin = self.wtc_masked[:, start:stop]
                coherence_bin = np.mean(wtc_bin)
                coherence_masked = np.mean(wtc_bin.mask)
                if np.ma.is_masked(coherence) or not np.isfinite(coherence):
                    coherence_bin = np.nan
                    coherence_masked = 1.0
                elif coherence_masked > MASK_THRESHOLD:
                    coherence_bin = np.nan
                self.coherence_metric_bins.append(coherence_bin)
                self.coherence_masked_bins.append(coherence_masked)

    
    def downsample_in_time(self, bins):
        self.times, self.wtc, self.coi, self.coif, _factor = downsample_in_time(self.times, self.wtc, self.coi, self.coif, bins=bins)
        # must recompute region of interest
        self.compute_coherence_in_coi()
    
    @property
    def as_frame_rows(self) -> List[List]:
        # IMPORTANT: must match the ordering of COHERENCE_FRAME_COLUMNS
        frames = []
        for bin_id in range(len(self.coherence_metric_bins)):
            coherence_metric = self.coherence_metric_bins[bin_id]
            coherence_masked = self.coherence_masked_bins[bin_id]
            frames.append([
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
                bin_id, 
                coherence_metric,
                coherence_masked,
            ])
        return frames
    
    def to_frame(self) -> CoherenceDataFrame:
        df = CoherenceDataFrame.from_wtcs(self.as_frame_rows)
        return df

    #
    # Plots
    #
    def plot(self, **kwargs):
        return plot_wtc(self.wtc, self.times, self.frequencies, self.coi, self.sfreq, **kwargs)

