from typing import List, Tuple
import numpy as np
import pandas as pd


from .pair_signals import PairSignals
from .coherence_data_frame import CoherenceDataFrame
from ..plots import plot_wtc
from ..utils import downsample_in_time

MASK_THRESHOLD = 0.5

class WTC:
    def __init__(self, wtc, times, scales, periods, coi, pair: PairSignals, bin_seconds:float|None=None, period_cuts:List[float]|None=None):
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
        self.coherence_masked: float

        self.bin_seconds = bin_seconds
        self.period_cuts = period_cuts
        self.coherence_bins: List[Tuple[float, float, str, str]] = []

        self.compute_coherence_in_coi()
    
    def compute_coherence_in_coi(self):
        # don't use self.dt because we want to deal with downsampled data as well
        dt = self.times[1] - self.times[0]

        mask = self.periods[:, np.newaxis] > self.coi
        self.wtc_masked = np.ma.masked_array(self.wtc, mask)

        coherence = np.mean(self.wtc_masked)
        coherence_masked = np.mean(self.wtc_masked.mask)
        if np.ma.is_masked(coherence) or not np.isfinite(coherence) or coherence_masked > MASK_THRESHOLD:
            coherence = np.nan

        self.coherence_metric = coherence
        self.coherence_masked = coherence_masked

        # Time and period bins split
        self.coherence_bins = []

        if self.bin_seconds is None:
            t_ranges = [(0, len(self.times))]
        else:
            duration = len(self.times) * dt
            t_steps = int(duration / self.bin_seconds)
            t_size = int(self.bin_seconds / dt)
            t_ranges = []
            for t_step in range(t_steps):
                t_start = t_step * t_size
                t_stop = t_start + t_size
                t_ranges.append((t_start, t_stop))

        if self.period_cuts is None:
            p_ranges = [(0, len(self.scales))]
        else:
            p_ranges = []
            start_look_at = 0
            for cut in self.period_cuts:
                cursor = start_look_at

                if cursor < len(self.periods) and cut < self.periods[cursor]:
                    # skip
                    continue

                while cursor < len(self.periods):
                    if self.periods[cursor] > cut and cursor > start_look_at:
                        p_ranges.append((start_look_at, cursor))
                        start_look_at = cursor
                        break
                    cursor += 1
            
            # add one last for the remaining
            if len(p_ranges) == 0:
                p_ranges.append((0, len(self.periods)))
            else:
                last_i = p_ranges[-1][1]
                if last_i < len(self.periods):
                    p_ranges.append((last_i, len(self.periods)))
            
        for t_start, t_stop in t_ranges:
            for p_start, p_stop in p_ranges:
                wtc_bin = self.wtc_masked[p_start:p_stop, t_start:t_stop]
                coherence_bin = np.mean(wtc_bin)
                coherence_masked = np.mean(wtc_bin.mask)
                if np.ma.is_masked(coherence_bin) or not np.isfinite(coherence_bin) or coherence_masked > MASK_THRESHOLD:
                    coherence_bin = np.nan
                self.coherence_bins.append((
                    coherence_bin,
                    coherence_masked,
                    f'{np.round(t_start*dt):.0f}-{np.round(t_stop*dt):.0f}',
                    f'{self.periods[p_start]:.1f}-{self.periods[p_stop-1]:.1f}',
                ))

    
    def downsample_in_time(self, bins):
        self.times, self.wtc, self.coi, self.coif, _factor = downsample_in_time(self.times, self.wtc, self.coi, self.coif, bins=bins)
        # must recompute coherence in cone of interest
        self.compute_coherence_in_coi()
    
    @property
    def as_frame_rows(self) -> List[List]:
        # IMPORTANT: must match the ordering of COHERENCE_FRAME_COLUMNS
        frames = []
        for bin_id in range(len(self.coherence_bins)):
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
                self.coherence_bins[bin_id][0], # metric
                self.coherence_bins[bin_id][1], # masked
                self.coherence_bins[bin_id][2], # time range
                self.coherence_bins[bin_id][3], # period range
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

