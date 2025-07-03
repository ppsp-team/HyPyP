from typing import List, Tuple

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .pair_signals import PairSignals
from .coherence_data_frame import CoherenceDataFrame
from ..plots import plot_wavelet_transform_weights
from ..utils import downsample_in_time

MASK_THRESHOLD = 0.5

class WTC:
    W: np.ndarray

    times: np.ndarray
    scales: np.ndarray
    periods: np.ndarray
    frequencies: np.ndarray

    coi: np.ndarray
    coif: np.ndarray

    is_intra: bool
    is_intra_of: int
    is_pseudo: bool

    task: str
    epoch_id: int
    section_id: int

    label_dyad: str
    label_pair: str
    label_s1: str
    label_s2: str
    label_ch1: str
    label_ch2: str
    label_roi1: str
    label_roi2: str

    dt: float
    sfreq: float
    nyquist: float

    wtc_masked: np.ma.MaskedArray
    coherence_metric: float
    coherence_masked: float

    bin_seconds: float|None
    period_cuts: List[float]|None
    coherence_bins: List[Tuple[float, float, str, str]]

    wavelet_library: str
    wavelet_name_with_args: str

    def __init__(
        self,
        W:np.ndarray,
        times:np.ndarray,
        scales:np.ndarray,
        periods:np.ndarray,
        coi:np.ndarray,
        pair: PairSignals,
        bin_seconds:float|None=None,
        period_cuts:List[float]|None=None,
        wavelet_library:str='',
        wavelet_name_with_args:str='',
    ):
        """
        The WTC object holds the results of a Wavelet Transform Coherence

        Args:
            wtc (np.ndarray): weights of the coherence
            times (np.ndarray): timecodes
            scales (np.ndarray): scales used
            periods (np.ndarray): scales in "seconds"
            coi (np.ndarray): cone of influence
            pair (PairSignals): pair of signals used
            bin_seconds (float | None, optional): split in bins every X seconds. Defaults to None.
            period_cuts (List[float] | None, optional): split in bins in frequency domain at the specified periods. Defaults to None.
            wavelet_library (str, optional): name of the library/implementation used. Defaults to ''.
            wavelet_name (str, optional): name of the wavelet (includes wavelet parameters). Defaults to ''.
        """
        self.W = W

        self.times = times
        self.scales = scales
        self.periods = periods
        self.frequencies = 1 / periods

        self.coi = coi
        self.coif = 1 / coi

        self.is_intra = pair.is_intra
        self.is_intra_of = pair.is_intra_of
        self.is_pseudo = pair.is_pseudo

        self.task = pair.label_task
        self.epoch_id = pair.epoch_idx
        self.section_id = pair.section_idx

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

        self.bin_seconds = bin_seconds
        self.period_cuts = period_cuts
        self.frequency_cuts = 1 / np.array(period_cuts) if period_cuts is not None else None
        self.coherence_bins = []

        self.wavelet_library = wavelet_library
        self.wavelet_name_with_args = wavelet_name_with_args

        self._compute_coherence_in_coi()
    
    @property
    def p_ranges(self):
        if self.period_cuts is None:
            # single bin
            p_ranges = [(0, len(self.periods))]
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
                
        return p_ranges
    
    
    @property
    def p_ranges_str(self):
        ret = []
        for p_start, p_end in self.p_ranges:
            ret.append(f"{self.frequencies[p_start]:.2f}Hz-{self.frequencies[p_end-1]:.2f}Hz")
        return ret
            
            
    @property
    def t_ranges(self):
        # don't use self.dt because we want to deal with downsampled data as well
        dt = self.times[1] - self.times[0]

        if self.bin_seconds is None:
            # single bin
            t_ranges = [(0, len(self.times))]
        else:
            t_ranges = []
            duration = len(self.times) * dt
            t_steps = int(duration / self.bin_seconds)
            t_size = int(self.bin_seconds / dt)
            for t_step in range(t_steps):
                t_start = t_step * t_size
                t_stop = t_start + t_size
                t_ranges.append((t_start, t_stop))
        return t_ranges

    def _compute_coherence_in_coi(self):
        # don't use self.dt because we want to deal with downsampled data as well
        dt = self.times[1] - self.times[0]

        mask = self.periods[:, np.newaxis] > self.coi
        self.wtc_masked = np.ma.masked_array(self.W, mask)

        coherence = np.mean(self.wtc_masked)
        coherence_masked = np.mean(self.wtc_masked.mask)
        if np.ma.is_masked(coherence) or not np.isfinite(coherence):
            coherence = np.nan

        self.coherence_metric = coherence
        self.coherence_masked = coherence_masked

        # Time and period bins split
        self.coherence_bins = []
            
        # loop over time bins and period bins
        for t_start, t_stop in self.t_ranges:
            for p_start, p_stop in self.p_ranges:
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

    
    def downsample_in_time(self, bins:int):
        """
        Merge weights together over time to save memory and allow for faster displaying

        Args:
            bins (int): number of bins to keep
        """
        self.times, self.W, self.coi, self.coif, _factor = downsample_in_time(self.times, self.W, self.coi, self.coif, bins=bins)
        # must recompute coherence in cone of interest
        self._compute_coherence_in_coi()
    
    def _moving_average_1d(self, arr, window_size):
        kernel = np.ones(window_size) / window_size

        # replace the masked values by the closest correct value, to avoid edge effects
        mask = np.array(arr.mask)
        valid_indices = np.where(~arr.mask)[0]
        if len(valid_indices) > 0:
            arr[:valid_indices[0]] = arr[valid_indices[0]]
            arr[valid_indices[-1]:] = arr[valid_indices[-1]]
        result = np.convolve(arr, kernel, mode='same') # keep same size array
        return np.ma.masked_array(result, mask)

    def get_as_time_series(self, window_size=10) -> np.ndarray:
        p_ranges = self.p_ranges
        data = np.ma.zeros((len(p_ranges), self.wtc_masked.shape[1]))
        for i in range(len(p_ranges)):
            p_start, p_stop = p_ranges[i]
            data[i,:] = self.wtc_masked[p_start:p_stop].mean(axis=0)
        #print(data.shape)
        result = np.apply_along_axis(self._moving_average_1d, 1, data, window_size)
        return result

    @property
    def as_frame_rows(self) -> List[List]:
        # IMPORTANT: must match the ordering of COHERENCE_FRAME_COLUMNS
        frames = []
        for bin_id in range(len(self.coherence_bins)):
            frames.append([
                self.label_dyad,
                self.is_intra,
                self.is_intra_of,
                self.is_pseudo,
                self.label_s1,
                self.label_s2,
                self.label_roi1,
                self.label_roi2,
                self.label_ch1,
                self.label_ch2,
                self.task,
                self.epoch_id,
                self.section_id,
                bin_id, 
                self.coherence_bins[bin_id][0], # metric
                self.coherence_bins[bin_id][1], # masked
                self.coherence_bins[bin_id][2], # time range
                self.coherence_bins[bin_id][3], # period range
                self.wavelet_library,
                self.wavelet_name_with_args,
            ])
        return frames
    
    def to_frame(self) -> CoherenceDataFrame:
        """
        Get a typed pandas DataFrame from the WTC

        Returns:
            CoherenceDataFrame: typed pandas dataframe of the computed coherence
        """
        df = CoherenceDataFrame.from_wtc_frame_rows(self.as_frame_rows)
        return df

    #
    # Plots
    #
    def plot(self, **kwargs) -> Figure:
        """
        Plot the weights of the WTC

        Returns:
            Figure: matplotlib.Figure
        """
        if 'title' not in kwargs:
            kwargs['title'] = f'{self.label_s1}[{self.label_ch1}] - {self.label_s2}[{self.label_ch2}]'

        return plot_wavelet_transform_weights(
            self.W,
            self.times,
            self.frequencies,
            self.coif,
            self.sfreq,
            bin_seconds=self.bin_seconds,
            frequency_cuts=self.frequency_cuts,
            **kwargs)

