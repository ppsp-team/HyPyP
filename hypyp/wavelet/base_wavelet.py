from abc import ABC, abstractmethod
from typing import List, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import fft
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

from .pair_signals import PairSignals
from .wtc import WTC
from .cwt import CWT
from ..profiling import TimeTracker

# Window size is in "scale"
DEFAULT_SMOOTH_WIN_SIZE = 0.6

DEFAULT_PERIOD_RANGE = (2, 20)
DEFAULT_PERIOD_DJ = 1/12

class BaseWavelet(ABC):
    """
    Base class for Wavelet implementations. See also ComplexMorletWavelet, which is the default implementation.

    Args:
        period_range (Tuple[float, float] | None, optional): The range for which the Wavelet Transform should be computed, in period_range. Defaults to (2, 20).
        frequency_range (Tuple[float, float] | None, optional): The range for which the Wavelet Transform should be computed, in frequency. Defaults to None, see period_range instead.
        evaluate (bool, optional): Should the PSI be evaluated at wavelet instanciation. Defaults to True.
        cache (dict | None, optional): Cache dictionary to reuse already computed results. Defaults to a new dict().
        disable_caching (bool, optional): Should the caching be disabled. Defaults to False.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        dj (float, optional): Step size between scales in one octave. Defaults to 1/12.
        wtc_smoothing_win_size (float, optional): The width of window for smooting the CWT in frequency. The value is in "scale" and does not depend on "dj". Defaults to 0.6.
    """

    verbose: bool
    dj: float
    wtc_smoothing_win_size: float
    period_range: Tuple[float, float]
    cache: dict|None
    cache_is_disabled: bool

    default_period_range: Tuple[float, float] = DEFAULT_PERIOD_RANGE
    default_period_dj: float = DEFAULT_PERIOD_DJ
    default_smooth_win_size: float = DEFAULT_SMOOTH_WIN_SIZE

    def __init__(
        self,
        period_range:Tuple[float, float]|None=None,
        frequency_range:Tuple[float, float]|None=None,
        evaluate:bool=True,
        cache:dict|None=None,
        disable_caching:bool=False,
        verbose:bool=False,
        dj:float=DEFAULT_PERIOD_DJ,
        wtc_smoothing_win_size:float=DEFAULT_SMOOTH_WIN_SIZE,
    ):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wtc = None
        self.verbose = verbose

        self.wtc_smoothing_win_size = wtc_smoothing_win_size

        if dj > 1:
            raise RuntimeError(f'dj must be smaller than 1. Received {dj}')
        self.dj = dj

        if period_range is not None and frequency_range is not None:
            raise RuntimeError('Cannot specify both period_range and frequency_range')

        if period_range is not None:
            self.period_range = period_range
        elif frequency_range is not None:
            self.period_range = (1 / frequency_range[0], 1 / frequency_range[1])
        else:
            self.period_range = DEFAULT_PERIOD_RANGE
        
        # swap to have them in order
        if self.period_range[0] > self.period_range[1]:
            self.period_range = (self.period_range[1], self.period_range[0])
        
        self.cache = cache
        self.cache_is_disabled = disable_caching

        if self.cache_is_disabled:
            self.cache = None
        elif self.cache is None:
            self.cache = dict()

        if evaluate:
            self.evaluate_psi()
    
    @property
    def psi(self):
        return self._psi
    
    @property
    def psi_x(self):
        return self._psi_x
    
    @property
    def psi_dx(self):
        if self._psi_x is None:
            raise RuntimeError('Wavelet not evaluated yet')
        return self._psi_x[1] - self._psi_x[0]

    @property
    def domain(self):
        if self._psi_x is None:
            raise RuntimeError('Wavelet not evaluated yet')
        return self._psi_x[0], self._psi_x[-1]

    @property
    def use_caching(self):
        return not self.cache_is_disabled and self.cache is not None
    
    @abstractmethod
    def evaluate_psi(self):
        pass
    
    @abstractmethod
    def cwt(self, y, dt) -> CWT:
        pass
    
    @property
    @abstractmethod
    def wavelet_library(self):
        pass

    @property
    @abstractmethod
    def wavelet_name(self):
        pass
    
    @property
    @abstractmethod
    def wavelet_name_with_args(self):
        pass
    
    @property
    @abstractmethod
    def flambda(self):
        pass

    def get_periods(self) -> np.ndarray:
        """
        Get list of period values for this specific wavelet settings

        Returns:
            np.array: list of periods that should be used with this wavelet
        """
        low, high =  self.period_range
        n_scales = np.log2(high/low) 
        n_steps = int(np.round(n_scales / self.dj))
        periods = np.logspace(np.log2(low), np.log2(high), n_steps, base=2)
        return periods
        
    def wtc(self,
            pair: PairSignals,
            bin_seconds:float|None=None,
            period_cuts:List[float]|None=None,
            cache_suffix:str='') -> WTC:
        """
        Compute the Wavalet Transform Coherence for a pair of signals. 
        
        First computes the Continuous Wavelet Transform of both signals,
        then filters them and finally the coherence between the 2 signals.

        It saves to a cache in memory the intermediary results for reuse on other pairs.
        For example, if there is a pair made of signal A and signal B and then a pair with 
        signal A and signal C, we don't have to re-compute the CWT of signal A twice.

        Args:
            pair (PairSignals): a pair of signals on which to compute coherence
            bin_seconds (float | None, optional): split the resulting WTC in time bins for balancing weights. Defaults to None.
            period_cuts (List[float] | None, optional): split the resulting WTC in period/frequency bins for balancing weights and finer analysis. Defaults to None.
            cache_suffix (str, optional): string to add to the caching key. Defaults to ''.

        Returns:
            WTC: A Wavelet Transform Coherence object that encapsulates the Weights and the parameters of the resulting transform
        """
        y1 = pair.y1
        y2 = pair.y2
        dt = pair.dt

        if len(y1) != len(y2):
            raise RuntimeError(f'Arrays not same size. y1:{len(y1)}, y2:{len(y2)}')

        N = len(y1)

        # TODO: have detrend as parameter
        # if detrend:
        #     y1 = signal.detrend(y1, type='linear')
        #     y2 = signal.detrend(y2, type='linear')
        # TODO: have normalize as parameter
        # TODO: maybe this should be in preprocessing instead
        y1 = (y1 - y1.mean()) / y1.std()
        y2 = (y2 - y2.mean()) / y2.std()
    
        cwt1_cached = self._get_cache_item(self._get_cache_key_pair(pair, 0, 'cwt', cache_suffix))
        cwt1: CWT = self.cwt(y1, dt) if cwt1_cached is None else cwt1_cached

        cwt2_cached = self._get_cache_item(self._get_cache_key_pair(pair, 1, 'cwt', cache_suffix))
        cwt2: CWT = self.cwt(y2, dt) if cwt2_cached is None else cwt2_cached

        if (cwt1.scales != cwt2.scales).any():
            raise RuntimeError('The two CWT have different scales')

        if (cwt1.frequencies != cwt2.frequencies).any():
            raise RuntimeError('The two CWT have different frequencies')

        W1 = cwt1.W
        W2 = cwt2.W
        # Take a look at caching keys if you fail with this kind of error. Make sure cache keys are well targetted
        #     ValueError: operands could not be broadcast together with shapes (40,782) (40,100)
        try:
            W12 = W1 * W2.conj()
        except ValueError as e:
            warnings.warn("Wrong operand shapes could mean that a wrong cached value is used. Please check that cache keys contain all the relevant arguments")
            raise e

        periods = cwt1.periods
        scales = cwt1.scales
        times = cwt1.times

        scaleMatrix = np.ones([1, N]) * scales[:, None]
        smoothing_kwargs = dict(
            dt=dt,
            scales=scales,
            cache_suffix=cache_suffix,
        )

        S1_cached = self._get_cache_item(self._get_cache_key_pair(pair, 0, 'smooth', cache_suffix))
        S1 = self.smoothing(np.abs(W1) ** 2 / scaleMatrix, **smoothing_kwargs) if S1_cached is None else S1_cached

        S2_cached = self._get_cache_item(self._get_cache_key_pair(pair, 1, 'smooth', cache_suffix))
        S2 = self.smoothing(np.abs(W2) ** 2 / scaleMatrix, **smoothing_kwargs) if S2_cached is None else S2_cached

        S12 = np.abs(self.smoothing(W12 / scaleMatrix, **smoothing_kwargs))
        wtc = S12 ** 2 / (S1 * S2)

        coi = self._get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)

        self._update_cache_if_none(self._get_cache_key_pair(pair, 0, 'cwt', cache_suffix), cwt1)
        self._update_cache_if_none(self._get_cache_key_pair(pair, 1, 'cwt', cache_suffix), cwt2)
        self._update_cache_if_none(self._get_cache_key_pair(pair, 0, 'smooth', cache_suffix), S1)
        self._update_cache_if_none(self._get_cache_key_pair(pair, 1, 'smooth', cache_suffix), S2)

        return WTC(
            wtc,
            times,
            scales,
            periods,
            coi,
            pair,
            bin_seconds=bin_seconds,
            period_cuts=period_cuts,
            wavelet_library=self.wavelet_library,
            wavelet_name_with_args=self.wavelet_name_with_args,
        )

    def _update_cache_if_none(self, key, value):
        if not self.use_caching:
            return

        if self._get_cache_item(key) is None:
            self._add_cache_item(key, value)

    def _get_and_cache_cone_of_influence(self, N, dt, cache_suffix=''):
        cache_key = self._get_cache_key('coi', N, dt, cache_suffix)
        coi = self._get_cache_item(cache_key)
        if coi is not None:
            return coi
        coi = self._get_cone_of_influence(N, dt)
        self._add_cache_item(cache_key, coi)
        return coi


    def _get_cone_of_influence(self, N:int, dt:float) -> np.ndarray:
        """
        Get the cone of influence of a Continuous Wavelet Transform, 
        that is the region where edge effects become significant due to the finite length of the signal

        Args:
            N (int): number of times for each scale of the WTC
            dt (float): delta time of the WTC

        Returns:
            np.ndarray: for each time, the periods at which the value should be ignored
        """
        # Equations come from "A Practical Guide to Wavelet Analysis" from Torrence and Compo (1998), Table 1

        # e-folding valid for cmor (complex morlet) and cgau (complex gaussian)
        e_folding_time = 1.0 / np.sqrt(2)
        flambda = self.flambda

        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = flambda * e_folding_time * dt * coi

        return coi
    
    #
    # Smoothing
    #
    def smoothing(self, W:np.ndarray, dt:float, scales:np.ndarray, cache_suffix:str='') -> np.ndarray:
        """
        Smooth the weights of a continuous wavelet transform in time and frequency domains.

        Args:
            W (np.ndarray): the weights of the transform to be smoothed
            dt (float): delta time
            scales (np.ndarray): the normalized scales used with W
            cache_suffix (str, optional): cache segregation suffix. Defaults to ''.

        Returns:
            np.ndarray: an numpy array of the same shape as W
        """
        #
        # Filter in time.
        #
        fft_kwargs = self._get_fft_kwargs(W[0, :])
        scales_norm = scales # scales are already normalized
        
        k = 2 * np.pi * fft.fftfreq(fft_kwargs['n'])
        k2 = k ** 2

        # Smoothing by Gaussian window (absolute value of wavelet function)
        # using the convolution theorem: multiplication by Gaussian curve in
        # Fourier domain for each scale, outer product of scale and frequency
        gaus_fft = np.exp(-0.5 * (scales_norm[:, np.newaxis] ** 2) * k2)  # Outer product

        W_fft = fft.fft(W, axis=1, **fft_kwargs)
        smooth_fft = gaus_fft * W_fft
        smooth = fft.ifft(smooth_fft, axis=1, **fft_kwargs, overwrite_x=True)
        T = smooth[:, :W.shape[1]]  # Remove possibly padded region due to FFT

        if np.isreal(W).all():
            T = T.real
        
        #
        # Filter in scale. 
        # 
        win_cache_key = self._get_cache_key('boxcar_window', self.wtc_smoothing_win_size, self.dj, cache_suffix)
        win = self._get_cache_item(win_cache_key)
        if win is None:
            win = self._get_smoothing_window(self.wtc_smoothing_win_size, self.dj)
            self._add_cache_item(win_cache_key, win)

        T = convolve1d(T, win, axis=0, mode='nearest')
        #T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

        return T

    @staticmethod
    def _get_fft_kwargs(signal, **kwargs):
        return dict(**kwargs, n = int(2 ** np.ceil(np.log2(len(signal)))))

    @staticmethod
    def _get_smoothing_window(boxcar_size, dj):
        # Copied from matlab
        # boxcar_size is "in scale"
        #size_in_scales = boxcar_size
        size_in_scales = boxcar_size * 2 # the pycwt code has a *2. TODO: find out why it is different from matlab code
        size_in_steps = size_in_scales / dj
        fraction = size_in_steps % 1
        fraction_half = fraction / 2
        size_in_steps = int(np.floor(size_in_steps))

        # edge case, return a dirac
        if size_in_steps <= 1:
            return np.array([1])

        if fraction == 0:
            win = np.ones(size_in_steps)
        else:
            win = np.ones(size_in_steps + 1)
            win[0] = fraction_half
            win[-1] = fraction_half

        # normalize
        win /= win.sum()
        return win
        
        

    #
    # Caching
    # NOTE: We could use functools.cache instead of a custom caching strategy.
    #       That would require to have static methods
    #
    def _get_cache_item(self, key):
        if not self.use_caching:
            return None
        if key is None:
            return None
        try:
            found = self.cache[key]
            if self.verbose:
                print(f'Found cache key {key}')
                pass
            return found
        except:
            if self.verbose:
                #print(f'Not found cache key {key}')
                pass
            return None
    
    def _add_cache_item(self, key, value):
        if self.use_caching:
            self.cache[key] = value

    def clear_cache(self):
        self.cache = dict()

    def _get_cache_key_pair(self, pair: PairSignals, subject_idx: int, obj_id: str, cache_suffix: str = ''):
        if not self.use_caching:
            return None

        if subject_idx == 0:
            subject_label = pair.label_s1
            ch_name = pair.label_ch1
        elif subject_idx == 1:
            subject_label = pair.label_s2
            ch_name = pair.label_ch2
        else:
            raise RuntimeError(f'subject_idx must be 0 or 1')
        
        if len(subject_label) < 1:
            raise RuntimeError(f'subjects must have labels to use caching')

        if pair.label_task == '':
            raise RuntimeError(f'must have task to have unique identifiers in caching')

        time_range_str = f'{pair.x[0]}-{pair.x[-1]}'
        key = f'[{subject_label}][{ch_name}][{pair.label_task}][{pair.epoch_idx}][{pair.x[0]}-{pair.x[-1]}][{time_range_str}][{obj_id}]'
        if cache_suffix != '':
            key += f'_{cache_suffix}'
        
        return key
    
    def _get_cache_key(self, *args):
        if not self.use_caching:
            return None
        return f'key_{"_".join([str(arg) for arg in args])}'
    

    #
    # Plots
    #
    def plot_mother_wavelet(self, show_legend=True, ax:Axes|None=None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(self.psi_x, np.real(self.psi))
        ax.plot(self.psi_x, np.imag(self.psi))
        ax.plot(self.psi_x, np.abs(self.psi))
        ax.title.set_text(f"mother wavelet ({self.wavelet_name_with_args})")
        if show_legend:
            ax.legend(['real', 'imag', 'abs'])
        return fig
