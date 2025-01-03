from abc import ABC, abstractmethod
from typing import List
import warnings

import numpy as np
from scipy import fft
from scipy.ndimage import convolve1d

from .pair_signals import PairSignals
from .wtc import WTC
from .cwt import CWT
from ..profiling import TimeTracker

DEFAULT_SMOOTHING_BOXCAR_SIZE = 0.6

class BaseWavelet(ABC):
    def __init__(
        self,
        evaluate=False,
        cache=None,
        disable_caching=False,
        wtc_smoothing_boxcar_size=None,
    ):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wtc = None
        self.wtc_smoothing_boxcar_size = wtc_smoothing_boxcar_size

        
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
    def cwt(self, y, dt, dj):
        pass
    
    @property
    @abstractmethod
    def wavelet_library(self):
        pass

    @property
    @abstractmethod
    def wavelet_name(self):
        pass
    
    def wtc(self, pair: PairSignals, bin_seconds:float|None=None, period_cuts:List[float]|None=None, cache_suffix=''):
        
        # TODO add verbose option
        #if s1_cwt is not None:
        #    print(f'Reusing cache for key "{s1_cwt_key}"')
        #if s2_cwt is not None:
        #    print(f'Reusing cache for key "{s2_cwt_key}"')

        y1 = pair.y1
        y2 = pair.y2
        dt = pair.dt
        if len(y1) != len(y2):
            raise RuntimeError("Arrays not same size")

        N = len(y1)

        dj = 1 / 12 # TODO have as parameter, or as class property with default
    
        # TODO: have detrend as parameter
        # if detrend:
        #     y1 = signal.detrend(y1, type='linear')
        #     y2 = signal.detrend(y2, type='linear')
        # TODO: have normalize as parameter
        # TODO: maybe this should be in preprocessing instead
        y1 = (y1 - y1.mean()) / y1.std()
        y2 = (y2 - y2.mean()) / y2.std()
    
        cwt1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'cwt', cache_suffix))
        cwt1: CWT = self.cwt(y1, dt, dj, cache_suffix=cache_suffix) if cwt1_cached is None else cwt1_cached

        cwt2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'cwt', cache_suffix))
        cwt2: CWT = self.cwt(y2, dt, dj, cache_suffix=cache_suffix) if cwt2_cached is None else cwt2_cached

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
            dj=dj,
            scales=scales,
            cache_suffix=cache_suffix,
        )
        if self.wtc_smoothing_boxcar_size is not None:
            smoothing_kwargs['boxcar_size'] = self.wtc_smoothing_boxcar_size

        S1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'smooth', cache_suffix))
        S1 = self.smoothing(np.abs(W1) ** 2 / scaleMatrix, **smoothing_kwargs) if S1_cached is None else S1_cached

        S2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'smooth', cache_suffix))
        S2 = self.smoothing(np.abs(W2) ** 2 / scaleMatrix, **smoothing_kwargs) if S2_cached is None else S2_cached

        S12 = np.abs(self.smoothing(W12 / scaleMatrix, **smoothing_kwargs))
        wtc = S12 ** 2 / (S1 * S2)

        coi = self.get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)

        self.update_cache_if_none(self.get_cache_key_pair(pair, 0, 'cwt', cache_suffix), cwt1)
        self.update_cache_if_none(self.get_cache_key_pair(pair, 1, 'cwt', cache_suffix), cwt2)
        self.update_cache_if_none(self.get_cache_key_pair(pair, 0, 'smooth', cache_suffix), S1)
        self.update_cache_if_none(self.get_cache_key_pair(pair, 1, 'smooth', cache_suffix), S2)

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
            wavelet_name=self.wavelet_name,
        )

    def update_cache_if_none(self, key, value):
        if not self.use_caching:
            return

        if self.get_cache_item(key) is None:
            self.add_cache_item(key, value)

    def get_and_cache_cone_of_influence(self, N, dt, cache_suffix=''):
        cache_key = self.get_cache_key('coi', N, dt, cache_suffix)
        coi = self.get_cache_item(cache_key)
        if coi is not None:
            return coi
        coi = self.get_cone_of_influence(N, dt)
        self.update_cache_if_none(cache_key, coi)
        return coi


    def get_cone_of_influence(self, N, dt):
        # See "A Practical Guide to Wavelet Analysis" from Torrence and Compo (1998), Table 1
        f0 = 2 * np.pi

        # e-folding valid for cmor (complex morlet) and cgau (complex gaussian)
        e_folding_time = 1.0 / np.sqrt(2)

        if self.wavelet_name.startswith('cmor'):
            flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        elif self.wavelet_name.startswith('cgau'):
            m = int(self.wavelet_name[4:])
            flambda = 2 * np.pi / np.sqrt(m + 0.5)
        else:
            raise RuntimeError(f'Unknown cone of influence for wavelet {self.wavelet_name}')

        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = flambda * e_folding_time * dt * coi

        return coi
    
    
    #
    # Smoothing
    #
    def smoothing(self, W, dt, dj, scales, boxcar_size=DEFAULT_SMOOTHING_BOXCAR_SIZE, cache_suffix=''):
        """Smoothing function used in coherence analysis.

        Parameters
        ----------
        W :
        dt :
        dj :
        scales :

        Returns
        -------
        T :

        """
        #
        # Filter in time.
        #
        # The smoothing is performed by using a filter given by the absolute
        # value of the wavelet function at each scale, normalized to have a
        # total weight of unity, according to suggestions by Torrence &
        # Webster (1999) and by Grinsted et al. (2004).
        fft_kwargs = self.get_fft_kwargs(W[0, :])
        scales_norm = scales / dt
        
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

        # For the Morlet wavelet it's simply a boxcar with 0.6 width.
        # TODO find where the above comments come from
        win_cache_key = self.get_cache_key('boxcar_window', boxcar_size, dj, cache_suffix)
        win = self.get_cache_item(win_cache_key)
        if win is None:
            win = self.get_boxcar_window(boxcar_size, dj)
            self.update_cache_if_none(win_cache_key, win)

        T = convolve1d(T, win, axis=0, mode='nearest')

        return T

    @staticmethod
    def get_fft_kwargs(signal, **kwargs):
        return dict(**kwargs, n = int(2 ** np.ceil(np.log2(len(signal)))))

    # TODO this should be called "weighted boxcar", since the edges differ. See if we want to test different windows
    # TODO have the possibility to compare with Hann window
    @staticmethod
    def get_boxcar_window(boxcar_size, dj):
        # Copied from matlab
        # boxcar_size is "in scale"
        size_in_scales = boxcar_size
        size_in_steps = size_in_scales/dj
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
    #
    def get_cache_item(self, key):
        if not self.use_caching:
            return None
        if key is None:
            return None
        try:
            return self.cache[key]
        except:
            return None
    
    def add_cache_item(self, key, value):
        self.cache[key] = value

    def clear_cache(self):
        self.cache = dict()

    def get_cache_key_pair(self, pair: PairSignals, subject_id: int, obj_id: str, cache_suffix: str = ''):
        if not self.use_caching:
            return None

        if subject_id == 0:
            subject_label = pair.label_s1
            ch_name = pair.label_ch1
        elif subject_id == 1:
            subject_label = pair.label_s2
            ch_name = pair.label_ch2
        else:
            raise RuntimeError(f'subject_id must be 0 or 1')
        
        if len(subject_label) < 1:
            raise RuntimeError(f'subjects must have labels to use caching')

        if pair.task == '':
            raise RuntimeError(f'must have task to have unique identifiers in caching')

        key = f'{subject_label}_{ch_name}_{pair.task}_{pair.epoch}_{pair.x[0]}_{pair.x[-1]}_{str(pair.time_range)}_{obj_id}'
        if cache_suffix != '':
            key += f'f{cache_suffix}'
        
        return key
    
    def get_cache_key(self, *args):
        if not self.use_caching:
            return None
        return f'key_{"_".join([str(arg) for arg in args])}'
    