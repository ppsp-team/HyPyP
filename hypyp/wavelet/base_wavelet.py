from abc import ABC, abstractmethod
import math
import numpy as np
from skimage.measure import block_reduce

from hypyp.wavelet.pair_signals import PairSignals

from ..plots import plot_cwt_weights, plot_wavelet_coherence
from .smooth import smoothing

def downsample_in_time(times, *args, bins=500):
    ret = []
    # We assume time is always the last column
    factor = math.ceil(times.shape[0] / bins)

    if factor == 1:
        return [times, *args, 1]
    
    # First deal with times. Need to pad (cval) with max value, we don't want to "go back in time" for the last values
    ret.append(block_reduce(times, block_size=factor, func=np.min, cval=np.max(times)))
    
    for item in args:
        if len(item.shape) == 1:
            ret.append(block_reduce(item, block_size=factor, func=np.mean, cval=np.mean(item)))
        elif len(item.shape) == 2:
            ret.append(block_reduce(item, block_size=(1,factor), func=np.mean, cval=np.mean(item)))
        else:
            raise RuntimeError(f'Unsupported number of column for downsampling: {len(item)}')
    
    ret.append(factor)

    return ret
        
class CWT:
    def __init__(self, weights, times, scales, frequencies, coif, tracer=None):
        self.W: np.ndarray = weights
        self.times: np.ndarray = times
        self.dt = times[1] - times[0]
        self.scales: np.ndarray = scales
        self.frequencies: np.ndarray = frequencies
        self.coi: np.ndarray = 1 / coif # Cone of influence, in scales
        self.coif: np.ndarray = coif # Cone of influence, in frequencies
        self.tracer: dict = tracer

    def plot(self, **kwargs):
        return plot_cwt_weights(self.W, self.times, self.frequencies, self.coif)

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
        self.ch_name1 = pair.ch_name1
        self.ch_name2 = pair.ch_name2

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
    
    def plot(self, **kwargs):
        return plot_wavelet_coherence(self.wtc_roi, self.times, self.frequencies, self.coif, self.sig, **kwargs)


class BaseWavelet(ABC):
    def __init__(self, evaluate=False, cache=None):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wtc = None
        self.cache = cache

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
        return self.cache is not None
    
    @abstractmethod
    def evaluate_psi(self):
        pass
    
    @abstractmethod
    def cwt(self, y, dt, dj):
        pass
    
    def wtc(self, pair: PairSignals, cache_suffix='', tracer=None):
        
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

        dj = 1 / 12 # TODO have as parameter
    
        # TODO: have detrend as parameter
        # TODO: have normalize as parameter
        y1 = (y1 - y1.mean()) / y1.std()
        y2 = (y2 - y2.mean()) / y2.std()
    
        # TODO add caching of smoothed transform
        # TODO add 'cwt' to cache key arguments
        cwt1_cached = None
        cwt2_cached = None
        S1_cached = None
        S2_cached = None
        if self.use_caching:
            cwt1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'cwt', cache_suffix))
            cwt2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'cwt', cache_suffix))
            S1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'smooth', cache_suffix))
            S2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'smooth', cache_suffix))

        cwt1 = cwt1_cached
        if cwt1 is None:
            cwt1 = self.cwt(y1, dt, dj)
        
        cwt2 = cwt2_cached
        if cwt2 is None:
            cwt2 = self.cwt(y2, dt, dj)

        if (cwt1.scales != cwt2.scales).any():
            raise RuntimeError('The two CWT have different scales')

        if (cwt1.frequencies != cwt2.frequencies).any():
            raise RuntimeError('The two CWT have different frequencies')

        W1 = cwt1.W
        W2 = cwt2.W
        W12 = W1 * W2.conj()

        frequencies = cwt1.frequencies
        scales = cwt1.scales
        times = cwt1.times

        # Compute cross wavelet transform and coherence
        # TODO: cross wavelet
        scaleMatrix = np.ones([1, N]) * scales[:, None]
        smoothing_kwargs = dict(
            dt=dt,
            dj=dj,
            scales=scales,
            smooth_factor=self.wtc_smoothing_smooth_factor,
            boxcar_size=self.wtc_smoothing_boxcar_size,
        )

        S1 = S1_cached
        S2 = S2_cached
        if S1 is None:
            S1 = smoothing(np.abs(W1) ** 2 / scaleMatrix, **smoothing_kwargs)
        if S2 is None:
            S2 = smoothing(np.abs(W2) ** 2 / scaleMatrix, **smoothing_kwargs)

        S12 = np.abs(smoothing(W12 / scaleMatrix, **smoothing_kwargs))
        wtc = S12 ** 2 / (S1 * S2)

        coi_cached = None
        coif_cached = None
        if self.use_caching:
            coi_cached = self.get_cache_item(self.get_cache_key_coi(N, dt))
            coif_cached = self.get_cache_item(self.get_cache_key_coif(N, dt))
        
        if coi_cached is not None and coif_cached is not None:
            coi = coi_cached
            coif = coif_cached
        else:
            coi, coif = self.get_cone_of_influence(N, dt)

        if self.use_caching:
            if cwt1_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 0, 'cwt', cache_suffix), cwt1)
            if cwt2_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 1, 'cwt', cache_suffix), cwt2)
            if S1_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 0, 'smooth', cache_suffix), S1)
            if S2_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 1, 'smooth', cache_suffix), S2)
            if coi_cached is None:
                self.add_cache_item(self.get_cache_key_coi(N, dt), coi)
            if coif_cached is None:
                self.add_cache_item(self.get_cache_key_coif(N, dt), coif)


        if tracer is not None:
            tracer['cwt1'] = cwt1
            tracer['cwt2'] = cwt2
            tracer['W1'] = W1
            tracer['W2'] = W2
            tracer['W12'] = W12
            tracer['S1'] = S1
            tracer['S2'] = S2
            tracer['S12'] = S12

        return WTC(wtc, times, scales, frequencies, coif, pair, tracer=tracer)

    def get_cone_of_influence(self, N, dt):
        # TODO this result is the same for every pair. It should be cached
        # Cone of influence calculations
        f0 = 2 * np.pi
        cmor_coi = 1.0 / np.sqrt(2)
        # TODO: this is hardcoded, we have to check where this equation comes from
        cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = cmor_flambda * cmor_coi * dt * coi
        coif = 1.0 / coi
        return coi, coif
    
        

    def get_cache_item(self, key):
        if not self.use_caching:
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
        if subject_id == 0:
            subject_label = pair.label_s1
            ch_name = pair.ch_name1
        elif subject_id == 1:
            subject_label = pair.label_s2
            ch_name = pair.ch_name2
        else:
            raise RuntimeError(f'subject_id must be 0 or 1')
        
        if len(subject_label) < 1:
            raise RuntimeError(f'subjects must have labels to use caching')

        if pair.task == '':
            raise RuntimeError(f'must have task to have unique identifiers in caching')

        key = f'{subject_label}-{ch_name}-{pair.task}-{str(pair.range)}-{obj_id}'
        if cache_suffix != '':
            key += f'f{cache_suffix}'
        
        return key
    
    def get_cache_key_coi(self, N, dt):
        return f'coi_{N}_{dt}'
    
    def get_cache_key_coif(self, N, dt):
        return f'coif_{N}_{dt}'
    
    