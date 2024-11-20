from abc import ABC, abstractmethod
import numpy as np
from skimage.measure import block_reduce

from hypyp.wavelet.pair_signals import PairSignals

from ..plots import plot_cwt_weights, plot_wavelet_coherence
from .smooth import smoothing

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

        frequencies_indices = frequencies[:, np.newaxis]
        mask = frequencies[:, np.newaxis] < coif
        self.wtc_masked = np.ma.masked_array(wtc, mask)

        self.sig_metric = np.mean(self.wtc_masked) # TODO this is just a PoC
        self.sig_p_value = None
        self.sig_t_stat = None
        self.sig = sig
    
    def plot(self, **kwargs):
        return plot_wavelet_coherence(self.wtc_masked, self.times, self.frequencies, self.coif, self.sig, **kwargs)


class BaseWavelet(ABC):
    def __init__(self, evaluate=False, cache_dict=None):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wtc = None
        self.cache_dict = cache_dict

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
        return self.cache_dict is not None
    
    @abstractmethod
    def evaluate_psi(self):
        pass
    
    @abstractmethod
    def cwt(self, y, dt, dj):
        pass
    
    def wtc(self, pair: PairSignals, tracer=None):
        
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
            cwt1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'cwt'))
            cwt2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'cwt'))
            S1_cached = self.get_cache_item(self.get_cache_key_pair(pair, 0, 'smooth'))
            S2_cached = self.get_cache_item(self.get_cache_key_pair(pair, 1, 'smooth'))

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
                self.add_cache_item(self.get_cache_key_pair(pair, 0, 'cwt'), cwt1)
            if cwt2_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 1, 'cwt'), cwt2)
            if S1_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 0, 'smooth'), S1)
            if S2_cached is None:
                self.add_cache_item(self.get_cache_key_pair(pair, 1, 'smooth'), S2)
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
            return self.cache_dict[key]
        except:
            return None
    
    def add_cache_item(self, key, value):
        self.cache_dict[key] = value

    def clear_cache(self):
        self.cache_dict = dict()

    def get_cache_key_pair(self, pair: PairSignals, subject_id: int, obj_id: str):
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

        return f'{subject_label}-{ch_name}-{pair.task}-{str(pair.range)}-{obj_id}'
    
    def get_cache_key_coi(self, N, dt):
        return f'coi_{N}_{dt}'
    
    def get_cache_key_coif(self, N, dt):
        return f'coif_{N}_{dt}'
    
    