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

        if tracer:
            self.tracer = tracer
        else:
            self.tracer = dict()

        self.sig_metric = np.mean(wtc) # TODO this is just a PoC
        self.sig_p_value = None
        self.sig_t_stat = None
        self.sig = sig
    
    def plot(self, **kwargs):
        return plot_wavelet_coherence(self.wtc, self.times, self.frequencies, self.coif, self.sig, **kwargs)


class BaseWavelet(ABC):
    def __init__(self, evaluate):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wtc = None

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

    @abstractmethod
    def evaluate_psi(self):
        pass
    
    @abstractmethod
    def cwt(self, y, dt, dj):
        pass
    
    def wtc(self, pair: PairSignals, cwt1_cache=None, cwt2_cache=None):
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
    
        if cwt1_cache is not None:
            cwt1 = cwt1_cache
        else:
            cwt1 = self.cwt(y1, dt, dj)
        
        if cwt2_cache is not None:
            cwt2 = cwt2_cache
        else:
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
        S1 = smoothing(np.abs(W1) ** 2 / scaleMatrix, **smoothing_kwargs)
        S2 = smoothing(np.abs(W2) ** 2 / scaleMatrix, **smoothing_kwargs)

        S12 = np.abs(smoothing(W12 / scaleMatrix, **smoothing_kwargs))
        wtc = S12 ** 2 / (S1 * S2)

        # Cone of influence calculations
        f0 = 2 * np.pi
        cmor_coi = 1.0 / np.sqrt(2)
        # TODO: this is hardcoded, we have to check where this equation comes from
        cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = cmor_flambda * cmor_coi * dt * coi
        coif = 1.0 / coi
    
        self.tracer['cwt1'] = cwt1
        self.tracer['cwt2'] = cwt2
        self.tracer['W1'] = W1
        self.tracer['W2'] = W2
        self.tracer['W12'] = W12
        self.tracer['S1'] = S1
        self.tracer['S2'] = S2
        self.tracer['S12'] = S12

        return WTC(wtc, times, scales, frequencies, coif, pair, tracer=self.tracer)

    