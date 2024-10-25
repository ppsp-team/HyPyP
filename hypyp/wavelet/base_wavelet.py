from abc import ABC, abstractmethod
import numpy as np
from skimage.measure import block_reduce

from ..plots import plot_cwt_weights, plot_wavelet_coherence

class CWT:
    def __init__(self, weights, times, scales, frequencies, coif, tracer=None):
        self.W: np.ndarray = weights
        self.times: np.ndarray = times
        self.scales: np.ndarray = scales
        self.frequencies: np.ndarray = frequencies
        self.coi: np.ndarray = 1 / coif # Cone of influence, in scales
        self.coif: np.ndarray = coif # Cone of influence, in frequencies
        self.tracer: dict = tracer

    def plot(self, **kwargs):
        return plot_cwt_weights(self.W, self.times, self.frequencies, self.coif)

class WCT:
    def __init__(self, wct, times, scales, frequencies, coif, tracer=None):
        self.wct = wct
        self.times = times
        self.scales = scales
        self.frequencies = frequencies
        self.coi = 1 / coif
        self.coif = coif
        self.tracer = tracer
    
    def plot(self, **kwargs):
        return plot_wavelet_coherence(self.wct, self.times, self.frequencies, self.coif, **kwargs)


class BaseWavelet(ABC):
    def __init__(self, evaluate):
        self._wavelet = None
        self._psi_x = None
        self._psi = None
        self._wct = None

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
    
    @abstractmethod
    def wct(self, y1, y2, dt):
        pass
    