from math import ceil, floor
import numpy as np

from ..base_wavelet import BaseWavelet
from ..cwt import CWT
import scipy.signal

# This will divided by sfreq
DEFAULT_SCIPY_CENTER_FREQUENCY = 6

class ScipyWavelet(BaseWavelet):
    def __init__(
        self,
        center_frequency=DEFAULT_SCIPY_CENTER_FREQUENCY,
        wtc_smoothing_win_size=1,
        cwt_params=dict(),
        **kwargs,
    ):
        self.wtc_smoothing_win_size = wtc_smoothing_win_size
        self.cwt_params = cwt_params
        # TODO check this normalisation
        self.center_frequency = center_frequency
        #self.center_frequency = center_frequency / (2 * np.pi)
        super().__init__(**kwargs)

    @property
    def wavelet_library(self):
        return 'scipy'

    @property
    def wavelet_name(self):
        return f'cmor{self.bandwidth_frequency},{self.center_frequency}'
    
    @property
    def bandwidth_frequency(self):
        return 2

    def evaluate_psi(self):
        M = 1000
        s = 100
        w = self.center_frequency
        wavelet = scipy.signal.morlet2(M, s, w)
        self._psi_x = np.arange(M)
        self._psi = wavelet

        return self._psi, self._psi_x

    def get_cone_of_influence(self, N, dt):
        # See "A Practical Guide to Wavelet Analysis" from Torrence and Compo (1998), Table 1

        # e-folding valid for cmor (complex morlet) and cgau (complex gaussian)
        e_folding_time = 1.0 / np.sqrt(2)

        # TODO check this computation. Seems wrong
        flambda = 2 * np.pi / self.center_frequency
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = flambda * e_folding_time * dt * coi

        return coi
    
    def cwt(self, y, dt, dj=1/12, cache_suffix:str='') -> CWT:
        N = len(y)
        fs = 1 / dt
        periods = self.get_periods(dj)
        scales = (self.center_frequency * fs * periods) / (2 * np.pi)

        wavelet_fn = scipy.signal.morlet2
        wavelet_kwargs = dict(w=self.center_frequency)
        W = scipy.signal.cwt(y, wavelet_fn, scales, **wavelet_kwargs, **self.cwt_params)

        times = np.linspace(0, N*dt, N)
        coi = self.get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)

        return CWT(weights=W, times=times, scales=scales, periods=periods, coi=coi)

