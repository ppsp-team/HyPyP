from math import ceil, floor
import numpy as np

from ..base_wavelet import BaseWavelet
from ..cwt import CWT
import pywt
import scipy


DEFAULT_PERIODS_RANGE = (2, 20)
DEFAULT_PERIODS_DJ = 1/12
# mother wavelet similar to pycwt and matlab results. Found by trial and error
DEFAULT_MORLET_BANDWIDTH = 10
DEFAULT_MORLET_CENTER_FREQUENCY = 0.25

class PywaveletsWavelet(BaseWavelet):
    def __init__(
        self,
        wavelet_name=f'cmor{DEFAULT_MORLET_BANDWIDTH},{DEFAULT_MORLET_CENTER_FREQUENCY}',
        lower_bound=-8,
        upper_bound=8,
        cwt_params=None,
        evaluate=True,
        periods_range=None,
        frequencies_range=None,
        **kwargs,
    ):
        if cwt_params is None:
            cwt_params = dict()
        self.cwt_params = cwt_params
        self._wavelet_name = wavelet_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if periods_range is not None and frequencies_range is not None:
            raise RuntimeError('Cannot specify both periods_range and frequencies_range')

        if periods_range is not None:
            self.periods_range = periods_range
        elif frequencies_range is not None:
            self.periods_range = (1 / frequencies_range[0], 1 / frequencies_range[1])
        else:
            self.periods_range = DEFAULT_PERIODS_RANGE
        
        if self.periods_range[0] > self.periods_range[1]:
            self.periods_range = (self.periods_range[1], self.periods_range[0])

        super().__init__(evaluate, **kwargs)

    @property
    def wavelet_library(self):
        return 'pywavelets'

    @property
    def wavelet_name(self):
        return self._wavelet_name

    def evaluate_psi(self):
        wavelet = pywt.ContinuousWavelet(self._wavelet_name)
        wavelet.lower_bound = self.lower_bound
        wavelet.upper_bound = self.upper_bound
        self._wavelet = wavelet
        # TODO unhardcode value here
        self._psi, self._psi_x = wavelet.wavefun(10)
        return self._psi, self._psi_x

    def get_periods(self, dj=DEFAULT_PERIODS_DJ):
        low, high =  self.periods_range
        n_scales = np.log2(high/low) 
        n_steps = int(np.round(n_scales / dj))
        periods = np.logspace(np.log2(low), np.log2(high), n_steps, base=2)
        return periods
        
    def get_scales(self, dt, dj):
        frequencies = 1 / self.get_periods(dj)
        scales = pywt.frequency2scale(self._wavelet, frequencies*dt)
        return scales


    def cwt(self, y, dt, dj=DEFAULT_PERIODS_DJ, cache_suffix:str='') -> CWT:
        N = len(y)
        times = np.arange(N) * dt
        scales = self.get_scales(dt, dj)
        W, freqs = pywt.cwt(y, scales, self._wavelet, sampling_period=dt, method='fft', **self.cwt_params)
        periods = 1 / freqs

        coi = self.get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)
    
        return CWT(weights=W, times=times, scales=scales, periods=periods, coi=coi)
