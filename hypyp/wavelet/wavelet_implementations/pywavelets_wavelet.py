
import numpy as np

from ..base_wavelet import DEFAULT_PERIODS_DJ, DEFAULT_PERIODS_RANGE, BaseWavelet
from ..cwt import CWT
import pywt
import scipy


# mother wavelet similar to pycwt and matlab results. Found by trial and error
# TODO check these defaults
#DEFAULT_MORLET_BANDWIDTH = 10
#DEFAULT_MORLET_CENTER_FREQUENCY = 0.25

# TODO: give reference to the equations
# See pywt/pywt/_extensions/c/cwt.template.c, search for "_cmor". The bandwidth ("FB") correspond to the /2 in psi equation
DEFAULT_MORLET_BANDWIDTH = 2
DEFAULT_MORLET_CENTER_FREQUENCY = 1

class PywaveletsWavelet(BaseWavelet):
    def __init__(
        self,
        wavelet_name=f'cmor{DEFAULT_MORLET_BANDWIDTH},{DEFAULT_MORLET_CENTER_FREQUENCY}',
        lower_bound=-8,
        upper_bound=8,
        cwt_params=None,
        **kwargs,
    ):
        if cwt_params is None:
            cwt_params = dict()
        self.cwt_params = cwt_params
        self._wavelet_name = wavelet_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__(**kwargs)

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

    def get_scales(self, dt, dj):
        frequencies = 1 / self.get_periods(dj)
        scales = pywt.frequency2scale(self._wavelet, frequencies*dt)
        return scales

    def cwt(self, y, dt, dj=DEFAULT_PERIODS_DJ, cache_suffix:str='') -> CWT:
        N = len(y)
        times = np.arange(N) * dt
        scales = self.get_scales(dt, dj)
        W, freqs = pywt.cwt(y, scales, self._wavelet, sampling_period=dt, method='conv', **self.cwt_params)
        periods = 1 / freqs

        coi = self.get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)
    
        return CWT(weights=W, times=times, scales=scales, periods=periods, coi=coi)
