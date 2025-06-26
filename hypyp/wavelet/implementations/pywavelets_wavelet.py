import numpy as np

from ..base_wavelet import BaseWavelet
from ..cwt import CWT
import pywt

# See pywt/pywt/_extensions/c/cwt.template.c, search for "_cmor". The bandwidth ("FB") correspond to the /2 in psi equation
DEFAULT_MORLET_BANDWIDTH_FREQUENCY = 2
DEFAULT_MORLET_CENTER_FREQUENCY = 1
DEFAULT_GAUSSIAN_DEGREE = 2

class PywaveletsWavelet(BaseWavelet):
    """
    Parent class for the default Wavelet implementation, using Pywavelets library.

    ComplexMorletWavelet or ComplexGaussian Wavelet should be used instead of this class directly.

    Args:
        wavelet_name (str, optional): name of the wavelet to send to pywavelets library. Defaults to f'cmor2,1'.
        lower_bound (float, optional): lower bound for mother wavelet evaluation. Defaults to -8.
        upper_bound (float, optional): upper bound for mother wavelet evaluation. Defaults to 8.
        cwt_params (dict | None, optional): params to be sent to cwt(), the Continuous Wavelet Transform. Defaults to None.
    """
    cwt_params: dict
    _wavelet_name: str
    lower_bound: float
    upper_bound: float
    degree: float | None

    def __init__(
        self,
        wavelet_name:str=f'cmor{DEFAULT_MORLET_BANDWIDTH_FREQUENCY},{DEFAULT_MORLET_CENTER_FREQUENCY}',
        lower_bound:float=-8,
        upper_bound:float=8,
        cwt_params:dict|None=None,
        **kwargs,
    ):
        self._wavelet_name = wavelet_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if cwt_params is None:
            cwt_params = dict()
        self.cwt_params = cwt_params

        super().__init__(**kwargs)

    @property
    def wavelet_library(self):
        return 'pywavelets'

    @property
    def wavelet_name_with_args(self):
        name = self._wavelet_name
        if self.wtc_smoothing_win_size is not None:
            name += f'[win:{self.wtc_smoothing_win_size}]'
        return name

    @property
    def wavelet_name(self):
        return self._wavelet_name

    def evaluate_psi(self):
        wavelet = pywt.ContinuousWavelet(self._wavelet_name)
        wavelet.lower_bound = self.lower_bound
        wavelet.upper_bound = self.upper_bound
        self._wavelet = wavelet

        self._psi, self._psi_x = wavelet.wavefun(10)
        return self._psi, self._psi_x

    def get_scales(self, dt):
        frequencies = 1 / self.get_periods()
        scales = pywt.frequency2scale(self._wavelet, frequencies*dt)
        return scales

    def cwt(self, y, dt, cache_suffix:str='') -> CWT:
        N = len(y)
        times = np.arange(N) * dt
        scales = self.get_scales(dt)
        W, freqs = pywt.cwt(y, scales, self._wavelet, sampling_period=dt, method='fft', **self.cwt_params)
        periods = 1 / freqs

        coi = self._get_and_cache_cone_of_influence(N, dt, cache_suffix=cache_suffix)
    
        return CWT(weights=W, times=times, scales=scales, periods=periods, coi=coi)

class ComplexMorletWavelet(PywaveletsWavelet):
    """
    A complex morlet wavelet to compute Continuous Wavelet Transform and Wavelet Transform Coherence

    See PywaveletsWavelet and BaseWavelet classes for possible arguments to constructor

    Args:
        bandwidth_frequency (float, optional): Defaults to 2.
        center_frequency (float, optional): Defaults to 1.
    """
    bandwidth_frequency: float
    center_frequency: float

    default_bandwidth_frequency: float = DEFAULT_MORLET_BANDWIDTH_FREQUENCY
    default_center_frequency: float = DEFAULT_MORLET_CENTER_FREQUENCY

    def __init__(
        self,
        bandwidth_frequency:float=DEFAULT_MORLET_BANDWIDTH_FREQUENCY,
        center_frequency:float=DEFAULT_MORLET_CENTER_FREQUENCY,
        **kwargs,
    ):
        self.bandwidth_frequency = bandwidth_frequency
        self.center_frequency = center_frequency
        return super().__init__(wavelet_name=f'cmor{bandwidth_frequency},{center_frequency}', **kwargs)
    
    @property
    def flambda(self):
        # Equations come from "A Practical Guide to Wavelet Analysis" from Torrence and Compo (1998), Table 1
        f0 = 2 * np.pi * self.center_frequency
        flambda = 2 * np.pi / f0
        #flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        return flambda
        

class ComplexGaussianWavelet(PywaveletsWavelet):
    """
    A complex gaussian wavelet to compute Continuous Wavelet Transform and Wavelet Transform Coherence

    See PywaveletsWavelet and BaseWavelet classes for possible arguments to constructor

    Args:
        degree (int, optional): the "degree" of "Degree of Gaussian" (DOG). Defaults to 2.
    """
    degree: int

    default_degree: int = DEFAULT_GAUSSIAN_DEGREE

    def __init__(
        self,
        degree:int=DEFAULT_GAUSSIAN_DEGREE,
        **kwargs
    ):
        self.degree = degree
        return super().__init__(wavelet_name=f'cgau{degree}', **kwargs)

    @property
    def flambda(self):
        # Equations come from "A Practical Guide to Wavelet Analysis" from Torrence and Compo (1998), Table 1
        m = self.degree
        flambda = 2 * np.pi / np.sqrt(m + 0.5)
        return flambda
