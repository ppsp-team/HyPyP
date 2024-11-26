from math import ceil, floor
import numpy as np

from .base_wavelet import BaseWavelet
from .cwt import CWT
import scipy.signal

from ..plots import plot_wavelet_coherence

DEFAULT_SCIPY_CENTER_FREQUENCY = 6

class ScipyWavelet(BaseWavelet):
    def __init__(
        self,
        center_frequency=DEFAULT_SCIPY_CENTER_FREQUENCY,
        wtc_smoothing_smooth_factor=-0.1, # TODO: this should be calculated automatically, based on the maths
        wtc_smoothing_boxcar_size=1,
        cwt_params=dict(),
        evaluate=True,
    ):
        self.wtc_smoothing_smooth_factor = wtc_smoothing_smooth_factor
        self.wtc_smoothing_boxcar_size = wtc_smoothing_boxcar_size
        self.cwt_params = cwt_params
        self.center_frequency = center_frequency
        self.wavelet_name = 'morlet_scipy'
        self.tracer = dict(name=self.wavelet_name)
        super().__init__(evaluate)

    def evaluate_psi(self):
        M = 1000
        s = 100
        w = self.center_frequency
        wavelet = scipy.signal.morlet2(M, s, w)
        self._psi_x = np.arange(M)
        self._psi = wavelet

        return self._psi, self._psi_x

    def cwt(self, y, dt, dj=1/12) -> CWT:
        N = len(y)
        fs = 1 / dt
        nOctaves = int(np.log2(np.floor(N / 2.0)))
        scales = 2 ** np.arange(1, nOctaves, dj)
        wavelet_fn = scipy.signal.morlet2
        wavelet_kwargs = dict(w=self.center_frequency)
        W = scipy.signal.cwt(y, wavelet_fn, scales, **wavelet_kwargs, **self.cwt_params)
        freqs = (self.center_frequency * fs) / (2 * np.pi * scales)
        times = np.linspace(0, N*dt, N)

        # serapately compute wavelets for tracing. Code is from scipy/signal/_wavelets.py
        self.tracer['psi_scales'] = []
        for ind, width in enumerate(scales):
            N = np.min([10 * width, len(y)])
            wav = wavelet_fn(N, width, **wavelet_kwargs)[::-1]
            self.tracer['psi_scales'].append(np.conj(wav))

        # TODO: this is hardcoded, we have to check where this equation comes from
        # Cone of influence calculations
        f0 = 2 * np.pi
        cmor_coi = 1.0 / np.sqrt(2)
        cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = cmor_flambda * cmor_coi * dt * coi
        coif = 1.0 / coi

        return CWT(weights=W, times=times, scales=scales, frequencies=freqs, coif=coif, tracer=self.tracer)

