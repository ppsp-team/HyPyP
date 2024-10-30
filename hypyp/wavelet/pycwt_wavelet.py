import pycwt
import numpy as np

from .base_wavelet import CWT, BaseWavelet, WTC

class PycwtWavelet(BaseWavelet):
    def __init__(
        self,
        precision=10,
        lower_bound=-8,
        upper_bound=8,
        compute_significance=False,
        evaluate=True,
    ):
        self.precision = precision
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.compute_significance = compute_significance
        self.tracer = dict(name='pycwt')
        self.wavelet_name = 'pycwt'
        super().__init__(evaluate)

    def evaluate_psi(self):
        self._psi_x = np.linspace(self.lower_bound, self.upper_bound, 2**self.precision)
        self._psi = pycwt.wavelet.Morlet().psi(self._psi_x)
        return self._psi, self._psi_x
    
    def cwt(self, y, dt, dj):
        W, scales, freqs, coi, _, _ = pycwt.cwt(y, dt=dt, dj=dj, tracer=self.tracer)
        coif = 1 / coi
        times = np.arange(len(y)) * dt
        return CWT(weights=W, times=times, scales=scales, frequencies=freqs, coif=coif)

    def wtc(self, y1, y2, dt):
        assert (len(y1) == len(y2)), "error: arrays not same size"
        N = len(y1)
        times = np.arange(N) * dt

        wtc, _, coif_periods, frequencies, sig = pycwt.wct(y1, y2, dt=dt, sig=self.compute_significance, tracer=self.tracer)
        coif = 1 / coif_periods
        if not self.compute_significance:
            sig = None
        return WTC(wtc, times, self.tracer['scales'], frequencies, coif, sig, tracer=self.tracer)
