import numpy as np

try:
    import pycwt

    from .base_wavelet import BaseWavelet
    from .wtc import WTC
    from .cwt import CWT
    from .pair_signals import PairSignals

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
            self.wavelet_name = 'pycwt'
            super().__init__(evaluate, disable_caching=True)

        def evaluate_psi(self):
            self._psi_x = np.linspace(self.lower_bound, self.upper_bound, 2**self.precision)
            self._psi = pycwt.wavelet.Morlet().psi(self._psi_x)
            return self._psi, self._psi_x
        
        def cwt(self, y, dt, dj):
            W, scales, freqs, coi, _, _ = pycwt.cwt(y, dt=dt, dj=dj)
            times = np.arange(len(y)) * dt
            return CWT(weights=W, times=times, scales=scales, frequencies=freqs, coi=coi)

        def wtc(self, pair: PairSignals, cache_suffix=''):
            y1 = pair.y1
            y2 = pair.y2
            dt = pair.dt
            assert (len(y1) == len(y2)), "error: arrays not same size"
            N = len(y1)
            times = np.arange(N) * dt

            wtc, _, coi, frequencies, sig = pycwt.wct(y1, y2, dt=dt, sig=self.compute_significance)
            if not self.compute_significance:
                sig = None

            # TODO get scales to send to WTC
            return WTC(wtc, times, [], frequencies, coi, pair, sig=sig)

except:
    PycwtWavelet = None