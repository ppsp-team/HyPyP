# This class is provided as an example for library comparison and exploration. It is not meant to be use in production code
#
# poetry run pip install pycwt

from typing import List
import numpy as np

DEFAULT_CENTER_FREQUENCY = 6

try:
    import pycwt

    from ..base_wavelet import BaseWavelet
    from ..wtc import WTC
    from ..cwt import CWT
    from ..pair_signals import PairSignals

    class PycwtWavelet(BaseWavelet):
        def __init__(
            self,
            center_frequency=DEFAULT_CENTER_FREQUENCY,
            precision=10,
            lower_bound=-8,
            upper_bound=8,
            **kwargs,
        ):
            self.center_frequency = center_frequency
            self.precision = precision
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self._wavelet = None
            super().__init__(**kwargs)

        @property
        def wavelet_library(self):
            return 'pycwt'

        @property
        def wavelet_name(self):
            return f'cmor'

        @property
        def wavelet_name_with_args(self):
            return f'cmor{self.center_frequency}'

        @property
        def flambda(self):
            return self._wavelet.flambda()

        def evaluate_psi(self):
            self._psi_x = np.linspace(self.lower_bound, self.upper_bound, 2**self.precision)
            wavelet = pycwt.wavelet.Morlet()
            # TODO: with this parameter, the deltaj0 should change in pycwt code
            wavelet.f0 = self.center_frequency
            self._wavelet = wavelet
            self._psi = wavelet.psi(self._psi_x)
            return self._psi, self._psi_x
        
        def cwt(self, y, dt) -> CWT:
            periods = self.get_periods()

            W, scales, freqs, coi, _, _ = pycwt.cwt(y, dt=dt, freqs=1/periods, wavelet=self._wavelet)
            periods = 1 / freqs
            times = np.arange(len(y)) * dt
            return CWT(weights=W, times=times, scales=scales, periods=periods, coi=coi)

        def smoothing(self, W, dt, scales, _win_size=0, _cache_suffix=''):
            return self._wavelet.smooth(W, dt, self.dj, scales)

        def wtc(self, pair: PairSignals, bin_seconds:float|None=None, period_cuts:List[float]|None=None, cache_suffix=''):
            y1 = pair.y1
            y2 = pair.y2
            dt = pair.dt
            assert (len(y1) == len(y2)), "error: arrays not same size"
            N = len(y1)
            times = np.arange(N) * dt

            wtc, _awtc, coi, freqs, _sig = pycwt.wct(y1, y2, dt=dt, wavelet=self._wavelet, sig=False)
            periods = 1 / freqs

            # Remove the periods that are out of our period_range
            # IMPROVEMENT: This would be better done by sending s0 and J to pycwt.wct()
            start = np.searchsorted(periods, self.period_range[0])
            stop = np.searchsorted(periods, self.period_range[1])
            periods = periods[start:stop]
            freqs = freqs[start:stop]
            wtc = wtc[start:stop,:]

            # TODO get scales to send to WTC
            return WTC(
                wtc,
                times,
                np.array([]),
                periods,
                coi,
                pair,
                wavelet_library=self.wavelet_library,
                wavelet_name_with_args=self.wavelet_name_with_args,
                bin_seconds=bin_seconds,
                period_cuts=period_cuts)

except:
    PycwtWavelet = None