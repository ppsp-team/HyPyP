# This class is provided as an example for library comparison and exploration. It is not meant to be use in production code
#
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/matlab/R2023b/bin/glnxa64
# poetry run pip install matlabengine==23.2.3

import os
from typing import List
import numpy as np
from pathlib import Path

try:
    import matlab.engine

    from ..base_wavelet import BaseWavelet, DEFAULT_PERIOD_DJ
    from ..wtc import WTC
    from ..pair_signals import PairSignals

    # In order to work, python must know the location of matlab
    # Add something like this to ~/.bashrc
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_path_to_matlab>/R2023b/bin/glnxa64

    # You also have to download wavelet-coherence code
    # cd <root-of-project>
    # git clone https://github.com/grinsted/wavelet-coherence

    class MatlabWavelet(BaseWavelet):
        def __init__(
            self,
            **kwargs,
        ):
            print("Starting matlab engine")
            self.eng = matlab.engine.start_matlab()
            print("Matlab engine started")

            current_file_path = Path(__file__).resolve()
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
            wavelet_coherence_code_path = os.path.join(root_path, 'wavelet-coherence')

            self.eng.cd(wavelet_coherence_code_path)
            print(f'Changed matlab working directory to {self.eng.pwd()}')
            super().__init__(**kwargs)

        @property
        def wavelet_library(self):
            return 'matlab'

        @property
        def wavelet_name(self):
            return 'cmor'

        @property
        def wavelet_name_with_args(self):
            return 'cmor'

        @property
        def flambda(self):
            raise RuntimeError('Not implemented')

        def evaluate_psi(self):
            self._psi_x = np.arange(10)
            self._psi = np.zeros((10, ))
            return self._psi, self.psi_x
        
        def cwt(self, y, dt):
            raise RuntimeError('Not implemented')

        def wtc(self, pair: PairSignals, bin_seconds:float|None=None, period_cuts:List[float]|None=None, cache_suffix=''):
            y1 = pair.y1
            y2 = pair.y2
            dt = pair.dt
            self.eng.workspace['y1'] = np.ascontiguousarray(y1)
            self.eng.workspace['y2'] = np.ascontiguousarray(y2)
            self.eng.eval("[Rsq, period, scale, coi, sig] = wtc(y1, y2, 'mcc', 0, 'MakeFigure', 1, 'ArrowSize', 0.2, 'S0', 2, 'MaxScale', 1000);  ", nargout=0)

            wtc = np.array(self.eng.workspace['Rsq'])
            N = len(y1)
            times = np.linspace(0, N*dt, N)
            periods = np.array(self.eng.workspace['period']).flatten() * dt
            scales = np.array(self.eng.workspace['scale']).flatten()
            coi = np.array(self.eng.workspace['coi']).flatten()

            # Remove the periods that are out of our period_range
            start = np.searchsorted(periods, self.period_range[0])
            stop = np.searchsorted(periods, self.period_range[1])
            periods = periods[start:stop]
            scales = scales[start:stop]
            wtc = wtc[start:stop,:]

            return WTC(
                wtc,
                times,
                scales,
                periods,
                coi,
                pair,
                wavelet_library=self.wavelet_library,
                wavelet_name_with_args=self.wavelet_name_with_args,
                bin_seconds=bin_seconds,
                period_cuts=period_cuts)

except:
    MatlabWavelet = None
