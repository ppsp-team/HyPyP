import pycwt
import numpy as np

import matlab.engine

from hypyp.signal import SynteticSignal
from .base_wavelet import CWT, BaseWavelet, WCT

class MatlabWavelet(BaseWavelet):
    def __init__(
        self,
        evaluate=True,
    ):
        self.tracer = dict(name='matlab')
        self.wavelet_name = 'matlab'
        print("Starting matlab engine")
        self.eng = matlab.engine.start_matlab()
        print("Matlab engine started")
        self.eng.cd('/home/patrice/work/ppsp/matlab-wct/Archive/wavelet-coherence/wavelet-coherence-master/')
        super().__init__(evaluate)

    def evaluate_psi(self):

        self._psi_x = np.arange(10)
        self._psi = np.zeros((10, ))
        return self._psi, self.psi_x

        #dt = signal.x[1] - signal.x[0]
        #self.eng.workspace['y'] = signal.y
        #self.eng.workspace['dt'] = dt
        #self.eng.workspace['pad'] = 1
        #self.eng.workspace['dj'] = 1/12
        #self.eng.workspace['s0'] = 2*dt
        #self.eng.eval("[y, period, scale, coi] = wavelet(y, dt, pad, dj, s0);", nargout=0)
        #self._psi_x = signal.x
        #self._psi = np.array(self.eng.workspace['y'])

        #self.eng.eval("[psi, xval] = wavefun('morl',10);", nargout=0)
        #self._psi = np.array(self.eng.workspace['psi']).flatten()
        #self._psi_x = np.array(self.eng.workspace['xval']).flatten()

        # TODO: Try this instead, it comes from private/wavelet.m
        # self.eng.eval("	[daughter,fourier_factor,coi,dofmin]=wave_bases(mother,k,scale(a1),param);	")

        return self._psi, self.psi_x
    
    def cwt(self, y, dt, dj):
        pass

    def wct(self, y1, y2, dt):
        #self.eng.eval("times_vec = linspace(0, 200, 1000); ", nargout=0)
        #self.eng.eval("freq1 = 0.1; ", nargout=0)
        #self.eng.eval("freq2 = 0.2; ", nargout=0)
        #self.eng.eval("y1 = sin(times_vec * 2 * pi * freq1/5);", nargout=0)
        #self.eng.eval("y2 = sin(times_vec * 2 * pi * freq2/5);", nargout=0)
        self.eng.workspace['y1'] = np.ascontiguousarray(y1)
        self.eng.workspace['y2'] = np.ascontiguousarray(y2)
        self.eng.eval("[Rsq, period, scale, coi, ~] = wtc(y1, y2, 'mcc', 0, 'MakeFigure', 1, 'ArrowSize', 0.2, 'S0', 2, 'MaxScale', 966.5);  ", nargout=0)

        wct = np.array(self.eng.workspace['Rsq'])
        N = len(y1)
        times = np.linspace(0, N*dt, N)
        periods = np.array(self.eng.workspace['period']).flatten()
        scales = np.array(self.eng.workspace['scale']).flatten()
        coif = 1 / np.array(self.eng.workspace['coi']).flatten()
        frequencies = 1 / periods
        return WCT(wct, times, scales, frequencies, coif, self.tracer)
