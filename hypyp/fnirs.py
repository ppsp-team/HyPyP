from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import mne
import itertools as itertools
import pywt
from matplotlib.colors import Normalize
from scipy import signal
from scipy.fft import ifft, fft, fftfreq
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Subject:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw
        self.best_ch_names: List[str] | None
        self.events: any # we should know what type this is
        self.epochs: mne.Epochs

    def load_snirf_file(self, filepath):
        self.filepath = filepath        
        self.raw = mne.io.read_raw_fif(filepath, verbose=True, preload=True)
        return self
    
    def set_best_ch_names(self, ch_names):
        self.best_ch_names = ch_names
        return self
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        if self.raw is None:
            raise RuntimeError('Must load raw data first')

        if self.best_ch_names is None:
            raise RuntimeError('No "best channels" has been set')

        ch_picks = mne.pick_channels(self.raw.ch_names, include = self.best_ch_names)
        best_channels = self.raw.copy().pick(ch_picks)
        self.events, self.event_dict = mne.events_from_annotations(best_channels)
        self.epochs = mne.Epochs(
            best_channels,
            self.events,
            event_id = self.event_dict,
            tmin = tmin,
            tmax = tmax,
            baseline = baseline,
            reject_by_annotation=False)
        return self

class DyadFNIRS:
    def __init__(self, s1: Subject, s2: Subject):
        self.s1: Subject = s1
        self.s2: Subject = s2

    
    @property 
    def subjects(self):
        return [self.s1, self.s2]
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        _ = [s.load_epochs(tmin, tmax, baseline) for s in self.subjects]
        return self
    

def rect(length, normalize=False):
    """ Rectangular function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

    Args:
        length (int): length of the rectangular function
        normalize (bool): normalize or not

    Returns:
        rect (array): the (normalized) rectangular function

    """
    rect = np.zeros(length)
    rect[0] = rect[-1] = 0.5
    rect[1:-1] = 1

    if normalize:
        rect /= rect.sum()

    return rect

def smoothing(coeff, snorm, dj, smooth_factor=0.1):
    """ Smoothing function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

    Args
    ----

    coeff : array
        the wavelet coefficients get from wavelet transform **in the form of a1 + a2*1j**
    snorm : array
        normalized scales
    dj : float
        it satisfies the equation [ Sj = S0 * 2**(j*dj) ]

    Returns
    -------

    rect : array
        the (normalized) rectangular function

    """
    def fft_kwargs(signal, **kwargs):
        return {'n': int(2 ** np.ceil(np.log2(len(signal))))}
    
    W = coeff #.transpose()
    m, n = np.shape(W)

    # Smooth in time
    k = 2 * np.pi * fftfreq(fft_kwargs(W[0, :])['n'])
    k2 = k ** 2
    # Notes by Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    
    F = np.exp(-smooth_factor * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product

    smooth = ifft(F * fft(W, axis=1, **fft_kwargs(W[0, :])),
                        axis=1,  # Along Fourier frequencies
                        **fft_kwargs(W[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT
    if np.isreal(W).all():
        T = T.real

    # Smooth in scale
    wsize = 0.6 / dj * 2
    win = rect(int(np.round(wsize)), normalize=True)
    T = signal.convolve2d(T, win[:, np.newaxis], 'same')

    return T