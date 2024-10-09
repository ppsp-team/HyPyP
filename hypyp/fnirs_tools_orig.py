import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import pywt

def xwt_coherence_morl_orig(x1, x2, fs, nNotes=12, detrend=False, normalize=False, tracer=None):
    """
    Calculates the cross wavelet transform coherence between two time series using the Morlet wavelet.

    Arguments:
        x1 : array
            Time series data of the first signal.
        x2 : array
            Time series data of the second signal.
        fs : int
            Sampling frequency of the time series data.
        nNotes : int, optional
            Number of notes per octave for scale decomposition, defaults to 12.
        detrend : bool, optional
            If True, linearly detrends the time series data, defaults to True.
        normalize : bool, optional
            If True, normalizes the time series data by its standard deviation, defaults to True.

    Note:
        This function uses PyWavelets for performing continuous wavelet transforms
        and scipy.ndimage for filtering operations.

    Returns:
        WCT : array
            Wavelet coherence transform values.
        times : array
            Time points corresponding to the time series data.
        frequencies : array
            Frequencies corresponding to the wavelet scales.
        coif : array
            Cone of influence in frequency, reflecting areas in the time-frequency space
            affected by edge artifacts.
    """
    # Assertions and initial computations
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2), "error: arrays not same size"
   
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt
 
    # Data preprocessing: detrend and normalize
    if detrend:
        x1 = signal.detrend(x1, type='linear')
        x2 = signal.detrend(x2, type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2
 
    # Wavelet transform parameters
    nOctaves = int(np.log2(2 * np.floor(N / 2.0)))
    scales = 2 ** np.arange(1, nOctaves, 1.0 / nNotes)
    coef1, freqs1 = pywt.cwt(x1, scales, 'cmor2.5-1.0')
    coef2, freqs2 = pywt.cwt(x2, scales, 'cmor2.5-1.0')
    frequencies = pywt.scale2frequency('cmor2.5-1.0', scales) / dt

    # Compute cross wavelet transform and coherence
    coef12 = coef1 * coef2.conj()
    scaleMatrix = np.ones([1, N]) * scales[:, None]

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

    scales = 1 / freqs1

    s0 = scales[0]
    sN = scales[-1]
    dj = np.log2(sN/s0) / np.size(scales)

    # def smoothing(X, snorm, dj):
    #     return scipy.ndimage.gaussian_filter(X, sigma=[9, 1])
    
    S1 = smoothing(np.abs(coef1) ** 2 / scaleMatrix, scales, dj)
    S2 = smoothing(np.abs(coef2) ** 2 / scaleMatrix, scales, dj)
    S12 = smoothing(coef12 / scaleMatrix, scales, dj)
    if tracer is not None:
      tracer['S1'] = S1
      tracer['S2'] = S2
      tracer['S12'] = S12
    WCT = np.abs(S12) ** 2 / (S1 * S2)

    # Cone of influence calculations
    f0 = 2 * np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    coif = 1.0 / coi
 
    return WCT, times, frequencies, coif