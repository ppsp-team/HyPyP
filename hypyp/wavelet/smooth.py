import numpy as np
from scipy import signal, fft

# TODO: test this
def smoothing(W, dt, dj, scales, smooth_factor=-0.5, boxcar_size=0.6):
    """Smoothing function used in coherence analysis.

    Parameters
    ----------
    W :
    dt :
    dj :
    scales :

    Returns
    -------
    T :

    """
    # The smoothing is performed by using a filter given by the absolute
    # value of the wavelet function at each scale, normalized to have a
    # total weight of unity, according to suggestions by Torrence &
    # Webster (1999) and by Grinsted et al. (2004).
    m, n = W.shape

    # Filter in time.
    # TODO: check that padding is applied here correctly
    def fft_kwargs(signal, **kwargs):
        return {'n': int(2 ** np.ceil(np.log2(len(signal))))}

    my_fft_kwargs = fft_kwargs(W[0, :])

    k = 2 * np.pi * fft.fftfreq(my_fft_kwargs['n'])
    k2 = k ** 2
    snorm = scales / dt

    # Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    gaus_fft = np.exp(smooth_factor * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
    W_fft = fft.fft(W, axis=1, **my_fft_kwargs)
    smooth = fft.ifft(gaus_fft * W_fft, axis=1,  **my_fft_kwargs, overwrite_x=True)
    T = smooth[:, :n]  # Remove possibly padded region due to FFT

    if np.isreal(W).all():
        T = T.real

    # Filter in scale. For the Morlet wavelet it's simply a boxcar with
    # 0.6 width.
    # TODO: check this. It's suspicious
    wsize = boxcar_size / dj * 2
    win = rect(int(np.round(wsize)), normalize=True)
    T = signal.convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

    return T

# TODO: test this
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
from math import ceil, floor

