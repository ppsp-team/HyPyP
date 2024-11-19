from math import ceil, floor
import numpy as np

from .base_wavelet import CWT, WTC, BaseWavelet
import pywt
import scipy

class PywaveletsWavelet(BaseWavelet):
    def __init__(
        self,
        wavelet_name='cmor2,1',
        precision=10, # TODO this is not used
        lower_bound=-8,
        upper_bound=8,
        wtc_smoothing_smooth_factor=-0.1, # TODO: this should be calculated automatically, based on the maths
        wtc_smoothing_boxcar_size=1,
        cwt_params=dict(),
        evaluate=True,
        cache_dict=None,
        periods_range=(5, 60),
    ):
        self.wtc_smoothing_smooth_factor = wtc_smoothing_smooth_factor
        self.wtc_smoothing_boxcar_size = wtc_smoothing_boxcar_size
        self.cwt_params = cwt_params
        self.wavelet_name = wavelet_name
        self.precision = precision
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tracer = dict(name='pywt')
        self.periods_range = periods_range
        super().__init__(evaluate, cache_dict)

    def evaluate_psi(self):
        wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        wavelet.lower_bound = self.lower_bound
        wavelet.upper_bound = self.upper_bound
        self._wavelet = wavelet
        self._psi, self._psi_x = wavelet.wavefun(self.precision)
        return self._psi, self._psi_x

    def cwt(self, y, dt, dj=1/12) -> CWT:
        N = len(y)
        times = np.arange(N) * dt
        #nOctaves = int(np.log2(np.floor(N / 2.0)))
        # TODO: find the right s0
        # scales = 2 ** np.arange(1, nOctaves, dj)

        # TODO unhardcode the number of periods
        # TODO see what kind of logspace we want (probably not 10)
        periods = np.logspace(np.log10(self.periods_range[0]), np.log10(self.periods_range[1]), 40)
        frequencies = 1 / periods
        scales = pywt.frequency2scale(self._wavelet, frequencies*dt)

        W, freqs = pywt_copy_cwt(y, scales, self._wavelet, sampling_period=dt, method='fft', tracer=self.tracer, **self.cwt_params)

        # TODO: this is hardcoded, we have to check where this equation comes from
        # Cone of influence calculations
        # TODO: this is probably only valid for morlet wavelet
        f0 = 2 * np.pi
        cmor_coi = 1.0 / np.sqrt(2)
        cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
        coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
        coi = cmor_flambda * cmor_coi * dt * coi
        coif = 1.0 / coi
    
        return CWT(weights=W, times=times, scales=scales, frequencies=freqs, coif=coif, tracer=self.tracer)

fftmodule = scipy.fft
next_fast_len = fftmodule.next_fast_len

def pywt_copy_cwt(data, scales, wavelet, sampling_period=1., method='conv', axis=-1, tracer=None):
    """
    cwt(data, scales, wavelet)

    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    scales : array_like
        The wavelet scales to use. One can use
        ``f = scale2frequency(wavelet, scale)/sampling_period`` to determine
        what physical frequency, ``f``. Here, ``f`` is in hertz when the
        ``sampling_period`` is given in seconds.
    wavelet : Wavelet object or name
        Wavelet to use
    sampling_period : float
        Sampling period for the frequencies output (optional).
        The values computed for ``coefs`` are independent of the choice of
        ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
        period).
    method : {'conv', 'fft'}, optional
        The method used to compute the CWT. Can be any of:
            - ``conv`` uses ``numpy.convolve``.
            - ``fft`` uses frequency domain convolution.
            - ``auto`` uses automatic selection based on an estimate of the
              computational complexity at each scale.

        The ``conv`` method complexity is ``O(len(scale) * len(data))``.
        The ``fft`` method is ``O(N * log2(N))`` with
        ``N = len(scale) + len(data) - 1``. It is well suited for large size
        signals but slightly slower than ``conv`` on small ones.
    axis: int, optional
        Axis over which to compute the CWT. If not given, the last axis is
        used.

    Returns
    -------
    coefs : array_like
        Continuous wavelet transform of the input signal for the given scales
        and wavelet. The first axis of ``coefs`` corresponds to the scales.
        The remaining axes match the shape of ``data``.
    frequencies : array_like
        If the unit of sampling period are seconds and given, then frequencies
        are in hertz. Otherwise, a sampling period of 1 is assumed.

    Notes
    -----
    Size of coefficients arrays depends on the length of the input array and
    the length of given scales.

    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(512)
    >>> y = np.sin(2*np.pi*x/32)
    >>> coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
    >>> plt.matshow(coef)
    >>> plt.show()

    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    >>> widths = np.arange(1, 31)
    >>> cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    >>> plt.show()
    """

    # accept array_like input; make a copy to ensure a contiguous array
    dt = np.dtype('complex128')
    data = np.asarray(data, dtype=dt)
    dt_cplx = np.result_type(dt, np.complex64)
    scales = np.atleast_1d(scales)
    if np.any(scales <= 0):
        raise ValueError("`scales` must only include positive values")

    dt_out = dt_cplx if wavelet.complex_cwt else dt
    out = np.empty((np.size(scales),) + data.shape, dtype=dt_out)
    # Local change: Increased precision to avoid artifacts in high scales (low freq)
    # TODO: this should come as a parameter
    #precision = 15 # avoids artifacts but is slow
    precision = 10
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi

    # convert int_psi, x to the same precision as the data
    dt_psi = dt_cplx if int_psi.dtype.kind == 'c' else dt
    int_psi = np.asarray(int_psi, dtype=dt_psi)
    x = np.asarray(x, dtype=data.real.dtype)

    if method == 'fft':
        size_scale0 = -1
        fft_data = None
    elif method != "conv":
        raise ValueError("method must be 'conv' or 'fft'")

    if data.ndim > 1:
        # move axis to be transformed last (so it is contiguous)
        data = data.swapaxes(-1, axis)

        # reshape to (n_batch, data.shape[-1])
        data_shape_pre = data.shape
        data = data.reshape((-1, data.shape[-1]))

    if tracer is not None:
        tracer['psi_scales'] = []

    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]

        if tracer is not None:
            tracer['psi_scales'].append(int_psi_scale)

        if method == 'conv':
            if data.ndim == 1:
                conv = np.convolve(data, int_psi_scale)
            else:
                # batch convolution via loop
                conv_shape = list(data.shape)
                conv_shape[-1] += int_psi_scale.size - 1
                conv_shape = tuple(conv_shape)
                conv = np.empty(conv_shape, dtype=dt_out)
                for n in range(data.shape[0]):
                    conv[n, :] = np.convolve(data[n], int_psi_scale)
        else:
            # The padding is selected for:
            # - optimal FFT complexity
            # - to be larger than the two signals length to avoid circular
            #   convolution
            size_scale = next_fast_len(
                data.shape[-1] + int_psi_scale.size - 1
            )
            if size_scale != size_scale0:
                # Must recompute fft_data when the padding size changes.
                fft_data = fftmodule.fft(data, size_scale, axis=-1)
            size_scale0 = size_scale
            fft_wav = fftmodule.fft(int_psi_scale, size_scale, axis=-1)
            conv = fftmodule.ifft(fft_wav * fft_data, axis=-1)
            conv = conv[..., :data.shape[-1] + int_psi_scale.size - 1]

        coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
        if out.dtype.kind != 'c':
            coef = coef.real
        # transform axis is always -1 due to the data reshape above
        d = (coef.shape[-1] - data.shape[-1]) / 2.
        if d > 0:
            coef = coef[..., floor(d):-ceil(d)]
        elif d < 0:
            raise ValueError(
                f"Selected scale of {scale} too small.")
        if data.ndim > 1:
            # restore original data shape and axis position
            coef = coef.reshape(data_shape_pre)
            coef = coef.swapaxes(axis, -1)
        out[i, ...] = coef


    frequencies = pywt.scale2frequency(wavelet, scales, precision)
    if np.isscalar(frequencies):
        frequencies = np.array([frequencies])
    frequencies /= sampling_period
    return out, frequencies
