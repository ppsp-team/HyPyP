""""
Essential Functions for applying MVARICA process on input signals.
"""

import numpy as np
import scipy as sp
from scipy.fftpack import fft


class MVAR:
    """
    Implementing a multivariate vector autoregressive model.

    Arguments:
        model_order: Int, defines order of MVAR model.

        fitting_method: String, the method that is used for fitting the data to MVAR model. Options in note.

        delta: Float, ridge penalty parameter.

    Returns:
        class: MVAR
        An instance of MVAR with predefined arguments.

    Note:
        *** fitting method options ***
        - 'default':
        If delta = 0 or None, least square method is used.
        If delta !=0, regularized least square is used.

        - fitting object:
        User can implement his/her own fitting method as an python class with the following requirements:
        a fit(x,y) method: fits the linear model with desired algorithm.
        a coef attribute for saving the estimated coefficients for the problem.

    """

    def __init__(self, model_order, fitting_method='default', delta=0):
        self.order = model_order
        self.fit_method = fitting_method
        self.fitting = None
        self.coeff = np.asarray([])
        self.residuals = np.asarray([])
        self.delta = delta

    def copy(self):
        """"
        creates a copy of model.
        """
        mvar_copy = self.__class__(self.order)
        mvar_copy.coeff = self.coeff.copy()
        mvar_copy.residuals = self.residuals.copy()
        return mvar_copy

    def predict(self, signal):
        """"
        Predicts data by MVAR model on input signal.

        Arguments:
            signal: ndarray with shape of (epochs, channels, samples).

        Returns:
            predicted: ndarray with the shape same as signal.
        """

        epoch, channel, sample = signal.shape
        coeff_shape = self.coeff.shape
        p = int(coeff_shape[1] / channel)
        predicted = np.zeros(signal.shape)
        if epoch > sample - channel:
            for i in range(1, p + 1):
                bp = self.coeff[:, (i - 1)::p]
                for j in range(p, sample):
                    predicted[:, :, j] += np.dot(signal[:, :, j - i], bp.T)
        else:
            for i in range(1, p + 1):
                bp = self.coeff[:, (i - 1)::p]
                for j in range(epoch):
                    predicted[j, :, p:] += np.dot(bp, signal[j, :, (p - i):(sample - i)])
        return predicted

    def stability(self):
        """
        Checks whether the MVAR model is stable or not.
        This function basically checks if all eigenvalue of coef. matrix have modulus less than one.

        Returns:
            bool.
            True/False
        """
        co_0, co_1 = self.coeff.shape
        p = co_1 // co_0
        assert (co_1 == co_0 * p)
        top_block = []
        for i in range(p):
            top_block.append(self.coeff[:, i::p])
        top_block = np.hstack(top_block)
        im = np.eye(co_0)
        eye_block = im
        for i in range(p - 2):
            eye_block = sp.linalg.block_diag(im, eye_block)
        eye_block = np.hstack([eye_block, np.zeros((co_0 * (p - 1), co_0))])
        tmp = np.vstack([top_block, eye_block])
        check_stability = np.all(np.abs(np.linalg.eig(tmp)[0]) < 1)
        return check_stability

    def construct_equation(self, signal, delta_1=None):
        """"
        Builds the MVAR equation system.

        Arguments:
            signal: ndarray with shape of (epochs, channels, samples).
            delta_1: Float, ridge penalty parameter.
        """
        mvar_order = self.order
        epoch, channel, sample = signal.shape
        n = (sample - mvar_order) * epoch
        rows = n if delta_1 is None else n + channel * mvar_order
        x = np.zeros((rows, channel * mvar_order))
        for i in range(channel):
            for j in range(1, mvar_order + 1):
                x[:n, i * mvar_order + j - 1] = np.reshape(signal[:, i, mvar_order - j:-j].T, n)
        if delta_1 is not None:
            np.fill_diagonal(x[n:, :], delta_1)
        y = np.zeros((rows, channel))
        for z in range(channel):
            y[:n, z] = np.reshape(signal[:, z, mvar_order:].T, n)
        return x, y

    def fit(self, signal):
        """"
        Fit MVAR model to input signal.

        Arguments:
            signal: ndarray with shape of (epochs, channels, samples).

        Returns:
            self: class:MVAR
        """
        if self.fit_method.lower() == 'default':
            if self.delta == 0 or self.delta is None:
                x, y = self.construct_equation(signal)
            else:
                x, y = self.construct_equation(signal, self.order, self.delta)
            coeff, res, rank, s = sp.linalg.lstsq(x, y)

            self.coeff = coeff.transpose()
            self.residuals = signal - self.predict(signal)

            return self

        else:
            x, y = self.construct_equation(signal)
            self.fitting = self.fit_method.fit(x, y)
            self.coeff = self.fitting.coef
            self.residuals = signal - self.predict(signal)

            return self


def ica_wrapper(ica_input, ica_method='infomax_extended', random_state=None):
    """"
    Performs ICA on the input.
    Arguments:
        ica_input: ndarray, shape(samples, features)

        ica_method: String, the method by which the ICA is performed on the input.

        random_state: int/None, this  parameter is used as the seed in numpy.random.RandomState(seed). Default: None.

    Returns:
        result: unmixing_matrix, ndarray, shape (features, features)
    """
    if ica_method.lower() == 'infomax_extended':
        from mne.preprocessing.infomax_ import infomax
        return infomax(ica_input, extended=True, random_state=random_state)
    elif ica_method.lower() == 'infomax':
        from mne.preprocessing.infomax_ import infomax
        return infomax(ica_input, extended=False, random_state=random_state)
    elif ica_method.lower() == 'fastica':
        from sklearn.decomposition import FastICA
        aux = FastICA(random_state=random_state)
        aux.fit(ica_input)
        return aux.components_
    else:
        raise ValueError(
            'This method is not defined!' + '\n' + 'supported methods: infomax, fastica, picard, infomax_extended')


def connectivity_mvarica(real_signal, ica_params, measure_name, n_fft=512, var_model=MVAR):
    """
    Applies MVARICA approach that uses MVAR models and ICA to jointly estimate sources and connectivity measures.

    Arguments:
        - real_signal: real-value ndarray with the shape of (epochs or trials, frequency, channels, time samples)

        - ica_params: a python dictionary consisting name of desired ica method and value of random_state parameter.

        - measure_name: name of desired connectivity measure. Supported connectivity measures are mentioned in Note.

        - var_model: an instance of predefined VAR/MVAR model.

        - n_fft: number of frequency bins for computing connectivity measures (for fft). default: 512

    Returns:
        result: assigned measure matrix,
        ndarray with the shape of (epochs or trials, frequency, channels, channels, n_fft)

    Note:
        ***available measures***

        'mvar_spectral' : Spectral representation of the VAR coefficients
        'mvar_tf': Transfer function
        'pdc' : Partial directed coherence
        'dtf' : Directed transfer function
    """
    fit_var = var_model.fit(real_signal)
    res = real_signal - var_model.predict(real_signal)

    unmix_matrix = ica_wrapper(np.concatenate(np.split(res, res.shape[0], 0), axis=2).squeeze(0).T,
                               ica_method=ica_params["method"], random_state=ica_params["random_state"]).T
    mix_matrix = sp.linalg.pinv(unmix_matrix)
    trns_unmix_matrix = unmix_matrix.T
    e = np.concatenate([trns_unmix_matrix.dot(res[i, ...])[np.newaxis, ...] for i in range(res.shape[0])])

    fit_var_b = fit_var.copy()
    for k in range(0, fit_var.order):
        fit_var_b.coeff[:, k::fit_var.order] = mix_matrix.dot(fit_var.coeff[:, k::fit_var.order].transpose()).dot(
            unmix_matrix).transpose()

    noise_cov = np.cov(np.concatenate(np.split(e, e.shape[0], 0), axis=2).squeeze(0).T, rowvar=False)

    coeffs = fit_var_b.coeff
    coeffs = np.asarray(coeffs)
    coshape_0, coshape_1 = coeffs.shape
    p = coshape_1 // coshape_0
    assert (coshape_1 == coshape_0 * p)

    re_coeffs = np.reshape(coeffs, (coshape_0, coshape_0, p), 'c')
    a = fft(np.dstack([np.eye(coshape_0), -re_coeffs]), n_fft * 2 - 1)[:, :, :n_fft]
    h = np.array([sp.linalg.solve(a, np.eye(a.shape[0])) for a in a.T]).T

    if measure_name.lower() == 'mvar_spectral':
        result = a
    elif measure_name.lower() == 'mvar_tf':
        result = h
    elif measure_name.lower() == 'pdc':
        result = np.abs(a / np.sqrt(np.sum(a.conj() * a, axis=0, keepdims=True)))
    elif measure_name.lower() == 'dtf':
        result = np.abs(h / np.sqrt(np.sum(h * h.conj(), axis=1, keepdims=True)))

    return result
