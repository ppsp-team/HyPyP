""""
Essential Functions for applying MVARICA process on input signals.
"""

import numpy as np
import scipy as sp
from scipy.fftpack import fft


class MVAR:
    """
    Multivariate Vector Autoregressive Model implementation.
    
    This class implements methods for fitting, predicting with, and checking stability of
    MVAR models. MVAR models are useful for analyzing directed interactions between multiple
    time series by modeling how past values of all series affect current values of each series.
    
    Parameters
    ----------
    model_order : int
        Order of the MVAR model, indicating how many past time points influence the current value.
        Higher orders can capture more complex temporal dependencies but require more data to fit.
        
    fitting_method : str or object, optional
        Method used for fitting the MVAR model (default='default'). Options:
        - 'default': Uses least squares method if delta=0, or regularized least squares if delta≠0
        - custom object: Must implement a fit(x, y) method and a coef attribute
        
    delta : float, optional
        Ridge penalty parameter for regularization (default=0):
        - 0: No regularization (standard least squares)
        - >0: Adds L2 regularization to stabilize parameter estimation
    
    Attributes
    ----------
    order : int
        The order of the MVAR model
        
    coeff : ndarray
        The estimated MVAR coefficients with shape (n_channels, n_channels * model_order)
        
    residuals : ndarray
        The residuals after fitting the model, with same shape as input signal
        
    Methods
    -------
    fit(signal)
        Fits the MVAR model to the input signal
        
    predict(signal)
        Predicts values using the fitted MVAR model
        
    stability()
        Checks whether the MVAR model is stable
        
    copy()
        Creates a copy of the MVAR model instance
    
    Notes
    -----
    Stability of an MVAR model is crucial for interpretability. An unstable model
    implies that the system would diverge over time, which is typically not
    physiologically plausible for brain activity.
    
    The fitting process constructs a system of equations based on the input signal
    and the specified model order, then solves for the coefficients.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create simulated data: 3 channels, 2 epochs, 1000 time points each
    >>> data = np.random.randn(2, 3, 1000)
    >>> # Initialize MVAR model with order 5
    >>> model = MVAR(model_order=5, delta=0.1)
    >>> # Fit the model
    >>> model.fit(data)
    >>> # Check stability
    >>> is_stable = model.stability()
    >>> print(f"Model is stable: {is_stable}")
    >>> # Generate predictions
    >>> predicted = model.predict(data)
    """

    def __init__(self, model_order, fitting_method='default', delta=0):
        """
        Initialize an MVAR model with specified parameters.
        
        Parameters
        ----------
        model_order : int
            Order of the MVAR model
            
        fitting_method : str or object, optional
            Method for fitting the MVAR model (default='default')
            
        delta : float, optional
            Ridge penalty parameter (default=0)
        """

        self.order = model_order
        self.fit_method = fitting_method
        self.fitting = None
        self.coeff = np.asarray([])
        self.residuals = np.asarray([])
        self.delta = delta

    def copy(self):
        """
        Create a deep copy of the current MVAR model.
        
        Returns
        -------
        mvar_copy : MVAR
            A new MVAR instance with the same parameters and coefficients
        """

        mvar_copy = self.__class__(self.order)
        mvar_copy.coeff = self.coeff.copy()
        mvar_copy.residuals = self.residuals.copy()
        return mvar_copy

    def predict(self, signal):
        """
        Predict time series data using the fitted MVAR model.
        
        This method applies the fitted MVAR coefficients to predict values
        based on previous time points in the signal.
        
        Parameters
        ----------
        signal : ndarray
            Input signal with shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        predicted : ndarray
            Predicted signal with the same shape as input
            
        Notes
        -----
        Predictions start from the (model_order)th time point, as earlier
        points don't have sufficient history for prediction.
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
        Check the stability of the fitted MVAR model.
        
        An MVAR model is stable if all eigenvalues of the coefficient matrix
        have modulus less than 1. Stability ensures that the model represents
        a stationary process.
        
        Returns
        -------
        is_stable : bool
            True if the model is stable, False otherwise
            
        Notes
        -----
        Stability is a necessary condition for valid connectivity analysis.
        Unstable models can produce misleading connectivity estimates.
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
        """
        Construct the system of equations for MVAR model fitting.
        
        This method reorganizes the input signal into a form suitable for
        least squares estimation of MVAR coefficients.
        
        Parameters
        ----------
        signal : ndarray
            Input signal with shape (n_epochs, n_channels, n_samples)
            
        delta_1 : float or None, optional
            Ridge penalty parameter for regularization
            
        Returns
        -------
        x : ndarray
            Design matrix containing lagged versions of the signal
            
        y : ndarray
            Target matrix containing the values to be predicted
            
        Notes
        -----
        If delta_1 is provided, regularization terms are added to the design matrix
        and target matrix to implement ridge regression.
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
        """
        Fit the MVAR model to input signal data.
        
        This method estimates the MVAR coefficients that best predict the
        signal values based on their past values.
        
        Parameters
        ----------
        signal : ndarray
            Input signal with shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        self : MVAR
            The fitted MVAR model instance
            
        Notes
        -----
        The fitting method depends on the 'fitting_method' parameter:
        - If 'default' and delta=0: Standard least squares
        - If 'default' and delta≠0: Regularized least squares
        - If custom object: Uses the object's fit method
        
        After fitting, the model coefficients and residuals are stored as attributes.
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
    """
    Performs Independent Component Analysis (ICA) on input data.
    
    This function serves as a unified interface to different ICA algorithms
    implemented in external packages like MNE and scikit-learn.
    
    Parameters
    ----------
    ica_input : ndarray
        Input data matrix with shape (n_samples, n_features)
        
    ica_method : str, optional
        ICA algorithm to use (default='infomax_extended'). Options:
        - 'infomax_extended': Extended Infomax algorithm from MNE
        - 'infomax': Standard Infomax algorithm from MNE
        - 'fastica': FastICA algorithm from scikit-learn
        
    random_state : int or None, optional
        Seed for random number generator (default=None):
        - None: Use default random state
        - int: Set specific random seed for reproducibility
    
    Returns
    -------
    unmixing_matrix : ndarray
        Unmixing matrix with shape (n_features, n_features) that transforms
        the input data into independent components
    
    Notes
    -----
    The different ICA methods have varying properties:
    - Extended Infomax can separate both super- and sub-Gaussian sources
    - Standard Infomax works best for super-Gaussian sources
    - FastICA is generally faster but may be less stable for certain data types
    
    Raises
    ------
    ValueError
        If an unsupported ICA method is specified
    
    Examples
    --------
    >>> # Apply Extended Infomax ICA to random data
    >>> data = np.random.randn(1000, 10)  # 1000 samples, 10 features
    >>> unmixing = ica_wrapper(data, ica_method='infomax_extended', random_state=42)
    >>> # Transform data to independent components
    >>> components = data @ unmixing.T
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
    Applies MVARICA approach to estimate connectivity between brain sources.
    
    MVARICA (Multivariate Autoregressive Independent Component Analysis) combines
    MVAR modeling with ICA to jointly estimate source activities and their causal
    interactions. This function implements the full pipeline from signal to
    connectivity measures.
    
    Parameters
    ----------
    real_signal : ndarray
        Input signal with shape (n_epochs, n_channels, n_samples)
        
    ica_params : dict
        Dictionary of ICA parameters with keys:
        - 'method': str, ICA algorithm to use (see ica_wrapper for options)
        - 'random_state': int or None, random seed for reproducibility
        
    measure_name : str
        Connectivity measure to compute. Options:
        - 'mvar_spectral': Spectral representation of VAR coefficients
        - 'mvar_tf': Transfer function
        - 'pdc': Partial directed coherence
        - 'dtf': Directed transfer function
        
    n_fft : int, optional
        Number of frequency bins for connectivity computation (default=512)
        
    var_model : MVAR, optional
        Pre-initialized MVAR model instance (default=MVAR)
    
    Returns
    -------
    result : ndarray
        Connectivity measure matrix with shape dependent on the measure:
        - For all measures: (n_channels, n_channels, n_fft)
        Where each [i, j] entry represents connectivity from channel j to channel i
    
    Notes
    -----
    Process steps:
    1. Fit an MVAR model to the input signals
    2. Extract residuals (innovations) from the fitted model
    3. Apply ICA to the residuals to estimate the mixing matrix
    4. Transform the MVAR coefficients to the source space
    5. Compute the specified connectivity measure in the frequency domain
    
    The different connectivity measures have different interpretations:
    - PDC (Partial Directed Coherence): Measures direct influence from j to i
      normalized by the total outflow from j
    - DTF (Directed Transfer Function): Measures total influence from j to i
      normalized by the total inflow to i
    
    References
    ----------
    Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a new
    concept in neural structure determination. Biological cybernetics, 84(6), 463-474.
    
    Kaminski, M., & Blinowska, K. J. (1991). A new method of the description of
    the information flow in the brain structures. Biological cybernetics, 65(3), 203-210.
    
    Examples
    --------
    >>> # Estimate PDC connectivity from simulated data
    >>> data = np.random.randn(2, 5, 1000)  # 2 epochs, 5 channels, 1000 samples
    >>> mvar_model = MVAR(model_order=5, delta=0.1)
    >>> ica_params = {'method': 'infomax_extended', 'random_state': 42}
    >>> pdc = connectivity_mvarica(data, ica_params, 'pdc', n_fft=128, var_model=mvar_model)
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
