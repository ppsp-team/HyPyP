"""
Information Theory connectivity measures for hyperscanning.

This module implements information-theoretic measures of connectivity
between brain signals from two participants. These measures complement
traditional frequency-domain metrics by capturing non-linear dependencies
in the temporal domain.

Functions
---------
compute_mi_gaussian : Compute Mutual Information using Gaussian estimator
compute_te_gaussian : Compute Transfer Entropy using Gaussian estimator
"""

import numpy as np
from typing import Union, List


def compute_mi_gaussian(epochs, epochs_average: bool = True) -> np.ndarray:
    """
    Compute Mutual Information using Gaussian estimator.

    Mutual Information (MI) measures the total amount of information
    shared between two signals. This implementation assumes Gaussian
    distributions and uses a closed-form formula based on covariance.

    Parameters
    ----------
    epochs : list of mne.Epochs
        List of 2 MNE Epochs objects, one per participant.
        Each Epochs object should have shape (n_epochs, n_channels, n_times).
    epochs_average : bool, optional
        If True, average MI across epochs. If False, return MI per epoch.
        Default is True.

    Returns
    -------
    mi_matrix : np.ndarray
        Mutual Information matrix.
        Shape: (1, 2*n_channels, 2*n_channels) if epochs_average=True
               (n_epochs, 1, 2*n_channels, 2*n_channels) if epochs_average=False
        The first dimension (n_freq=1) is artificial for compatibility
        with HyPyP's stats and visualization functions.
        Matrix is symmetric: mi_matrix[i, j] == mi_matrix[j, i]

    Notes
    -----
    The Gaussian MI estimator assumes that signals follow a multivariate
    Gaussian distribution. For non-Gaussian signals, this may underestimate
    the true MI.

    Formula:
        MI(X,Y) = 0.5 * log(var(X) * var(Y) / det(Cov(X,Y)))

    References
    ----------
    .. [1] Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory.
    .. [2] Dumas et al. (2020). "Towards an informational account of
           interpersonal coordination."

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from hypyp import compute_mi_gaussian
    >>>
    >>> # Create synthetic data
    >>> info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=250, ch_types='eeg')
    >>> data1 = np.random.randn(50, 2, 500)
    >>> data2 = np.random.randn(50, 2, 500)
    >>> epochs1 = mne.EpochsArray(data1, info)
    >>> epochs2 = mne.EpochsArray(data2, info)
    >>>
    >>> # Compute MI
    >>> mi = compute_mi_gaussian([epochs1, epochs2], epochs_average=True)
    >>> print(mi.shape)  # (1, 4, 4)
    """
    # Input validation
    if not isinstance(epochs, list) or len(epochs) != 2:
        raise ValueError("epochs must be a list of 2 mne.Epochs objects")

    # Extract data from Epochs objects
    # Shape: (n_epochs, n_channels, n_times)
    data1 = epochs[0].get_data()
    data2 = epochs[1].get_data()

    # Validate shapes match
    if data1.shape != data2.shape:
        raise ValueError(f"Epochs shape mismatch: {data1.shape} vs {data2.shape}")

    n_epochs, n_channels, n_times = data1.shape
    n_total_channels = 2 * n_channels

    # Concatenate data: (2, n_epochs, n_channels, n_times)
    data = np.array([data1, data2])

    # Compute MI for all channel pairs
    if epochs_average:
        # Average across epochs, then compute MI
        # Output: (1, 2*n_channels, 2*n_channels)
        mi_matrix = np.zeros((1, n_total_channels, n_total_channels))

        for i in range(n_total_channels):
            part_i = 0 if i < n_channels else 1
            chan_i = i % n_channels

            for j in range(n_total_channels):
                part_j = 0 if j < n_channels else 1
                chan_j = j % n_channels

                # Concatenate all epochs for this pair
                # Shape: (n_epochs * n_times,)
                signal_i = data[part_i, :, chan_i, :].flatten()
                signal_j = data[part_j, :, chan_j, :].flatten()

                # Compute MI
                mi_matrix[0, i, j] = _mi_gaussian_pair(signal_i, signal_j)

    else:
        # Compute MI per epoch
        # Output: (n_epochs, 1, 2*n_channels, 2*n_channels)
        mi_matrix = np.zeros((n_epochs, 1, n_total_channels, n_total_channels))

        for epoch in range(n_epochs):
            for i in range(n_total_channels):
                part_i = 0 if i < n_channels else 1
                chan_i = i % n_channels

                for j in range(n_total_channels):
                    part_j = 0 if j < n_channels else 1
                    chan_j = j % n_channels

                    # Get signals for this epoch
                    # Shape: (n_times,)
                    signal_i = data[part_i, epoch, chan_i, :]
                    signal_j = data[part_j, epoch, chan_j, :]

                    # Compute MI
                    mi_matrix[epoch, 0, i, j] = _mi_gaussian_pair(signal_i, signal_j)

    return mi_matrix


def compute_te_gaussian(epochs, delay: int = 1,
                       epochs_average: bool = True) -> np.ndarray:
    """
    Compute Transfer Entropy using Gaussian estimator.

    Transfer Entropy (TE) measures the directional information flow from
    a source signal to a target signal. Unlike Mutual Information, TE is
    asymmetric and captures causal relationships.

    Parameters
    ----------
    epochs : list of mne.Epochs
        List of 2 MNE Epochs objects, one per participant.
        Each Epochs object should have shape (n_epochs, n_channels, n_times).
    delay : int, optional
        Time delay in samples for computing TE. Default is 1.
        This represents the lag between source and target signals.
    epochs_average : bool, optional
        If True, average TE across epochs. If False, return TE per epoch.
        Default is True.

    Returns
    -------
    te_matrix : np.ndarray
        Transfer Entropy matrix.
        Shape: (1, 2*n_channels, 2*n_channels) if epochs_average=True
               (n_epochs, 1, 2*n_channels, 2*n_channels) if epochs_average=False
        The first dimension (n_freq=1) is artificial for compatibility
        with HyPyP's stats and visualization functions.
        Matrix is asymmetric: te_matrix[i, j] = influence from j to i

    Notes
    -----
    The Gaussian TE estimator assumes linear dependencies. For strongly
    non-linear systems, this may underestimate the true TE.

    Formula:
        TE(S→T) = I(T_future; S_past | T_past)
                = H(T_future | T_past) - H(T_future | T_past, S_past)

    References
    ----------
    .. [1] Schreiber, T. (2000). Measuring information transfer.
           Physical Review Letters, 85(2), 461.
    .. [2] Vicente, R., et al. (2011). Transfer entropy—a model-free
           measure of effective connectivity for the neurosciences.

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from hypyp import compute_te_gaussian
    >>>
    >>> # Create synthetic data with directional coupling
    >>> info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=250, ch_types='eeg')
    >>> data1 = np.random.randn(50, 2, 500)
    >>> data2 = np.roll(data1, shift=1, axis=2) + np.random.randn(50, 2, 500) * 0.1
    >>> epochs1 = mne.EpochsArray(data1, info)
    >>> epochs2 = mne.EpochsArray(data2, info)
    >>>
    >>> # Compute TE
    >>> te = compute_te_gaussian([epochs1, epochs2], delay=1, epochs_average=True)
    >>> print(te.shape)  # (1, 4, 4)
    """
    # Input validation
    if not isinstance(epochs, list) or len(epochs) != 2:
        raise ValueError("epochs must be a list of 2 mne.Epochs objects")

    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    # Extract data from Epochs objects
    data1 = epochs[0].get_data()
    data2 = epochs[1].get_data()

    # Validate shapes match
    if data1.shape != data2.shape:
        raise ValueError(f"Epochs shape mismatch: {data1.shape} vs {data2.shape}")

    n_epochs, n_channels, n_times = data1.shape
    n_total_channels = 2 * n_channels

    # Check if delay is valid
    if delay >= n_times:
        raise ValueError(f"delay ({delay}) must be < n_times ({n_times})")

    # Concatenate data: (2, n_epochs, n_channels, n_times)
    data = np.array([data1, data2])

    # Compute TE for all channel pairs
    if epochs_average:
        # Average across epochs, then compute TE
        # Output: (1, 2*n_channels, 2*n_channels)
        te_matrix = np.zeros((1, n_total_channels, n_total_channels))

        for i in range(n_total_channels):
            part_i = 0 if i < n_channels else 1
            chan_i = i % n_channels

            for j in range(n_total_channels):
                part_j = 0 if j < n_channels else 1
                chan_j = j % n_channels

                # Concatenate all epochs for this pair
                # Shape: (n_epochs * n_times,)
                target = data[part_i, :, chan_i, :].flatten()
                source = data[part_j, :, chan_j, :].flatten()

                # Compute TE (source → target)
                te_matrix[0, i, j] = _te_gaussian_pair(source, target, delay)

    else:
        # Compute TE per epoch
        # Output: (n_epochs, 1, 2*n_channels, 2*n_channels)
        te_matrix = np.zeros((n_epochs, 1, n_total_channels, n_total_channels))

        for epoch in range(n_epochs):
            for i in range(n_total_channels):
                part_i = 0 if i < n_channels else 1
                chan_i = i % n_channels

                for j in range(n_total_channels):
                    part_j = 0 if j < n_channels else 1
                    chan_j = j % n_channels

                    # Get signals for this epoch
                    # Shape: (n_times,)
                    target = data[part_i, epoch, chan_i, :]
                    source = data[part_j, epoch, chan_j, :]

                    # Compute TE (source → target)
                    te_matrix[epoch, 0, i, j] = _te_gaussian_pair(source, target, delay)

    return te_matrix


def _mi_gaussian_pair(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute MI between two 1D signals assuming Gaussian distribution.

    Parameters
    ----------
    x : np.ndarray
        First signal, shape (n_samples,)
    y : np.ndarray
        Second signal, shape (n_samples,)

    Returns
    -------
    mi : float
        Mutual Information in nats (natural logarithm)
    """
    # Stack signals
    data = np.column_stack([x, y])

    # Compute covariance matrix
    cov = np.cov(data.T)

    # Extract variances
    var_x = cov[0, 0]
    var_y = cov[1, 1]

    # Compute determinant
    det_cov = np.linalg.det(cov)

    # Handle edge cases
    if det_cov <= 0 or var_x <= 0 or var_y <= 0:
        return 0.0

    # MI formula for Gaussian: 0.5 * log(var_x * var_y / det_cov)
    mi = 0.5 * np.log(var_x * var_y / det_cov)

    return max(0.0, mi)  # MI cannot be negative


def _te_gaussian_pair(source: np.ndarray, target: np.ndarray,
                     delay: int = 1) -> float:
    """
    Compute TE from source to target assuming Gaussian distribution.

    Parameters
    ----------
    source : np.ndarray
        Source signal, shape (n_samples,)
    target : np.ndarray
        Target signal, shape (n_samples,)
    delay : int, optional
        Time delay in samples, default is 1

    Returns
    -------
    te : float
        Transfer Entropy in nats (natural logarithm)
    """
    # Construct time-lagged signals
    n = len(target) - delay

    t_future = target[delay:]
    t_past = target[:-delay]
    s_past = source[:-delay]

    # TE = H(T_future | T_past) - H(T_future | T_past, S_past)
    # Using Gaussian formula with conditional variance

    # Compute conditional variance: H(T_future | T_past)
    data_t = np.column_stack([t_future, t_past])
    cov_t = np.cov(data_t.T)

    # Residual variance after predicting T_future from T_past
    if cov_t[1, 1] > 0:
        var_residual_1 = cov_t[0, 0] - (cov_t[0, 1] ** 2) / cov_t[1, 1]
    else:
        var_residual_1 = cov_t[0, 0]

    # Compute conditional variance: H(T_future | T_past, S_past)
    data_full = np.column_stack([t_future, t_past, s_past])
    cov_full = np.cov(data_full.T)

    # Residual variance after predicting T_future from T_past and S_past
    cov_predictors = cov_full[1:, 1:]  # Cov([T_past, S_past])
    cov_cross = cov_full[0, 1:]  # Cov(T_future, [T_past, S_past])

    # Compute conditional variance using Schur complement
    try:
        var_residual_2 = cov_full[0, 0] - cov_cross @ np.linalg.inv(cov_predictors) @ cov_cross.T
    except np.linalg.LinAlgError:
        # Singular matrix, return 0
        return 0.0

    # Handle edge cases
    if var_residual_1 <= 0 or var_residual_2 <= 0:
        return 0.0

    # TE = 0.5 * log(var_residual_1 / var_residual_2)
    te = 0.5 * np.log(var_residual_1 / var_residual_2)

    return max(0.0, te)  # TE cannot be negative
