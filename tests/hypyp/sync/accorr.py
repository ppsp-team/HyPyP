"""
Reference implementations for testing optimized versions.

This module contains the original, unoptimized implementations that serve as
ground truth for validating optimized versions. All optimized implementations
must produce numerically identical results to these reference versions.
"""

import numpy as np
from tqdm import tqdm
from hypyp.analyses import _multiply_conjugate, _multiply_product


def accorr_reference(
    complex_signal: np.ndarray,
    epochs_average: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Reference implementation of Adjusted Circular Correlation (unoptimized).

    This is the original implementation using nested loops for the denominator
    calculation. It serves as the ground truth for testing optimized versions.

    All optimized implementations in hypyp.sync.accorr must produce results
    that match this reference within numerical precision (typically < 1e-9).

    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
        Note: This is the already reshaped signal from compute_sync.

    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
        If False, epoch-by-epoch connectivity is preserved

    show_progress : bool, optional
        If True, display a progress bar during computation (default: False for tests)
        If False, no progress bar is shown

    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix with shape:
        - If epochs_average=True: (n_freq, 2*n_channels, 2*n_channels)
        - If epochs_average=False: (n_freq, n_epochs, 2*n_channels, 2*n_channels)

    Notes
    -----
    The adjusted circular correlation is computed as:

    1. Numerator (vectorized): Uses the difference between the absolute values of
       the conjugate product and the direct product of normalized complex signals.

    2. Denominator (loop): For each channel pair, computes optimal phase centering
       parameters (m_adj, n_adj) that minimize the denominator, then calculates
       the normalization factor.

    This metric provides a more accurate measure of circular correlation by
    adjusting the phase centering for each channel pair individually, rather than
    using a global circular mean.

    References
    ----------
    Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
    Arbitrary methodological decisions skew inter-brain synchronization estimates
    in hyperscanning-EEG studies. Imaging Neuroscience, 2.
    https://doi.org/10.1162/imag_a_00350
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]

    transpose_axes = (0, 1, 3, 2)

    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)

    cross_conj = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)

    cross_prod = _multiply_product(c, s, transpose_axes=transpose_axes)
    r_plus = np.abs(cross_prod)

    num = r_minus - r_plus

    # Denominator (loop) - UNOPTIMIZED REFERENCE VERSION
    angle = np.angle(complex_signal)
    den = np.zeros((n_epochs, n_freq, n_ch_total, n_ch_total))

    total_pairs = (n_ch_total * (n_ch_total + 1)) // 2
    pbar = tqdm(
        total=total_pairs,
        desc="    accorr_reference (denominator)",
        disable=not show_progress,
        leave=False,
    )

    for i in range(n_ch_total):
        for j in range(i, n_ch_total):
            alpha1 = angle[:, :, i, :]
            alpha2 = angle[:, :, j, :]

            phase_diff = alpha1 - alpha2
            phase_sum = alpha1 + alpha2

            mean_diff = np.angle(np.mean(np.exp(1j * phase_diff), axis=2, keepdims=True))
            mean_sum = np.angle(np.mean(np.exp(1j * phase_sum), axis=2, keepdims=True))

            n_adj = -1 * (mean_diff - mean_sum) / 2
            m_adj = mean_diff + n_adj

            x_sin = np.sin(alpha1 - m_adj)
            y_sin = np.sin(alpha2 - n_adj)

            den_ij = 2 * np.sqrt(
                np.sum(x_sin**2, axis=2) * np.sum(y_sin**2, axis=2)
            )
            den[:, :, i, j] = den_ij
            den[:, :, j, i] = den_ij

            pbar.update(1)

    pbar.close()

    den = np.where(den == 0, 1, den)
    con = num / den
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch

    if epochs_average:
        con = np.nanmean(con, axis=1)

    return con
