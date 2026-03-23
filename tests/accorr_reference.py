"""
Reference implementations for testing optimized versions.

This module contains the original, unoptimized implementations that serve as
ground truth for validating optimized versions. All optimized implementations
must produce numerically identical results to these reference versions.
"""

import numpy as np
from tqdm import tqdm
from hypyp.sync.base import multiply_conjugate, multiply_product


def accorr_reference(
    complex_signal: np.ndarray,
    epochs_average: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Reference implementation of Adjusted Circular Correlation (unoptimized).

    This is the original implementation using nested loops for the denominator
    calculation. It serves as the ground truth for testing optimized versions.

    Parameters
    ----------
    complex_signal : np.ndarray
        Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times)
    epochs_average : bool, optional
        If True, connectivity values are averaged across epochs (default)
    show_progress : bool, optional
        If True, display a progress bar (default: False for tests)

    Returns
    -------
    con : np.ndarray
        Adjusted circular correlation matrix.
    """
    n_epochs = complex_signal.shape[0]
    n_freq = complex_signal.shape[1]
    n_ch_total = complex_signal.shape[2]

    transpose_axes = (0, 1, 3, 2)

    # Numerator (vectorized)
    z = complex_signal / np.abs(complex_signal)
    c, s = np.real(z), np.imag(z)

    cross_conj = multiply_conjugate(c, s, transpose_axes=transpose_axes)
    r_minus = np.abs(cross_conj)

    cross_prod = multiply_product(c, s, transpose_axes=transpose_axes)
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
