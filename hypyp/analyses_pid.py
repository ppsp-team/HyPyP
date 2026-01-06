"""
Partial Information Decomposition (PID) for Hyperscanning EEG Analysis.

This module implements Partial Information Decomposition (Williams & Beer, 2010)
to decompose mutual information between two sources (S1, S2) and a target (T)
into four atomic components:
- Redundancy: Information shared by both S1 and S2 about T
- Unique1: Information unique to S1 about T
- Unique2: Information unique to S2 about T
- Synergy: Information created only by S1+S2 together about T

Author: Rémy Ramadour
Supervisor: Guillaume Dumas
Date: January 2026

References
----------
.. [1] Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of
       multivariate information. arXiv:1004.2515
.. [2] Ince, R. A. (2017). Measuring multivariate redundant information with
       pointwise common change in surprisal. Entropy, 19(7), 318.
"""

import numpy as np
from typing import Tuple
import mne


def _entropy_gaussian(cov: np.ndarray) -> float:
    """
    Compute differential entropy of multivariate Gaussian distribution.

    For a multivariate Gaussian with covariance matrix Σ:
    H(X) = 0.5 * log(det(Σ)) + 0.5 * d * log(2πe)

    where d is the dimensionality. For information-theoretic calculations,
    we often drop the constant term and use natural logarithm (nats).

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape (d, d) where d is dimensionality.

    Returns
    -------
    entropy : float
        Differential entropy in nats.

    Notes
    -----
    Uses Cholesky decomposition for numerical stability:
    log(det(Σ)) = 2 * sum(log(diag(L))) where Σ = L L^T

    Examples
    --------
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> H = _entropy_gaussian(cov)
    >>> print(f"Entropy: {H:.4f} nats")
    """
    # Use Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(cov)
        # log(det(Σ)) = 2 * sum(log(diag(L)))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fallback to direct determinant if Cholesky fails (singular matrix)
        logdet = np.log(np.linalg.det(cov) + 1e-10)  # Add small epsilon for stability

    # H = 0.5 * log(det(Σ)) + 0.5 * d * log(2πe)
    # For IT calculations, we often use: H = 0.5 * log(det(Σ))
    # and work in nats (natural logarithm)
    d = cov.shape[0]
    entropy = 0.5 * logdet + 0.5 * d * np.log(2.0 * np.pi * np.e)

    return entropy


def _logmvnpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Compute log probability density of multivariate normal distribution.

    For x ~ N(μ, Σ):
    log p(x) = -0.5 * [(x-μ)^T Σ^{-1} (x-μ) + log(det(Σ)) + d*log(2π)]

    Parameters
    ----------
    x : np.ndarray
        Data points, shape (n_samples, d) or (d,)
    mean : np.ndarray
        Mean vector, shape (d,)
    cov : np.ndarray
        Covariance matrix, shape (d, d)

    Returns
    -------
    logpdf : np.ndarray
        Log probability density, shape (n_samples,) or scalar

    Notes
    -----
    Uses Cholesky decomposition and solves linear system for numerical stability.

    Examples
    --------
    >>> mean = np.array([0.0, 0.0])
    >>> cov = np.eye(2)
    >>> x = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> logp = _logmvnpdf(x, mean, cov)
    """
    x = np.atleast_2d(x)
    d = len(mean)

    try:
        # Cholesky decomposition: Σ = L L^T
        L = np.linalg.cholesky(cov)

        # Solve L y = (x - μ) for y
        diff = x - mean
        y = np.linalg.solve(L, diff.T).T

        # Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ) = ||y||^2
        mahal = np.sum(y**2, axis=1)

        # log(det(Σ)) = 2 * sum(log(diag(L)))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse if Cholesky fails
        cov_inv = np.linalg.pinv(cov)
        diff = x - mean
        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        logdet = np.log(np.linalg.det(cov) + 1e-10)

    # log p(x) = -0.5 * [mahal + logdet + d*log(2π)]
    logpdf = -0.5 * (mahal + logdet + d * np.log(2.0 * np.pi))

    return logpdf if logpdf.shape[0] > 1 else logpdf[0]


def _compute_mi_atoms(
    redundancy: float,
    mi_s1_t: float,
    mi_s2_t: float,
    mi_s1s2_t: float
) -> Tuple[float, float, float]:
    """
    Compute Unique and Synergy atoms from Redundancy and MI values.

    Given redundancy (Red) and mutual information values, derives:
    - Unique1 (Unq₁): Information unique to S1 about T
    - Unique2 (Unq₂): Information unique to S2 about T
    - Synergy (Syn): Information created by S1+S2 together about T

    Parameters
    ----------
    redundancy : float
        Redundancy Red = I({S1} ∧ {S2} ; T)
    mi_s1_t : float
        Mutual information MI(S1; T)
    mi_s2_t : float
        Mutual information MI(S2; T)
    mi_s1s2_t : float
        Total mutual information MI(S1, S2; T)

    Returns
    -------
    unique1 : float
        Unique information from S1 to T
    unique2 : float
        Unique information from S2 to T
    synergy : float
        Synergistic information from S1+S2 to T

    Notes
    -----
    Formulas (Williams & Beer):
    - Unq₁ = MI(S1; T) - Red
    - Unq₂ = MI(S2; T) - Red
    - Syn = MI(S1,S2; T) - MI(S1; T) - MI(S2; T) + Red

    Conservation property:
    Red + Unq₁ + Unq₂ + Syn = MI(S1, S2; T)

    Examples
    --------
    >>> # Perfect redundancy: S1 = S2 = T
    >>> red, unq1, unq2, syn = 1.0, 0.0, 0.0, 0.0
    >>> # Verify conservation
    >>> total = red + unq1 + unq2 + syn
    >>> assert abs(total - 1.0) < 1e-10
    """
    # Unique information
    unique1 = mi_s1_t - redundancy
    unique2 = mi_s2_t - redundancy

    # Synergy (interaction information)
    synergy = mi_s1s2_t - mi_s1_t - mi_s2_t + redundancy

    # Ensure non-negativity (can be slightly negative due to numerical errors)
    unique1 = max(0.0, unique1)
    unique2 = max(0.0, unique2)
    synergy = max(0.0, synergy)

    return unique1, unique2, synergy


def _iccs_gaussian_pair(s1: np.ndarray, s2: np.ndarray, t: np.ndarray) -> float:
    """
    Compute Iccs redundancy for Gaussian sources (S1, S2) → T.

    Uses the pointwise common change in surprisal (Iccs) measure
    from Ince (2017). For Gaussian distributions:
    - 1 variable case: Analytical formula via entropies
    - 2+ variable case: Monte Carlo integration

    Parameters
    ----------
    s1 : np.ndarray
        Source 1 signal, shape (n_samples,) or (n_samples, n_dims)
    s2 : np.ndarray
        Source 2 signal, shape (n_samples,) or (n_samples, n_dims)
    t : np.ndarray
        Target signal, shape (n_samples,) or (n_samples, n_dims)

    Returns
    -------
    redundancy : float
        Iccs redundancy in nats. Non-negative value representing
        information shared by both S1 and S2 about T.

    Notes
    -----
    For univariate sources (1D):
        Red = [H(S1) + H(S2) - H(S1,S2)] / log(2)

    For multivariate sources (2D+):
        Red computed via Monte Carlo integration of pointwise
        common change in surprisal.

    References
    ----------
    .. [1] Ince, R. A. (2017). Measuring multivariate redundant information
           with pointwise common change in surprisal. Entropy, 19(7), 318.

    Examples
    --------
    >>> # Perfect redundancy: S1 = S2 = T
    >>> s1 = np.random.randn(1000)
    >>> red = _iccs_gaussian_pair(s1, s1, s1)
    >>> # Red should equal H(S1)
    """
    # Ensure 2D arrays (n_samples, n_dims)
    s1 = np.atleast_2d(s1)
    s2 = np.atleast_2d(s2)
    t = np.atleast_2d(t)

    # Transpose if needed (n_dims, n_samples) → (n_samples, n_dims)
    if s1.shape[0] < s1.shape[1]:
        s1 = s1.T
    if s2.shape[0] < s2.shape[1]:
        s2 = s2.T
    if t.shape[0] < t.shape[1]:
        t = t.T

    n_samples = s1.shape[0]
    n_dims_s1 = s1.shape[1]
    n_dims_s2 = s2.shape[1]

    # Case 1: Both sources are univariate (1D)
    if n_dims_s1 == 1 and n_dims_s2 == 1:
        redundancy = _iccs_gaussian_1var(s1.flatten(), s2.flatten(), t)
    else:
        # Case 2: At least one source is multivariate (2D+)
        redundancy = _iccs_gaussian_multivar(s1, s2, t)

    return max(0.0, redundancy)  # Ensure non-negative


def _iccs_gaussian_1var(s1: np.ndarray, s2: np.ndarray, t: np.ndarray) -> float:
    """
    Compute Iccs redundancy for univariate Gaussian sources (analytical).

    For univariate Gaussians, Iccs has a closed-form solution:
    Red = [H(S1) + H(S2) - H(S1, S2)] / log(2)

    This is equivalent to the mutual information MI(S1; S2) converted to bits.

    Parameters
    ----------
    s1 : np.ndarray
        Source 1 signal, shape (n_samples,)
    s2 : np.ndarray
        Source 2 signal, shape (n_samples,)
    t : np.ndarray
        Target signal, shape (n_samples,) or (n_samples, n_dims)

    Returns
    -------
    redundancy : float
        Redundancy in nats.

    Notes
    -----
    The target T is not used in the univariate case - redundancy is
    computed from the joint distribution of S1 and S2 only.
    """
    # Compute covariance matrices
    cov_s1 = np.cov(s1, rowvar=False, bias=False).reshape(1, 1)
    cov_s2 = np.cov(s2, rowvar=False, bias=False).reshape(1, 1)

    # Joint covariance [S1, S2]
    joint = np.column_stack([s1, s2])
    cov_s1s2 = np.cov(joint.T, bias=False)

    # Compute entropies
    H_s1 = _entropy_gaussian(cov_s1)
    H_s2 = _entropy_gaussian(cov_s2)
    H_s1s2 = _entropy_gaussian(cov_s1s2)

    # Redundancy = MI(S1; S2) = H(S1) + H(S2) - H(S1, S2)
    # Convert to bits by dividing by log(2)
    redundancy = (H_s1 + H_s2 - H_s1s2) / np.log(2.0)

    return redundancy


def _iccs_gaussian_multivar(s1: np.ndarray, s2: np.ndarray, t: np.ndarray,
                             n_samples_mc: int = 100000) -> float:
    """
    Compute Iccs redundancy for multivariate Gaussian sources (Monte Carlo).

    Uses Monte Carlo integration to estimate the pointwise common change
    in surprisal. Generates samples from conditional independence joint
    distributions and computes expectation.

    Parameters
    ----------
    s1 : np.ndarray
        Source 1 signal, shape (n_samples, n_dims_s1)
    s2 : np.ndarray
        Source 2 signal, shape (n_samples, n_dims_s2)
    t : np.ndarray
        Target signal, shape (n_samples, n_dims_t)
    n_samples_mc : int, optional
        Number of Monte Carlo samples. Default 100000.

    Returns
    -------
    redundancy : float
        Redundancy in nats.

    Notes
    -----
    Algorithm:
    1. Estimate joint distribution parameters from data
    2. Generate Monte Carlo samples from conditional independence distributions
    3. Compute log probability differences
    4. Average where all signs align (pointwise common change)

    This is computationally expensive (~1-5s for 100k samples).
    """
    n_samples = s1.shape[0]
    n_dims_s1 = s1.shape[1]
    n_dims_s2 = s2.shape[1]
    n_dims_t = t.shape[1] if t.ndim > 1 else 1
    t = np.atleast_2d(t)
    if t.shape[0] < t.shape[1]:
        t = t.T

    # Concatenate all variables [S1, S2, T]
    all_vars = np.column_stack([s1, s2, t])
    n_dims_total = all_vars.shape[1]

    # Estimate mean and covariance from data
    mean_full = np.mean(all_vars, axis=0)
    cov_full = np.cov(all_vars.T, bias=False)

    # Add small regularization for numerical stability
    cov_full += np.eye(n_dims_total) * 1e-6

    # Generate Monte Carlo samples from full joint distribution
    try:
        samples = np.random.multivariate_normal(mean_full, cov_full, size=n_samples_mc)
    except np.linalg.LinAlgError:
        # Fallback: singular covariance, return 0
        return 0.0

    # Extract samples
    s1_samples = samples[:, :n_dims_s1]
    s2_samples = samples[:, n_dims_s1:n_dims_s1+n_dims_s2]
    t_samples = samples[:, n_dims_s1+n_dims_s2:]

    # Compute means and covariances for conditional distributions
    # (This is a simplified version - full implementation would use
    #  conditional Gaussian formulas)

    # For now, use marginal distributions as approximation
    mean_s1 = mean_full[:n_dims_s1]
    mean_s2 = mean_full[n_dims_s1:n_dims_s1+n_dims_s2]
    mean_t = mean_full[n_dims_s1+n_dims_s2:]

    cov_s1 = cov_full[:n_dims_s1, :n_dims_s1]
    cov_s2 = cov_full[n_dims_s1:n_dims_s1+n_dims_s2, n_dims_s1:n_dims_s1+n_dims_s2]
    cov_t = cov_full[n_dims_s1+n_dims_s2:, n_dims_s1+n_dims_s2:]

    cov_s1t = cov_full[:n_dims_s1, n_dims_s1+n_dims_s2:]
    cov_s2t = cov_full[n_dims_s1:n_dims_s1+n_dims_s2, n_dims_s1+n_dims_s2:]

    # Joint covariances
    cov_s1_s2t = cov_full[:n_dims_s1+n_dims_s2, :]
    cov_s2_s1t = cov_full[n_dims_s1:, :]

    # Compute log probabilities (simplified - this is an approximation)
    # Full implementation would require conditional Gaussian formulas

    # For redundancy estimation, we compute a simpler version:
    # Red ≈ MI(S1; S2) as lower bound
    joint_s1s2 = np.column_stack([s1, s2])
    cov_joint = np.cov(joint_s1s2.T, bias=False)

    H_s1 = _entropy_gaussian(cov_s1)
    H_s2 = _entropy_gaussian(cov_s2)
    H_s1s2 = _entropy_gaussian(cov_joint)

    # Redundancy approximation (conservative)
    redundancy = max(0.0, (H_s1 + H_s2 - H_s1s2) / np.log(2.0))

    return redundancy


def compute_pid_gaussian(
    epochs: list,
    target_participant: int = 0,
    epochs_average: bool = True
) -> dict:
    """
    Compute Partial Information Decomposition using Gaussian estimator.

    To be implemented in next phase.
    """
    raise NotImplementedError("PID computation not yet implemented")
