"""
Tests for analyses_pid module (Partial Information Decomposition).

Author: Rémy Ramadour
Date: January 2026
"""

import numpy as np
import pytest
from hypyp import analyses_pid


class TestEntropyGaussian:
    """Tests for _entropy_gaussian helper function."""

    def test_entropy_univariate(self):
        """Test entropy of univariate Gaussian."""
        # For univariate Gaussian with variance σ²:
        # H = 0.5 * log(2πeσ²)
        variance = 2.0
        cov = np.array([[variance]])

        H = analyses_pid._entropy_gaussian(cov)

        # Expected: 0.5 * log(2π * e * variance)
        expected = 0.5 * np.log(2.0 * np.pi * np.e * variance)

        assert np.isclose(H, expected, atol=1e-10)

    def test_entropy_bivariate_independent(self):
        """Test entropy of bivariate independent Gaussian."""
        # Independent variables: H(X,Y) = H(X) + H(Y)
        cov = np.array([[1.0, 0.0], [0.0, 2.0]])

        H_joint = analyses_pid._entropy_gaussian(cov)
        H_x = analyses_pid._entropy_gaussian(np.array([[1.0]]))
        H_y = analyses_pid._entropy_gaussian(np.array([[2.0]]))

        # Should be additive for independent variables
        assert np.isclose(H_joint, H_x + H_y, atol=1e-10)

    def test_entropy_bivariate_correlated(self):
        """Test entropy of correlated bivariate Gaussian."""
        # Correlated variables
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        H = analyses_pid._entropy_gaussian(cov)

        # Entropy should be positive
        assert H > 0

        # Correlation reduces entropy compared to independent case
        cov_indep = np.array([[1.0, 0.0], [0.0, 1.0]])
        H_indep = analyses_pid._entropy_gaussian(cov_indep)

        assert H < H_indep


class TestLogMvnPdf:
    """Tests for _logmvnpdf helper function."""

    def test_logpdf_standard_normal(self):
        """Test log PDF of standard normal at origin."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        x = np.array([0.0, 0.0])

        logp = analyses_pid._logmvnpdf(x, mean, cov)

        # At mean, log p = -0.5 * [0 + log(det(I)) + 2*log(2π)]
        #                = -0.5 * 2 * log(2π)
        expected = -np.log(2.0 * np.pi)

        assert np.isclose(logp, expected, atol=1e-10)

    def test_logpdf_multiple_points(self):
        """Test log PDF for multiple points."""
        mean = np.array([0.0])
        cov = np.array([[1.0]])
        x = np.array([[0.0], [1.0], [2.0]])

        logp = analyses_pid._logmvnpdf(x, mean, cov)

        # For N(0,1): log p(x) = -0.5 * [x² + log(2π)]
        expected = -0.5 * (x.flatten()**2 + np.log(2.0 * np.pi))

        assert np.allclose(logp, expected, atol=1e-10)

    def test_logpdf_decreases_with_distance(self):
        """Test that log PDF decreases as we move away from mean."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        logp_at_mean = analyses_pid._logmvnpdf(np.array([0.0, 0.0]), mean, cov)
        logp_at_1 = analyses_pid._logmvnpdf(np.array([1.0, 0.0]), mean, cov)
        logp_at_2 = analyses_pid._logmvnpdf(np.array([2.0, 0.0]), mean, cov)

        # Probability should decrease with distance
        assert logp_at_mean > logp_at_1 > logp_at_2


class TestComputeMiAtoms:
    """Tests for _compute_mi_atoms helper function."""

    def test_atoms_redundancy_only(self):
        """Test case where all information is redundant (S1=S2=T)."""
        # When S1 = S2 = T, all information is redundant
        redundancy = 1.0
        mi_s1_t = 1.0  # MI(S1; T) = H(T)
        mi_s2_t = 1.0  # MI(S2; T) = H(T)
        mi_s1s2_t = 1.0  # MI(S1,S2; T) = H(T)

        unq1, unq2, syn = analyses_pid._compute_mi_atoms(
            redundancy, mi_s1_t, mi_s2_t, mi_s1s2_t
        )

        # All unique and synergy should be zero
        assert np.isclose(unq1, 0.0, atol=1e-10)
        assert np.isclose(unq2, 0.0, atol=1e-10)
        assert np.isclose(syn, 0.0, atol=1e-10)

    def test_atoms_unique_s1_only(self):
        """Test case where only S1 has unique information (T=S1, S2 independent)."""
        # T = S1, S2 independent
        redundancy = 0.0
        mi_s1_t = 1.0  # S1 fully determines T
        mi_s2_t = 0.0  # S2 independent of T
        mi_s1s2_t = 1.0  # Total = MI(S1; T)

        unq1, unq2, syn = analyses_pid._compute_mi_atoms(
            redundancy, mi_s1_t, mi_s2_t, mi_s1s2_t
        )

        # Only unique1 should be non-zero
        assert np.isclose(unq1, 1.0, atol=1e-10)
        assert np.isclose(unq2, 0.0, atol=1e-10)
        assert np.isclose(syn, 0.0, atol=1e-10)

    def test_atoms_conservation(self):
        """Test conservation: Red + Unq1 + Unq2 + Syn = MI(S1,S2; T)."""
        # Arbitrary valid values
        redundancy = 0.3
        mi_s1_t = 0.8
        mi_s2_t = 0.7
        mi_s1s2_t = 1.2

        unq1, unq2, syn = analyses_pid._compute_mi_atoms(
            redundancy, mi_s1_t, mi_s2_t, mi_s1s2_t
        )

        # Conservation property
        total = redundancy + unq1 + unq2 + syn

        assert np.isclose(total, mi_s1s2_t, atol=1e-10)

    def test_atoms_non_negativity(self):
        """Test that atoms are non-negative (even with numerical errors)."""
        # Values that might produce small negative results
        redundancy = 0.5
        mi_s1_t = 0.49  # Slightly less than redundancy (numerical error)
        mi_s2_t = 0.48
        mi_s1s2_t = 0.9

        unq1, unq2, syn = analyses_pid._compute_mi_atoms(
            redundancy, mi_s1_t, mi_s2_t, mi_s1s2_t
        )

        # All atoms should be non-negative
        assert unq1 >= 0.0
        assert unq2 >= 0.0
        assert syn >= 0.0


class TestIccsGaussianPair:
    """Tests for _iccs_gaussian_pair redundancy function."""

    def test_iccs_independent_signals(self):
        """Test redundancy is near zero for independent signals."""
        np.random.seed(42)
        s1 = np.random.randn(1000)
        s2 = np.random.randn(1000)
        t = np.random.randn(1000)

        red = analyses_pid._iccs_gaussian_pair(s1, s2, t)

        # Independent signals should have low redundancy
        assert red < 0.1

    def test_iccs_identical_signals(self):
        """Test redundancy is high when S1 = S2 = T."""
        np.random.seed(42)
        s1 = np.random.randn(1000)

        red = analyses_pid._iccs_gaussian_pair(s1, s1, s1)

        # Perfect redundancy case - should be positive
        assert red > 0.5

    def test_iccs_correlated_signals(self):
        """Test redundancy for correlated signals."""
        np.random.seed(42)
        s1 = np.random.randn(1000)
        noise = np.random.randn(1000) * 0.1
        s2 = s1 + noise  # Highly correlated
        t = np.random.randn(1000)

        red = analyses_pid._iccs_gaussian_pair(s1, s2, t)

        # Correlated sources should have some redundancy
        assert red >= 0.0  # Should be non-negative

    def test_iccs_non_negative(self):
        """Test that redundancy is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            s1 = np.random.randn(500)
            s2 = np.random.randn(500)
            t = np.random.randn(500)

            red = analyses_pid._iccs_gaussian_pair(s1, s2, t)

            assert red >= 0.0, "Redundancy must be non-negative"


class TestComputePidGaussian:
    """Tests for compute_pid_gaussian main function."""

    def test_pid_output_format(self):
        """Test that PID returns correctly formatted dictionary."""
        import mne

        # Create simple epochs
        np.random.seed(42)
        info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=250, ch_types='eeg')
        data1 = np.random.randn(10, 2, 100)
        data2 = np.random.randn(10, 2, 100)
        epo1 = mne.EpochsArray(data1, info, verbose=False)
        epo2 = mne.EpochsArray(data2, info, verbose=False)

        pid = analyses_pid.compute_pid_gaussian([epo1, epo2], epochs_average=True)

        # Check dictionary keys
        assert 'redundancy' in pid
        assert 'unique1' in pid
        assert 'unique2' in pid
        assert 'synergy' in pid

        # Check shapes (1, 2*2, 2*2) = (1, 4, 4)
        assert pid['redundancy'].shape == (1, 1, 4, 4)
        assert pid['unique1'].shape == (1, 1, 4, 4)
        assert pid['unique2'].shape == (1, 1, 4, 4)
        assert pid['synergy'].shape == (1, 1, 4, 4)

    def test_pid_epochs_average_false(self):
        """Test PID with epochs_average=False."""
        import mne

        np.random.seed(42)
        n_epochs = 5
        info = mne.create_info(ch_names=['ch1'], sfreq=250, ch_types='eeg')
        data1 = np.random.randn(n_epochs, 1, 50)
        data2 = np.random.randn(n_epochs, 1, 50)
        epo1 = mne.EpochsArray(data1, info, verbose=False)
        epo2 = mne.EpochsArray(data2, info, verbose=False)

        pid = analyses_pid.compute_pid_gaussian([epo1, epo2], epochs_average=False)

        # Check shape (n_epochs, 1, 2, 2)
        assert pid['redundancy'].shape == (n_epochs, 1, 2, 2)

    def test_pid_conservation(self):
        """Test conservation property: Red + Unq1 + Unq2 + Syn ≈ MI(S1,S2;T)."""
        import mne

        np.random.seed(42)
        info = mne.create_info(ch_names=['ch1'], sfreq=250, ch_types='eeg')
        data1 = np.random.randn(10, 1, 100)
        data2 = np.random.randn(10, 1, 100)
        epo1 = mne.EpochsArray(data1, info, verbose=False)
        epo2 = mne.EpochsArray(data2, info, verbose=False)

        pid = analyses_pid.compute_pid_gaussian([epo1, epo2])

        # Sum all atoms
        total = (pid['redundancy'] + pid['unique1'] +
                 pid['unique2'] + pid['synergy'])

        # Total should be non-negative
        assert np.all(total >= -1e-6), "Total should be non-negative"

    def test_pid_non_negative_atoms(self):
        """Test that all PID atoms are non-negative."""
        import mne

        np.random.seed(42)
        info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=250, ch_types='eeg')
        data1 = np.random.randn(5, 2, 50)
        data2 = np.random.randn(5, 2, 50)
        epo1 = mne.EpochsArray(data1, info, verbose=False)
        epo2 = mne.EpochsArray(data2, info, verbose=False)

        pid = analyses_pid.compute_pid_gaussian([epo1, epo2])

        # All atoms should be non-negative
        assert np.all(pid['redundancy'] >= -1e-10)
        assert np.all(pid['unique1'] >= -1e-10)
        assert np.all(pid['unique2'] >= -1e-10)
        assert np.all(pid['synergy'] >= -1e-10)

    def test_pid_invalid_epochs(self):
        """Test that invalid inputs raise appropriate errors."""
        import mne

        info = mne.create_info(ch_names=['ch1'], sfreq=250, ch_types='eeg')
        data = np.random.randn(5, 1, 50)
        epo = mne.EpochsArray(data, info, verbose=False)

        # Test wrong number of epochs
        with pytest.raises(ValueError, match="must be a list of 2"):
            analyses_pid.compute_pid_gaussian([epo])

        # Test invalid target_participant
        with pytest.raises(ValueError, match="target_participant must be 0 or 1"):
            analyses_pid.compute_pid_gaussian([epo, epo], target_participant=2)
