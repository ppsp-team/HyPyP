"""
Tests for information theory connectivity measures.

This module tests the implementation of Mutual Information (MI) and
Transfer Entropy (TE) functions in hypyp.analyses_it.
"""

import pytest
import numpy as np
import mne
from hypyp import analyses_it


class TestMutualInformation:
    """Test suite for Mutual Information computation."""

    def test_mi_independent_signals(self):
        """Test MI between independent signals should be close to 0."""
        # Create independent signals
        np.random.seed(42)
        n_epochs, n_channels, n_times = 50, 2, 500
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute MI
        mi = analyses_it.compute_mi_gaussian([epochs1, epochs2], epochs_average=True)

        # Check shape
        assert mi.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {mi.shape}"

        # MI should be close to 0 for independent signals
        # Off-diagonal blocks (inter-brain) should be near 0
        inter_brain_mi = mi[0, :n_channels, n_channels:]
        assert np.mean(inter_brain_mi) < 0.1, \
            f"Expected low MI for independent signals, got {np.mean(inter_brain_mi)}"

    def test_mi_correlated_signals(self):
        """Test MI between correlated signals should be > 0."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 50, 2, 500
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Create correlated signals
        data1 = np.random.randn(n_epochs, n_channels, n_times)
        noise = np.random.randn(n_epochs, n_channels, n_times) * 0.1
        data2 = data1 + noise  # High correlation

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute MI
        mi = analyses_it.compute_mi_gaussian([epochs1, epochs2], epochs_average=True)

        # MI should be high for correlated signals
        inter_brain_mi = mi[0, :n_channels, n_channels:]
        assert np.mean(inter_brain_mi) > 0.5, \
            f"Expected high MI for correlated signals, got {np.mean(inter_brain_mi)}"

    def test_mi_symmetry(self):
        """Test that MI matrix is symmetric."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 30, 2, 300
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute MI
        mi = analyses_it.compute_mi_gaussian([epochs1, epochs2], epochs_average=True)

        # Check symmetry: mi[i, j] should equal mi[j, i]
        assert np.allclose(mi[0], mi[0].T, atol=1e-10), \
            "MI matrix should be symmetric"

    def test_mi_epochs_average_false(self):
        """Test MI with epochs_average=False."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 10, 2, 200
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute MI without averaging
        mi = analyses_it.compute_mi_gaussian([epochs1, epochs2], epochs_average=False)

        # Check shape
        assert mi.shape == (n_epochs, 1, 4, 4), \
            f"Expected shape ({n_epochs}, 1, 4, 4), got {mi.shape}"

        # Each epoch should have symmetric MI
        for epoch in range(n_epochs):
            assert np.allclose(mi[epoch, 0], mi[epoch, 0].T, atol=1e-10), \
                f"MI matrix for epoch {epoch} should be symmetric"

    def test_mi_invalid_input(self):
        """Test MI with invalid inputs."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 10, 2, 200
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data = np.random.randn(n_epochs, n_channels, n_times)
        epochs = mne.EpochsArray(data, info, verbose=False)

        # Test with wrong number of epochs
        with pytest.raises(ValueError, match="must be a list of 2"):
            analyses_it.compute_mi_gaussian([epochs], epochs_average=True)

        with pytest.raises(ValueError, match="must be a list of 2"):
            analyses_it.compute_mi_gaussian([epochs, epochs, epochs], epochs_average=True)

    def test_mi_shape_mismatch(self):
        """Test MI with mismatched epoch shapes."""
        np.random.seed(42)
        sfreq = 250

        # Different number of channels
        info1 = mne.create_info(ch_names=['ch0', 'ch1'], sfreq=sfreq, ch_types='eeg')
        info2 = mne.create_info(ch_names=['ch0', 'ch1', 'ch2'], sfreq=sfreq, ch_types='eeg')

        data1 = np.random.randn(10, 2, 200)
        data2 = np.random.randn(10, 3, 200)

        epochs1 = mne.EpochsArray(data1, info1, verbose=False)
        epochs2 = mne.EpochsArray(data2, info2, verbose=False)

        with pytest.raises(ValueError, match="shape mismatch"):
            analyses_it.compute_mi_gaussian([epochs1, epochs2], epochs_average=True)


class TestTransferEntropy:
    """Test suite for Transfer Entropy computation."""

    def test_te_independent_signals(self):
        """Test TE between independent signals should be close to 0."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 50, 2, 500
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute TE
        te = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=1, epochs_average=True)

        # Check shape
        assert te.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {te.shape}"

        # TE should be close to 0 for independent signals
        inter_brain_te = te[0, :n_channels, n_channels:]
        assert np.mean(inter_brain_te) < 0.1, \
            f"Expected low TE for independent signals, got {np.mean(inter_brain_te)}"

    def test_te_directional_coupling(self):
        """Test TE detects directional coupling."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 50, 2, 500
        sfreq = 250
        delay = 1

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Create directional coupling: data2 depends on data1 with delay
        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.zeros((n_epochs, n_channels, n_times))
        data2[:, :, delay:] = 0.8 * data1[:, :, :-delay] + \
                              0.2 * np.random.randn(n_epochs, n_channels, n_times - delay)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute TE
        te = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=delay, epochs_average=True)

        # TE from participant 1 to 2 should be higher than reverse
        # te[i, j] = influence from j to i
        # So te[2:, :2] = influence from participant 1 to participant 2
        te_1to2 = te[0, n_channels:, :n_channels]
        te_2to1 = te[0, :n_channels, n_channels:]

        assert np.mean(te_1to2) > np.mean(te_2to1), \
            "TE should be higher in the direction of coupling (1→2)"

    def test_te_asymmetry(self):
        """Test that TE matrix is asymmetric."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 30, 2, 300
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute TE
        te = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=1, epochs_average=True)

        # TE should NOT be symmetric (unlike MI)
        # Allow for small numerical differences
        is_symmetric = np.allclose(te[0], te[0].T, atol=1e-5)
        assert not is_symmetric or np.max(np.abs(te[0] - te[0].T)) > 1e-10, \
            "TE matrix should be asymmetric for general signals"

    def test_te_different_delays(self):
        """Test TE with different delay values."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 30, 2, 500
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute TE with different delays
        te_delay1 = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=1, epochs_average=True)
        te_delay5 = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=5, epochs_average=True)

        # Both should have same shape
        assert te_delay1.shape == te_delay5.shape == (1, 4, 4)

        # Results should differ (different delays capture different dynamics)
        assert not np.allclose(te_delay1, te_delay5), \
            "TE should differ for different delay values"

    def test_te_epochs_average_false(self):
        """Test TE with epochs_average=False."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 10, 2, 200
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data1 = np.random.randn(n_epochs, n_channels, n_times)
        data2 = np.random.randn(n_epochs, n_channels, n_times)

        epochs1 = mne.EpochsArray(data1, info, verbose=False)
        epochs2 = mne.EpochsArray(data2, info, verbose=False)

        # Compute TE without averaging
        te = analyses_it.compute_te_gaussian([epochs1, epochs2], delay=1, epochs_average=False)

        # Check shape
        assert te.shape == (n_epochs, 1, 4, 4), \
            f"Expected shape ({n_epochs}, 1, 4, 4), got {te.shape}"

    def test_te_invalid_delay(self):
        """Test TE with invalid delay values."""
        np.random.seed(42)
        n_epochs, n_channels, n_times = 10, 2, 100
        sfreq = 250

        info = mne.create_info(
            ch_names=[f'ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        data = np.random.randn(n_epochs, n_channels, n_times)
        epochs = mne.EpochsArray(data, info, verbose=False)

        # Test with delay < 1
        with pytest.raises(ValueError, match="delay must be >= 1"):
            analyses_it.compute_te_gaussian([epochs, epochs], delay=0, epochs_average=True)

        # Test with delay >= n_times
        with pytest.raises(ValueError, match="delay .* must be < n_times"):
            analyses_it.compute_te_gaussian([epochs, epochs], delay=n_times, epochs_average=True)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_mi_gaussian_pair_identical_signals(self):
        """Test MI between identical signals."""
        np.random.seed(42)
        x = np.random.randn(1000)
        # Use nearly identical signals (not exactly identical to avoid singular matrix)
        y = x + np.random.randn(1000) * 1e-6

        mi = analyses_it._mi_gaussian_pair(x, y)

        # MI should be very high for nearly identical signals
        assert mi > 5.0, f"Expected high MI for nearly identical signals, got {mi}"

    def test_mi_gaussian_pair_independent_signals(self):
        """Test MI between independent signals."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        mi = analyses_it._mi_gaussian_pair(x, y)

        # MI should be close to 0
        assert mi < 0.1, f"Expected low MI for independent signals, got {mi}"

    def test_te_gaussian_pair_no_coupling(self):
        """Test TE between uncoupled signals."""
        np.random.seed(42)
        source = np.random.randn(1000)
        target = np.random.randn(1000)

        te = analyses_it._te_gaussian_pair(source, target, delay=1)

        # TE should be close to 0
        assert te < 0.1, f"Expected low TE for uncoupled signals, got {te}"

    def test_te_gaussian_pair_strong_coupling(self):
        """Test TE with strong directional coupling."""
        np.random.seed(42)
        delay = 1
        n_samples = 1000

        source = np.random.randn(n_samples)
        target = np.zeros(n_samples)
        target[delay:] = 0.9 * source[:-delay] + 0.1 * np.random.randn(n_samples - delay)

        te = analyses_it._te_gaussian_pair(source, target, delay=delay)

        # TE should be high
        assert te > 0.5, f"Expected high TE for strongly coupled signals, got {te}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
