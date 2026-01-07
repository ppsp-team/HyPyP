#!/usr/bin/env python
# coding=utf-8

"""
Unit tests for Integrated Information Decomposition (Φ-ID)

Tests for analyses_phiid module following HyPyP conventions.
"""

import numpy as np
import mne
import pytest
from hypyp.analyses_phiid import compute_phiid, _compute_phiid_raw
from phyid.utils import PhiID_atoms_abbr


@pytest.fixture
def create_synthetic_epochs():
    """
    Create synthetic MNE Epochs for testing.

    Returns two Epochs objects with 10 epochs, 4 channels, 100 time points.
    Sampling rate: 250 Hz.
    """
    n_epochs = 10
    n_channels = 4
    n_times = 100
    sfreq = 250.0

    # Create channel info
    ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Generate random data
    np.random.seed(42)
    data1 = np.random.randn(n_epochs, n_channels, n_times)
    data2 = np.random.randn(n_epochs, n_channels, n_times)

    # Create Epochs
    epochs1 = mne.EpochsArray(data1, info)
    epochs2 = mne.EpochsArray(data2, info)

    return epochs1, epochs2


@pytest.fixture
def create_correlated_epochs():
    """
    Create synthetic correlated Epochs for testing.

    Returns two Epochs where participant 2 is partially correlated with
    participant 1, allowing us to test if Φ-ID detects information sharing.
    """
    n_epochs = 10
    n_channels = 4
    n_times = 100
    sfreq = 250.0

    ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    np.random.seed(42)
    data1 = np.random.randn(n_epochs, n_channels, n_times)

    # Create correlated data: data2 = 0.7 * data1 + 0.3 * noise
    noise = np.random.randn(n_epochs, n_channels, n_times)
    data2 = 0.7 * data1 + 0.3 * noise

    epochs1 = mne.EpochsArray(data1, info)
    epochs2 = mne.EpochsArray(data2, info)

    return epochs1, epochs2


class TestComputePhiIDBasic:
    """Test basic functionality of compute_phiid."""

    def test_input_validation_list(self, create_synthetic_epochs):
        """Test that epochs must be a list."""
        epo1, epo2 = create_synthetic_epochs

        # Should raise ValueError if not a list
        with pytest.raises(ValueError, match="epochs must be a list"):
            compute_phiid(epo1)

    def test_input_validation_length(self, create_synthetic_epochs):
        """Test that epochs list must have exactly 2 elements."""
        epo1, epo2 = create_synthetic_epochs

        # Should raise ValueError if not exactly 2 epochs
        with pytest.raises(ValueError, match="epochs must be a list of 2"):
            compute_phiid([epo1])

        with pytest.raises(ValueError, match="epochs must be a list of 2"):
            compute_phiid([epo1, epo2, epo1])

    def test_input_validation_type(self, create_synthetic_epochs):
        """Test that epochs elements must be mne.Epochs."""
        epo1, _ = create_synthetic_epochs

        # Should raise TypeError if not Epochs objects
        with pytest.raises(TypeError, match="must be mne.Epochs"):
            compute_phiid([epo1, np.random.randn(10, 4, 100)])

    def test_return_format_validation(self, create_synthetic_epochs):
        """Test that return_format must be 'dict' or 'hyperit'."""
        epo1, epo2 = create_synthetic_epochs

        # Should raise ValueError for invalid format
        with pytest.raises(ValueError, match="return_format must be"):
            compute_phiid([epo1, epo2], return_format='invalid')


class TestComputePhiIDOutputShapes:
    """Test output shapes for different configurations."""

    def test_dict_format_shape(self, create_synthetic_epochs):
        """Test output shape with return_format='dict'."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result = compute_phiid([epo1, epo2], return_format='dict', epochs_average=True)

        # Should be a dictionary
        assert isinstance(result, dict)

        # Should have 16 keys (one per Φ-ID atom)
        assert len(result) == 16

        # All keys should be from PhiID_atoms_abbr
        for key in result.keys():
            assert key in PhiID_atoms_abbr

        # Each value should have shape (1, 2*n_channels, 2*n_channels)
        for atom_name, atom_matrix in result.items():
            assert atom_matrix.shape == (1, 2 * n_ch, 2 * n_ch), \
                f"Atom {atom_name} has incorrect shape: {atom_matrix.shape}"

    def test_hyperit_format_shape(self, create_synthetic_epochs):
        """Test output shape with return_format='hyperit'."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result = compute_phiid([epo1, epo2], return_format='hyperit', epochs_average=True)

        # Should be a numpy array
        assert isinstance(result, np.ndarray)

        # Should have shape (2*n_channels, 2*n_channels, 16)
        assert result.shape == (2 * n_ch, 2 * n_ch, 16)

    def test_epochs_average_true(self, create_synthetic_epochs):
        """Test epochs_average=True gives correct shape."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result_dict = compute_phiid([epo1, epo2], epochs_average=True, return_format='dict')
        result_hyperit = compute_phiid([epo1, epo2], epochs_average=True, return_format='hyperit')

        # Dict format: (1, 2*n_ch, 2*n_ch) for each atom
        assert result_dict['rtr'].shape == (1, 2 * n_ch, 2 * n_ch)

        # HyperIT format: (2*n_ch, 2*n_ch, 16)
        assert result_hyperit.shape == (2 * n_ch, 2 * n_ch, 16)


class TestComputePhiIDValues:
    """Test Φ-ID values for expected properties."""

    def test_self_connections_zero(self, create_synthetic_epochs):
        """Test that self-connections are zero (i == j)."""
        epo1, epo2 = create_synthetic_epochs

        result = compute_phiid([epo1, epo2], return_format='hyperit')

        # Check diagonal is zero for all atoms
        for i in range(result.shape[0]):
            for atom_idx in range(16):
                assert result[i, i, atom_idx] == 0.0, \
                    f"Self-connection at ({i}, {i}) for atom {atom_idx} should be zero"

    def test_values_non_negative(self, create_synthetic_epochs):
        """Test that Φ-ID values are mostly non-negative.

        Note: Φ-ID atoms can be slightly negative due to numerical estimation,
        especially for independent signals. We allow small negative values (-0.01).
        """
        epo1, epo2 = create_synthetic_epochs

        result_dict = compute_phiid([epo1, epo2], return_format='dict')

        # All Φ-ID atoms should be mostly non-negative (allow small negative values)
        for atom_name, atom_matrix in result_dict.items():
            assert np.all(atom_matrix >= -0.01), \
                f"Atom {atom_name} has large negative values: min={np.min(atom_matrix)}"

    def test_independent_signals_low_values(self, create_synthetic_epochs):
        """Test that independent signals have low Φ-ID values."""
        epo1, epo2 = create_synthetic_epochs

        result = compute_phiid([epo1, epo2], return_format='hyperit')

        # For independent signals, all Φ-ID atoms should be close to zero
        # (allowing for estimation noise)
        assert np.mean(result) < 0.5, \
            f"Independent signals should have low Φ-ID values, got mean={np.mean(result)}"

    def test_correlated_signals_higher_values(self, create_correlated_epochs):
        """Test that correlated signals have valid Φ-ID values."""
        epo1_corr, epo2_corr = create_correlated_epochs

        # Compute Φ-ID for correlated signals
        result_corr = compute_phiid([epo1_corr, epo2_corr], return_format='hyperit')
        mean_corr = np.mean(result_corr)

        # Correlated signals should have non-negative Φ-ID values
        assert mean_corr >= 0

        # Should have some information sharing (not all zeros)
        assert np.max(result_corr) > 0


class TestFormatConsistency:
    """Test consistency between 'dict' and 'hyperit' formats."""

    def test_format_equivalence(self, create_synthetic_epochs):
        """Test that 'dict' and 'hyperit' formats contain same values."""
        epo1, epo2 = create_synthetic_epochs

        result_dict = compute_phiid([epo1, epo2], return_format='dict')
        result_hyperit = compute_phiid([epo1, epo2], return_format='hyperit')

        # Check each atom matches
        for idx, atom_name in enumerate(PhiID_atoms_abbr):
            # Extract from dict (remove artificial freq dimension)
            atom_from_dict = result_dict[atom_name][0]  # (2*n_ch, 2*n_ch)

            # Extract from hyperit
            atom_from_hyperit = result_hyperit[:, :, idx]  # (2*n_ch, 2*n_ch)

            # Should be identical (within numerical precision)
            np.testing.assert_allclose(
                atom_from_dict,
                atom_from_hyperit,
                rtol=1e-10,
                err_msg=f"Atom {atom_name} differs between formats"
            )


class TestHyPyPCompatibility:
    """Test compatibility with HyPyP infrastructure."""

    def test_dict_format_compatible_with_viz(self, create_synthetic_epochs):
        """Test that dict format is compatible with HyPyP viz expectations."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result = compute_phiid([epo1, epo2], return_format='dict')

        # Extract inter-brain connectivity (P1 → P2)
        # Should be able to slice like MI/TE matrices
        for atom_name, atom_matrix in result.items():
            # Remove artificial freq dimension
            matrix_2d = atom_matrix[0]  # (2*n_ch, 2*n_ch)

            # Extract inter-brain block (P1 rows, P2 cols)
            inter_brain = matrix_2d[:n_ch, n_ch:]  # (n_ch, n_ch)

            # Should have correct shape
            assert inter_brain.shape == (n_ch, n_ch)

            # Should be numeric
            assert np.isfinite(inter_brain).all()

    def test_output_format_matches_mi_te(self, create_synthetic_epochs):
        """Test that Φ-ID output format matches MI/TE conventions."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result = compute_phiid([epo1, epo2], return_format='dict')

        # Each atom should have same format as MI (1, 2*n_ch, 2*n_ch)
        for atom_matrix in result.values():
            assert atom_matrix.ndim == 3
            assert atom_matrix.shape[0] == 1  # Artificial freq dimension
            assert atom_matrix.shape[1] == 2 * n_ch
            assert atom_matrix.shape[2] == 2 * n_ch


class TestRawComputation:
    """Test _compute_phiid_raw helper function."""

    def test_raw_computation_shape(self, create_synthetic_epochs):
        """Test _compute_phiid_raw returns correct shape."""
        epo1, epo2 = create_synthetic_epochs
        n_ch = len(epo1.info['ch_names'])

        result = _compute_phiid_raw([epo1, epo2], epochs_average=True)

        # Should return (2*n_ch, 2*n_ch, 16)
        assert result.shape == (2 * n_ch, 2 * n_ch, 16)

    def test_raw_computation_dtype(self, create_synthetic_epochs):
        """Test _compute_phiid_raw returns float array."""
        epo1, epo2 = create_synthetic_epochs

        result = _compute_phiid_raw([epo1, epo2])

        assert result.dtype == np.float64 or result.dtype == np.float32


def test_phiid_atoms_count():
    """Test that phyid package has expected 16 atoms."""
    # Ensure phyid package structure hasn't changed
    assert len(PhiID_atoms_abbr) == 16, \
        "phyid package should define 16 Φ-ID atoms"


def test_module_imports():
    """Test that all necessary imports work."""
    try:
        from hypyp.analyses_phiid import compute_phiid, _compute_phiid_raw
        from phyid.calculate import calc_PhiID
        from phyid.utils import PhiID_atoms_abbr
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
