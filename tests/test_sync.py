"""
Tests for synchronization metrics, particularly adjusted circular correlation (accorr).

All optimized implementations are tested against the unoptimized reference
implementation to ensure numerical correctness.
"""

import numpy as np
import pytest

from hypyp.sync.accorr import accorr, NUMBA_AVAILABLE, TORCH_AVAILABLE, MPS_AVAILABLE
from tests.hypyp.sync.accorr import accorr_reference


class TestAccorrReference:
    """Basic properties of the reference implementation."""

    def test_reference_shape_no_average(self, complex_signal):
        result = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_freq, n_epochs, n_ch, n_ch)

    def test_reference_shape_with_average(self, complex_signal):
        result = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_freq, n_ch, n_ch)

    def test_reference_value_range(self, complex_signal):
        result = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        assert np.all(result >= -1) and np.all(result <= 1)
        assert not np.any(np.isnan(result))

    def test_reference_symmetry(self, complex_signal):
        result = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        for freq_idx in range(result.shape[0]):
            matrix = result[freq_idx]
            np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)


class TestAccorrOptimizations:
    """Optimized implementations must match reference."""

    MPS_TOL = 1e-5

    def test_default_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_optimized = accorr(complex_signal, epochs_average=False, show_progress=False, optimization=None)
        np.testing.assert_allclose(result_optimized, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_numba = accorr(complex_signal, epochs_average=False, show_progress=False, optimization="numba")
        np.testing.assert_allclose(result_numba, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_torch_cpu_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_torch = accorr(complex_signal, epochs_average=False, show_progress=False, optimization="torch_cpu")
        np.testing.assert_allclose(result_torch, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE or not MPS_AVAILABLE, reason="Torch or MPS not available")
    def test_torch_mps_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_torch_mps = accorr(complex_signal, epochs_average=False, show_progress=False, optimization="torch_mps")
        np.testing.assert_allclose(result_torch_mps, result_reference, rtol=self.MPS_TOL, atol=self.MPS_TOL)


class TestAccorrFeatures:
    """Specific feature behavior."""

    def test_epochs_averaging(self, complex_signal):
        result_avg = accorr(complex_signal, epochs_average=True, show_progress=False, optimization=None)
        result_no_avg = accorr(complex_signal, epochs_average=False, show_progress=False, optimization=None)

        n_freq = complex_signal.shape[1]
        n_ch = complex_signal.shape[2]
        assert result_avg.shape == (n_freq, n_ch, n_ch)

        manual_avg = np.nanmean(result_no_avg, axis=1)
        np.testing.assert_allclose(result_avg, manual_avg, rtol=1e-10, atol=1e-12)

    def test_epochs_averaging_matches_reference(self, complex_signal):
        result_ref = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        result_opt = accorr(complex_signal, epochs_average=True, show_progress=False, optimization=None)
        np.testing.assert_allclose(result_opt, result_ref, rtol=1e-9, atol=1e-10)


class TestAccorrErrorHandling:
    """Error handling."""

    def test_invalid_optimization(self, complex_signal):
        with pytest.raises(ValueError, match="Optimization parameter is none of the accepted"):
            accorr(complex_signal, epochs_average=False, show_progress=False, optimization="invalid_option")

    @pytest.mark.skipif(NUMBA_AVAILABLE, reason="Test requires numba to be unavailable")
    def test_numba_unavailable(self, complex_signal):
        with pytest.raises(ValueError, match="Numba library not available"):
            accorr(complex_signal, epochs_average=False, show_progress=False, optimization="numba")

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="Test requires torch to be unavailable")
    def test_torch_unavailable(self, complex_signal):
        with pytest.raises(ValueError, match="Torch library not available"):
            accorr(complex_signal, epochs_average=False, show_progress=False, optimization="torch_cpu")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE and MPS_AVAILABLE, reason="Test requires MPS to be unavailable")
    def test_mps_unavailable(self, complex_signal):
        with pytest.raises(ValueError, match="MPS not available"):
            accorr(complex_signal, epochs_average=False, show_progress=False, optimization="torch_mps")