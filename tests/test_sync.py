"""
Tests for synchronization metrics, particularly adjusted circular correlation (accorr).

All optimized implementations are tested against the unoptimized reference
implementation to ensure numerical correctness.
"""

from unittest.mock import patch

import numpy as np
import pytest

from hypyp.analyses import compute_sync
from hypyp.sync.accorr import ACCorr
from hypyp.sync.base import BaseMetric, NUMBA_AVAILABLE, TORCH_AVAILABLE, MPS_AVAILABLE
from tests.accorr_reference import accorr_reference


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
        assert np.all(result >= -1 - 1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    def test_reference_symmetry(self, complex_signal):
        result = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        for freq_idx in range(result.shape[0]):
            matrix = result[freq_idx]
            np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)


class TestAccorrOptimizations:
    """Optimized implementations must match reference."""

    MPS_TOL = 1e-5
    TRANSPOSE_AXES = (0, 1, 3, 2)

    def _compute_with_class(self, complex_signal, optimization=None):
        """Helper: compute accorr using the ACCorr class, then swapaxes to match reference."""
        metric = ACCorr(optimization=optimization, show_progress=False)
        n_samp = complex_signal.shape[3]
        con = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        # ACCorr.compute returns (n_epochs, n_freq, n_ch, n_ch)
        # Reference returns (n_freq, n_epochs, n_ch, n_ch) with epochs_average=False
        return con.swapaxes(0, 1)

    def test_numpy_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_numpy = self._compute_with_class(complex_signal, optimization=None)
        np.testing.assert_allclose(result_numpy, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        result_numba = self._compute_with_class(complex_signal, optimization='numba')
        np.testing.assert_allclose(result_numba, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_torch_vs_reference(self, complex_signal):
        result_reference = accorr_reference(complex_signal, epochs_average=False, show_progress=False)
        metric = ACCorr(optimization='torch', show_progress=False)
        n_samp = complex_signal.shape[3]
        con = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_torch = con.swapaxes(0, 1)

        # MPS uses float32, so tolerance is lower
        if metric._device == 'mps':
            np.testing.assert_allclose(result_torch, result_reference,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_reference,
                                       rtol=1e-9, atol=1e-10)


class TestAccorrViaComputeSync:
    """Test accorr through the compute_sync API with optimization parameter."""

    MPS_TOL = 1e-5

    def test_compute_sync_default(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization=None should match reference."""
        result_reference = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        result = compute_sync(complex_signal_raw, 'accorr', optimization=None, epochs_average=True)
        np.testing.assert_allclose(result, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_compute_sync_numba(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization='numba' should match reference."""
        result_reference = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        result = compute_sync(complex_signal_raw, 'accorr', optimization='numba', epochs_average=True)
        np.testing.assert_allclose(result, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_compute_sync_torch(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization='torch' should match reference."""
        result_reference = accorr_reference(complex_signal, epochs_average=True, show_progress=False)
        result = compute_sync(complex_signal_raw, 'accorr', optimization='torch', epochs_average=True)
        # MPS uses float32, so a looser tolerance is required
        if MPS_AVAILABLE:
            np.testing.assert_allclose(result, result_reference,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result, result_reference, rtol=1e-9, atol=1e-10)


class TestAccorrErrorHandling:
    """Error handling and fallback behavior."""

    def test_invalid_optimization(self):
        with pytest.raises(ValueError, match="Unknown optimization"):
            ACCorr(optimization="invalid_option")

    def test_numba_fallback_warning(self):
        """When numba is unavailable, optimization='numba' warns and falls back to numpy."""
        with patch('hypyp.sync.base.NUMBA_AVAILABLE', False):
            with pytest.warns(UserWarning, match="numba not installed"):
                metric = ACCorr(optimization='numba')
            assert metric._backend == 'numpy'

    def test_torch_fallback_warning(self):
        """When torch is unavailable, optimization='torch' warns and falls back to numpy."""
        with patch('hypyp.sync.base.TORCH_AVAILABLE', False):
            with pytest.warns(UserWarning, match="torch not installed"):
                metric = ACCorr(optimization='torch')
            assert metric._backend == 'numpy'

    def test_auto_resolves(self):
        """optimization='auto' should resolve without error."""
        metric = ACCorr(optimization='auto')
        assert metric._backend in ('numpy', 'numba', 'torch')
