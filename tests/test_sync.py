"""
Tests for synchronization metrics, particularly adjusted circular correlation (accorr).

All optimized implementations are tested against the unoptimized reference
implementation to ensure numerical correctness.
"""

from unittest.mock import patch

import numpy as np
import pytest

from hypyp.analyses import compute_sync
from hypyp.sync import get_metric
from hypyp.sync.accorr import ACCorr
from hypyp.sync.base import (
    BaseMetric, AUTO_PRIORITY,
    NUMBA_AVAILABLE, TORCH_AVAILABLE, MPS_AVAILABLE, METAL_AVAILABLE,
)
from hypyp.sync.kernels import CUPY_AVAILABLE
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


class TestPLV:
    """Tests for Phase Locking Value with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5  # PLV uses smooth operations (sin, cos, abs) — tight tolerance

    def test_plv_shape(self, complex_signal):
        """PLV output shape should match input dimensions."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result = PLV().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_plv_value_range(self, complex_signal):
        """PLV values should be in [0, 1]."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result = PLV().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_plv_numba_vs_numpy(self, complex_signal):
        """Numba PLV should match numpy PLV exactly (both float64)."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = PLV(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_plv_torch_vs_numpy(self, complex_signal):
        """Torch PLV should match numpy PLV within MPS tolerance."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = PLV(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    def test_plv_symmetry(self, complex_signal):
        """PLV matrix should be symmetric (PLV(i,j) == PLV(j,i))."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result = PLV().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_plv_metal_vs_numpy(self, complex_signal):
        """Metal PLV should match numpy PLV within float32 tolerance."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = PLV(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_plv_cuda_vs_numpy(self, complex_signal):
        """CUDA PLV should match numpy PLV exactly (both float64)."""
        from hypyp.sync.plv import PLV
        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = PLV(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestCCorr:
    """Tests for circular correlation metric."""

    TRANSPOSE_AXES = (0, 1, 3, 2)

    def test_ccorr_shape(self, complex_signal):
        """CCorr output shape should match input dimensions."""
        from hypyp.sync.ccorr import CCorr
        metric = CCorr()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_ccorr_value_range(self, complex_signal):
        """CCorr values should be non-negative (abs of correlation)."""
        from hypyp.sync.ccorr import CCorr
        metric = CCorr()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10)
        assert not np.any(np.isnan(result))

    def test_ccorr_vs_scipy_reference(self, complex_signal):
        """New inline circmean should match scipy.stats.circmean exactly."""
        from scipy.stats import circmean
        from hypyp.sync.ccorr import CCorr

        # Compute with new implementation
        metric = CCorr()
        n_samp = complex_signal.shape[3]
        result_new = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)

        # Compute reference using scipy circmean
        n_epoch, n_freq, n_ch_total = complex_signal.shape[:3]
        angle = np.angle(complex_signal)
        mu_angle_scipy = circmean(angle, high=np.pi, low=-np.pi, axis=3).reshape(
            n_epoch, n_freq, n_ch_total, 1
        )
        angle_centered = np.sin(angle - mu_angle_scipy)
        formula = 'nilm,nimk->nilk'
        transpose_axes = self.TRANSPOSE_AXES
        result_scipy = np.abs(
            np.einsum(formula, angle_centered, angle_centered.transpose(transpose_axes)) /
            np.sqrt(np.einsum('nil,nik->nilk',
                              np.sum(angle_centered ** 2, axis=3),
                              np.sum(angle_centered ** 2, axis=3)))
        )

        np.testing.assert_allclose(result_new, result_scipy, rtol=1e-12, atol=1e-14)

    def test_ccorr_symmetry(self, complex_signal):
        """CCorr matrix should be symmetric for each epoch/freq."""
        from hypyp.sync.ccorr import CCorr
        metric = CCorr()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )


    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_ccorr_numba_vs_numpy(self, complex_signal):
        """Numba CCorr should match numpy CCorr exactly (both float64)."""
        from hypyp.sync.ccorr import CCorr
        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = CCorr(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_ccorr_torch_vs_numpy(self, complex_signal):
        """Torch CCorr should match numpy CCorr within MPS tolerance."""
        from hypyp.sync.ccorr import CCorr
        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = CCorr(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        # Angle-free reformulation eliminates transcendental function chain,
        # bringing MPS precision in line with PLV.
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_ccorr_metal_vs_numpy(self, complex_signal):
        """Metal CCorr should match numpy CCorr within float32 tolerance.

        Uses Kahan summation with fastMath=OFF to preserve IEEE-754 compliance.
        """
        from hypyp.sync.ccorr import CCorr
        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = CCorr(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_ccorr_cuda_vs_numpy(self, complex_signal):
        """CUDA CCorr should match numpy CCorr exactly (both float64)."""
        from hypyp.sync.ccorr import CCorr
        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = CCorr(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestCoh:
    """Tests for Coherence with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5

    def test_coh_shape(self, complex_signal):
        """Coh output shape should match input dimensions."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result = Coh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_coh_value_range(self, complex_signal):
        """Coh values should be in [0, 1]."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result = Coh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_coh_numba_vs_numpy(self, complex_signal):
        """Numba Coh should match numpy Coh exactly (both float64)."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = Coh(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_coh_torch_vs_numpy(self, complex_signal):
        """Torch Coh should match numpy Coh within MPS tolerance."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = Coh(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    def test_coh_symmetry(self, complex_signal):
        """Coh matrix should be symmetric."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result = Coh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_coh_metal_vs_numpy(self, complex_signal):
        """Metal Coh should match numpy Coh within float32 tolerance."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = Coh(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_coh_cuda_vs_numpy(self, complex_signal):
        """CUDA Coh should match numpy Coh exactly (both float64)."""
        from hypyp.sync.coh import Coh
        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = Coh(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestImCoh:
    """Tests for Imaginary Coherence with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5

    def test_imcoh_shape(self, complex_signal):
        """ImCoh output shape should match input dimensions."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result = ImCoh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_imcoh_value_range(self, complex_signal):
        """ImCoh values should be in [0, 1]."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result = ImCoh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_imcoh_numba_vs_numpy(self, complex_signal):
        """Numba ImCoh should match numpy ImCoh exactly (both float64)."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = ImCoh(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_imcoh_torch_vs_numpy(self, complex_signal):
        """Torch ImCoh should match numpy ImCoh within MPS tolerance."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = ImCoh(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    def test_imcoh_symmetry(self, complex_signal):
        """ImCoh matrix should be symmetric."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result = ImCoh().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_imcoh_metal_vs_numpy(self, complex_signal):
        """Metal ImCoh should match numpy ImCoh within float32 tolerance."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = ImCoh(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_imcoh_cuda_vs_numpy(self, complex_signal):
        """CUDA ImCoh should match numpy ImCoh exactly (both float64)."""
        from hypyp.sync.imaginary_coh import ImCoh
        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = ImCoh(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestEnvCorr:
    """Tests for Envelope Correlation with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5

    def test_envcorr_shape(self, complex_signal):
        """EnvCorr output shape should match input dimensions."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_envcorr_value_range(self, complex_signal):
        """EnvCorr values should be in [-1, 1]."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1 - 1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_envcorr_numba_vs_numpy(self, complex_signal):
        """Numba EnvCorr should match numpy exactly (both float64)."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = EnvCorr(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_envcorr_torch_vs_numpy(self, complex_signal):
        """Torch EnvCorr should match numpy within MPS tolerance."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = EnvCorr(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    def test_envcorr_symmetry(self, complex_signal):
        """EnvCorr matrix should be symmetric."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )


class TestPLI:
    """Tests for Phase Lag Index with torch backend."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    # PLI uses sign() which is discontinuous at zero. MPS float32 can round
    # imaginary parts near zero differently than float64, flipping the sign
    # for a tiny fraction of values. A looser tolerance is needed.
    MPS_TOL = 1e-2

    def test_pli_shape(self, complex_signal):
        """PLI output shape should match input dimensions."""
        from hypyp.sync.pli import PLI
        metric = PLI()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_pli_value_range(self, complex_signal):
        """PLI values should be in [0, 1]."""
        from hypyp.sync.pli import PLI
        metric = PLI()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_pli_numba_vs_numpy(self, complex_signal):
        """Numba PLI should match numpy PLI exactly (both float64)."""
        from hypyp.sync.pli import PLI
        n_samp = complex_signal.shape[3]
        result_np = PLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = PLI(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_vs_numpy(self, complex_signal):
        """Torch PLI should match numpy PLI."""
        from hypyp.sync.pli import PLI
        n_samp = complex_signal.shape[3]

        result_np = PLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = PLI(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)

        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_shape(self, complex_signal):
        """Torch PLI output shape should match numpy."""
        from hypyp.sync.pli import PLI
        n_samp = complex_signal.shape[3]
        result = PLI(optimization='torch').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)


    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_large_channels(self):
        """PLI torch should handle 128ch/subject (256 total) without MPS INT_MAX crash."""
        from hypyp.sync.pli import PLI

        rng = np.random.default_rng(42)
        n_ch_per_subject = 128
        sig = rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256)) + \
              1j * rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256))
        n_samp = sig.shape[3]

        result_np = PLI().compute(sig, n_samp, self.TRANSPOSE_AXES)
        result_torch = PLI(optimization='torch').compute(sig, n_samp, self.TRANSPOSE_AXES)

        assert result_torch.shape == result_np.shape
        assert not np.any(np.isnan(result_torch))

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_pli_metal_vs_numpy(self, complex_signal):
        """Metal PLI should match numpy PLI within float32 tolerance."""
        from hypyp.sync.pli import PLI
        n_samp = complex_signal.shape[3]
        result_np = PLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = PLI(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        # Float32 precision — sign() near zero can flip
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_pli_metal_large_channels(self):
        """Metal PLI should handle 128ch/subject (256 total)."""
        from hypyp.sync.pli import PLI
        rng = np.random.default_rng(42)
        sig = rng.standard_normal((2, 1, 256, 256)) + 1j * rng.standard_normal((2, 1, 256, 256))
        n_samp = sig.shape[3]
        result = PLI(optimization='metal').compute(sig, n_samp, self.TRANSPOSE_AXES)
        assert result.shape == (2, 1, 256, 256)
        assert not np.any(np.isnan(result))
        assert np.allclose(np.diagonal(result[0, 0]), 0)  # diagonal = 0

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_pli_cuda_vs_numpy(self, complex_signal):
        """CUDA PLI should match numpy PLI exactly (both float64)."""
        from hypyp.sync.pli import PLI
        n_samp = complex_signal.shape[3]
        result_np = PLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = PLI(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        # Float64: should match to machine precision
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestWPLI:
    """Tests for Weighted Phase Lag Index with torch backend."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    # Same sign() precision issue as PLI, though wPLI weights mitigate it somewhat
    MPS_TOL = 1e-2

    def test_wpli_shape(self, complex_signal):
        """wPLI output shape should match input dimensions."""
        from hypyp.sync.wpli import WPLI
        metric = WPLI()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_wpli_value_range(self, complex_signal):
        """wPLI values should be in [0, 1]."""
        from hypyp.sync.wpli import WPLI
        metric = WPLI()
        n_samp = complex_signal.shape[3]
        result = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_wpli_numba_vs_numpy(self, complex_signal):
        """Numba wPLI should match numpy wPLI exactly (both float64)."""
        from hypyp.sync.wpli import WPLI
        n_samp = complex_signal.shape[3]
        result_np = WPLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = WPLI(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_vs_numpy(self, complex_signal):
        """Torch wPLI should match numpy wPLI."""
        from hypyp.sync.wpli import WPLI
        n_samp = complex_signal.shape[3]

        result_np = WPLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = WPLI(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)

        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_shape(self, complex_signal):
        """Torch wPLI output shape should match numpy."""
        from hypyp.sync.wpli import WPLI
        n_samp = complex_signal.shape[3]
        result = WPLI(optimization='torch').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)


    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_large_channels(self):
        """wPLI torch should handle 128ch/subject (256 total) without MPS INT_MAX crash."""
        from hypyp.sync.wpli import WPLI

        rng = np.random.default_rng(42)
        n_ch_per_subject = 128
        sig = rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256)) + \
              1j * rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256))
        n_samp = sig.shape[3]

        result_np = WPLI().compute(sig, n_samp, self.TRANSPOSE_AXES)
        result_torch = WPLI(optimization='torch').compute(sig, n_samp, self.TRANSPOSE_AXES)

        assert result_torch.shape == result_np.shape
        assert not np.any(np.isnan(result_torch))


class TestEnvCorr:
    """Tests for Envelope Correlation with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5

    def test_envcorr_shape(self, complex_signal):
        """EnvCorr output shape should match input dimensions."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_envcorr_value_range(self, complex_signal):
        """EnvCorr values should be in [-1, 1] (Pearson correlation)."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1 - 1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    def test_envcorr_symmetry(self, complex_signal):
        """EnvCorr matrix should be symmetric."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result = EnvCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_envcorr_numba_vs_numpy(self, complex_signal):
        """Numba EnvCorr should match numpy EnvCorr exactly (both float64)."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = EnvCorr(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_envcorr_torch_vs_numpy(self, complex_signal):
        """Torch EnvCorr should match numpy EnvCorr within MPS tolerance."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = EnvCorr(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_envcorr_metal_vs_numpy(self, complex_signal):
        """Metal EnvCorr should match numpy EnvCorr within float32 tolerance."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = EnvCorr(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_envcorr_cuda_vs_numpy(self, complex_signal):
        """CUDA EnvCorr should match numpy EnvCorr exactly (both float64)."""
        from hypyp.sync.envelope_corr import EnvCorr
        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = EnvCorr(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestPowCorr:
    """Tests for Power Correlation with all backends."""

    TRANSPOSE_AXES = (0, 1, 3, 2)
    MPS_TOL = 1e-5

    def test_powcorr_shape(self, complex_signal):
        """PowCorr output shape should match input dimensions."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result = PowCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    def test_powcorr_value_range(self, complex_signal):
        """PowCorr values should be in [-1, 1] (Pearson correlation)."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result = PowCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        assert np.all(result >= -1 - 1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    def test_powcorr_symmetry(self, complex_signal):
        """PowCorr matrix should be symmetric."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result = PowCorr().compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        for e in range(result.shape[0]):
            for f in range(result.shape[1]):
                np.testing.assert_allclose(
                    result[e, f], result[e, f].T, rtol=1e-10, atol=1e-12
                )

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_powcorr_numba_vs_numpy(self, complex_signal):
        """Numba PowCorr should match numpy PowCorr exactly (both float64)."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_numba = PowCorr(optimization='numba').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_powcorr_torch_vs_numpy(self, complex_signal):
        """Torch PowCorr should match numpy PowCorr within MPS tolerance."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        metric_torch = PowCorr(optimization='torch')
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == 'mps':
            np.testing.assert_allclose(result_torch, result_np,
                                       rtol=self.MPS_TOL, atol=self.MPS_TOL)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)


    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_powcorr_metal_vs_numpy(self, complex_signal):
        """Metal PowCorr should match numpy PowCorr within float32 tolerance."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = PowCorr(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_powcorr_cuda_vs_numpy(self, complex_signal):
        """CUDA PowCorr should match numpy PowCorr exactly (both float64)."""
        from hypyp.sync.pow_corr import PowCorr
        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = PowCorr(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_wpli_metal_vs_numpy(self, complex_signal):
        """Metal wPLI should match numpy wPLI within float32 tolerance."""
        from hypyp.sync.wpli import WPLI
        n_samp = complex_signal.shape[3]
        result_np = WPLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = WPLI(optimization='metal').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_wpli_cuda_vs_numpy(self, complex_signal):
        """CUDA wPLI should match numpy wPLI exactly (both float64)."""
        from hypyp.sync.wpli import WPLI
        n_samp = complex_signal.shape[3]
        result_np = WPLI(optimization=None).compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = WPLI(optimization='cuda_kernel').compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestAccorrKernels:
    """Tests for ACCorr Metal and CUDA kernels."""

    TRANSPOSE_AXES = (0, 1, 3, 2)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_accorr_metal_vs_numpy(self, complex_signal):
        """Metal ACCorr should match numpy within float32 tolerance."""
        from hypyp.sync.accorr import ACCorr
        n_samp = complex_signal.shape[3]
        result_np = ACCorr(optimization=None, show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_metal = ACCorr(optimization='metal', show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_accorr_cuda_vs_numpy(self, complex_signal):
        """CUDA ACCorr should match numpy exactly (both float64)."""
        from hypyp.sync.accorr import ACCorr
        n_samp = complex_signal.shape[3]
        result_np = ACCorr(optimization=None, show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_cuda = ACCorr(optimization='cuda_kernel', show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES)
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


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
        assert metric._backend in ('numpy', 'numba', 'torch', 'metal', 'cuda_kernel')


class TestAutoDispatch:
    """Benchmark-driven 'auto' dispatch per metric and platform."""

    def test_auto_all_metrics_resolve(self):
        """optimization='auto' resolves for every metric without error."""
        for metric_name in AUTO_PRIORITY:
            m = get_metric(metric_name, optimization='auto')
            assert m._backend in ('numpy', 'numba', 'torch', 'metal', 'cuda_kernel')

    @pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
    def test_auto_einsum_prefers_torch_on_mps(self):
        """Einsum metrics should prefer torch on Apple Silicon."""
        for metric_name in ['plv', 'ccorr', 'coh', 'imcoh', 'envcorr', 'powcorr']:
            m = get_metric(metric_name, optimization='auto')
            assert m._backend == 'torch' and m._device == 'mps', (
                f"{metric_name} auto: expected torch/mps, got {m._backend}/{m._device}"
            )

    @pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_auto_sign_prefers_metal_on_mps(self):
        """PLI/wPLI/ACCorr should prefer Metal on Apple Silicon."""
        for metric_name in ['pli', 'wpli', 'accorr']:
            m = get_metric(metric_name, optimization='auto')
            assert m._backend == 'metal', (
                f"{metric_name} auto: expected metal, got {m._backend}"
            )

    def test_auto_priority_override(self):
        """Custom priority overrides the AUTO_PRIORITY table."""
        m = get_metric('plv', optimization='auto', priority=['numba'])
        if NUMBA_AVAILABLE:
            assert m._backend == 'numba'
        else:
            assert m._backend == 'numpy'

    def test_auto_priority_skips_unavailable(self):
        """Priority list gracefully skips unavailable backends."""
        with patch('hypyp.sync.base.METAL_AVAILABLE', False), \
             patch('hypyp.sync.base.CUPY_AVAILABLE', False):
            m = get_metric('pli', optimization='auto', priority=['metal', 'cuda_kernel', 'numba'])
            if NUMBA_AVAILABLE:
                assert m._backend == 'numba'
            else:
                assert m._backend == 'numpy'

    def test_auto_fallback_cpu_only(self):
        """On CPU-only machines, auto warns and falls back to numba or numpy."""
        with patch('hypyp.sync.base.MPS_AVAILABLE', False), \
             patch('hypyp.sync.base.CUDA_AVAILABLE', False):
            with pytest.warns(UserWarning, match="No GPU available"):
                m = get_metric('plv', optimization='auto')
            if NUMBA_AVAILABLE:
                assert m._backend == 'numba'
            else:
                assert m._backend == 'numpy'

    def test_priority_parameter_propagated_via_get_metric(self):
        """get_metric passes priority through to the metric class."""
        m = get_metric('accorr', optimization='auto', priority=['numba'])
        assert m._priority == ['numba']
