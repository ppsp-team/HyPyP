"""
Tests for synchronization metrics, particularly adjusted circular correlation (accorr).

All optimized implementations are tested against the unoptimized reference
implementation to ensure numerical correctness.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from hypyp.analyses import compute_sync
from hypyp.sync import METRICS, get_metric
from hypyp.sync.accorr import ACCorr
from hypyp.sync.base import (
    BaseMetric,
    AUTO_PRIORITY,
    NUMBA_AVAILABLE,
    TORCH_AVAILABLE,
    MPS_AVAILABLE,
    METAL_AVAILABLE,
)
from hypyp.sync.kernels import CUPY_AVAILABLE
from tests.accorr_reference import accorr_reference


#: Written by hand on purpose: the tests below must not derive their cases
#: from supports(), the function they check.
EXPECTED_BACKENDS = {
    mode: {"numpy", "numba", "torch", "cuda_kernel"}
    | ({"metal"} if mode in {"pli", "wpli", "accorr"} else set())
    for mode in (
        "plv",
        "ccorr",
        "accorr",
        "coh",
        "imcoh",
        "pli",
        "wpli",
        "envcorr",
        "powcorr",
    )
}
ALL_BACKENDS = ("numpy", "numba", "torch", "metal", "cuda_kernel")

#: Also by hand: the routing test must not read the method names from
#: BaseMetric._BACKEND_METHODS, the table it checks.
EXPECTED_METHODS = {
    "numpy": "_compute_numpy",
    "numba": "_compute_numba",
    "torch": "_compute_torch",
    "metal": "_compute_metal",
    "cuda_kernel": "_compute_cuda",
}


def spy_on_kernel(module_name, function_name):
    """
    Patch a kernel function of ``hypyp.sync.kernels`` with a spy that still
    runs it, to prove the kernel itself was entered. The metrics import their
    kernel inside the method, so patching the module attribute is seen.
    """
    import importlib

    module = importlib.import_module(f"hypyp.sync.kernels.{module_name}")
    return patch.object(
        module, function_name, side_effect=getattr(module, function_name)
    )


def spy_on(cls, method_name):
    """
    Patch ``cls.<method_name>`` with a spy that still runs the real method.

    Asserting ``metric._backend == 'metal'`` only proves what the metric
    reports. The spy proves that ``compute`` really went through the method of
    that backend, which is what a broken dispatch would get wrong.
    """
    return patch.object(
        cls, method_name, autospec=True, side_effect=getattr(cls, method_name)
    )


class TestAccorrReference:
    """Basic properties of the reference implementation."""

    def test_reference_shape_no_average(self, complex_signal):
        result = accorr_reference(
            complex_signal, epochs_average=False, show_progress=False
        )
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_freq, n_epochs, n_ch, n_ch)

    def test_reference_shape_with_average(self, complex_signal):
        result = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_freq, n_ch, n_ch)

    def test_reference_value_range(self, complex_signal):
        result = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
        assert np.all(result >= -1 - 1e-10) and np.all(result <= 1 + 1e-10)
        assert not np.any(np.isnan(result))

    def test_reference_symmetry(self, complex_signal):
        result = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
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
        result_reference = accorr_reference(
            complex_signal, epochs_average=False, show_progress=False
        )
        result_numpy = self._compute_with_class(complex_signal, optimization=None)
        np.testing.assert_allclose(
            result_numpy, result_reference, rtol=1e-9, atol=1e-10
        )

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_vs_reference(self, complex_signal):
        result_reference = accorr_reference(
            complex_signal, epochs_average=False, show_progress=False
        )
        result_numba = self._compute_with_class(complex_signal, optimization="numba")
        np.testing.assert_allclose(
            result_numba, result_reference, rtol=1e-9, atol=1e-10
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_torch_vs_reference(self, complex_signal):
        result_reference = accorr_reference(
            complex_signal, epochs_average=False, show_progress=False
        )
        metric = ACCorr(optimization="torch", show_progress=False)
        n_samp = complex_signal.shape[3]
        con = metric.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        result_torch = con.swapaxes(0, 1)

        # MPS uses float32, so tolerance is lower
        if metric._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_reference, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
        else:
            np.testing.assert_allclose(
                result_torch, result_reference, rtol=1e-9, atol=1e-10
            )


class TestAccorrViaComputeSync:
    """Test accorr through the compute_sync API with optimization parameter."""

    MPS_TOL = 1e-5

    def test_compute_sync_default(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization=None should match reference."""
        result_reference = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
        result = compute_sync(
            complex_signal_raw, "accorr", optimization=None, epochs_average=True
        )
        np.testing.assert_allclose(result, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_compute_sync_numba(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization='numba' should match reference."""
        result_reference = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
        result = compute_sync(
            complex_signal_raw, "accorr", optimization="numba", epochs_average=True
        )
        np.testing.assert_allclose(result, result_reference, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_compute_sync_torch(self, complex_signal, complex_signal_raw):
        """compute_sync with optimization='torch' should match reference."""
        result_reference = accorr_reference(
            complex_signal, epochs_average=True, show_progress=False
        )
        result = compute_sync(
            complex_signal_raw, "accorr", optimization="torch", epochs_average=True
        )
        # MPS uses float32, so a looser tolerance is required
        if MPS_AVAILABLE:
            np.testing.assert_allclose(
                result, result_reference, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
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
        result_np = PLV(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = PLV(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_plv_torch_vs_numpy(self, complex_signal):
        """Torch PLV should match numpy PLV within MPS tolerance."""
        from hypyp.sync.plv import PLV

        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = PLV(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
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

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_plv_cuda_vs_numpy(self, complex_signal):
        """CUDA PLV should match numpy PLV exactly (both float64)."""
        from hypyp.sync.plv import PLV

        n_samp = complex_signal.shape[3]
        result_np = PLV(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = PLV(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        formula = "nilm,nimk->nilk"
        transpose_axes = self.TRANSPOSE_AXES
        result_scipy = np.abs(
            np.einsum(formula, angle_centered, angle_centered.transpose(transpose_axes))
            / np.sqrt(
                np.einsum(
                    "nil,nik->nilk",
                    np.sum(angle_centered**2, axis=3),
                    np.sum(angle_centered**2, axis=3),
                )
            )
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
        result_np = CCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = CCorr(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_ccorr_torch_vs_numpy(self, complex_signal):
        """Torch CCorr should match numpy CCorr within MPS tolerance."""
        from hypyp.sync.ccorr import CCorr

        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = CCorr(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        # Angle-free reformulation eliminates transcendental function chain,
        # bringing MPS precision in line with PLV.
        if metric_torch._device == "mps":
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_ccorr_cuda_vs_numpy(self, complex_signal):
        """CUDA CCorr should match numpy CCorr exactly (both float64)."""
        from hypyp.sync.ccorr import CCorr

        n_samp = complex_signal.shape[3]
        result_np = CCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = CCorr(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        result_np = Coh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = Coh(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_coh_torch_vs_numpy(self, complex_signal):
        """Torch Coh should match numpy Coh within MPS tolerance."""
        from hypyp.sync.coh import Coh

        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = Coh(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
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

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_coh_cuda_vs_numpy(self, complex_signal):
        """CUDA Coh should match numpy Coh exactly (both float64)."""
        from hypyp.sync.coh import Coh

        n_samp = complex_signal.shape[3]
        result_np = Coh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = Coh(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        result_np = ImCoh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = ImCoh(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_imcoh_torch_vs_numpy(self, complex_signal):
        """Torch ImCoh should match numpy ImCoh within MPS tolerance."""
        from hypyp.sync.imaginary_coh import ImCoh

        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = ImCoh(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
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

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_imcoh_cuda_vs_numpy(self, complex_signal):
        """CUDA ImCoh should match numpy ImCoh exactly (both float64)."""
        from hypyp.sync.imaginary_coh import ImCoh

        n_samp = complex_signal.shape[3]
        result_np = ImCoh(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = ImCoh(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        result_np = EnvCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = EnvCorr(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_envcorr_torch_vs_numpy(self, complex_signal):
        """Torch EnvCorr should match numpy within MPS tolerance."""
        from hypyp.sync.envelope_corr import EnvCorr

        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = EnvCorr(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
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
        result_np = PLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = PLI(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_vs_numpy(self, complex_signal):
        """Torch PLI should match numpy PLI."""
        from hypyp.sync.pli import PLI

        n_samp = complex_signal.shape[3]

        result_np = PLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = PLI(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)

        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_shape(self, complex_signal):
        """Torch PLI output shape should match numpy."""
        from hypyp.sync.pli import PLI

        n_samp = complex_signal.shape[3]
        result = PLI(optimization="torch").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_pli_torch_large_channels(self):
        """PLI torch should handle 128ch/subject (256 total) without MPS INT_MAX crash."""
        from hypyp.sync.pli import PLI

        rng = np.random.default_rng(42)
        n_ch_per_subject = 128
        sig = rng.standard_normal(
            (2, 1, 2 * n_ch_per_subject, 256)
        ) + 1j * rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256))
        n_samp = sig.shape[3]

        result_np = PLI().compute(sig, n_samp, self.TRANSPOSE_AXES)
        result_torch = PLI(optimization="torch").compute(
            sig, n_samp, self.TRANSPOSE_AXES
        )

        assert result_torch.shape == result_np.shape
        assert not np.any(np.isnan(result_torch))

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_pli_metal_vs_numpy(self, complex_signal):
        """Metal PLI should match numpy PLI within float32 tolerance."""
        from hypyp.sync.pli import PLI

        n_samp = complex_signal.shape[3]
        result_np = PLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_metal = PLI(optimization="metal")
        # A silent fallback to numpy would make this comparison numpy-vs-numpy
        # and therefore vacuous: check the backend, then that the Metal kernel
        # function itself is entered.
        assert metric_metal._backend == "metal"
        with spy_on_kernel("metal_phase", "pli_metal") as spy:
            result_metal = metric_metal.compute(
                complex_signal, n_samp, self.TRANSPOSE_AXES
            )
        assert spy.call_count == 1
        # Float32 precision — sign() near zero can flip
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_pli_metal_large_channels(self):
        """Metal PLI should handle 128ch/subject (256 total)."""
        from hypyp.sync.pli import PLI

        rng = np.random.default_rng(42)
        sig = rng.standard_normal((2, 1, 256, 256)) + 1j * rng.standard_normal(
            (2, 1, 256, 256)
        )
        n_samp = sig.shape[3]
        metric = PLI(optimization="metal")
        assert metric._backend == "metal"
        with spy_on_kernel("metal_phase", "pli_metal") as spy:
            result = metric.compute(sig, n_samp, self.TRANSPOSE_AXES)
        assert spy.call_count == 1
        assert result.shape == (2, 1, 256, 256)
        assert not np.any(np.isnan(result))
        assert np.allclose(np.diagonal(result[0, 0]), 0)  # diagonal = 0

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_pli_cuda_vs_numpy(self, complex_signal):
        """CUDA PLI should match numpy PLI exactly (both float64)."""
        from hypyp.sync.pli import PLI

        n_samp = complex_signal.shape[3]
        result_np = PLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = PLI(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        result_np = WPLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = WPLI(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_vs_numpy(self, complex_signal):
        """Torch wPLI should match numpy wPLI."""
        from hypyp.sync.wpli import WPLI

        n_samp = complex_signal.shape[3]

        result_np = WPLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = WPLI(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)

        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_shape(self, complex_signal):
        """Torch wPLI output shape should match numpy."""
        from hypyp.sync.wpli import WPLI

        n_samp = complex_signal.shape[3]
        result = WPLI(optimization="torch").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        n_epochs, n_freq, n_ch, _ = complex_signal.shape
        assert result.shape == (n_epochs, n_freq, n_ch, n_ch)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_wpli_torch_large_channels(self):
        """wPLI torch should handle 128ch/subject (256 total) without MPS INT_MAX crash."""
        from hypyp.sync.wpli import WPLI

        rng = np.random.default_rng(42)
        n_ch_per_subject = 128
        sig = rng.standard_normal(
            (2, 1, 2 * n_ch_per_subject, 256)
        ) + 1j * rng.standard_normal((2, 1, 2 * n_ch_per_subject, 256))
        n_samp = sig.shape[3]

        result_np = WPLI().compute(sig, n_samp, self.TRANSPOSE_AXES)
        result_torch = WPLI(optimization="torch").compute(
            sig, n_samp, self.TRANSPOSE_AXES
        )

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
        result_np = EnvCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = EnvCorr(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_envcorr_torch_vs_numpy(self, complex_signal):
        """Torch EnvCorr should match numpy EnvCorr within MPS tolerance."""
        from hypyp.sync.envelope_corr import EnvCorr

        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = EnvCorr(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_envcorr_cuda_vs_numpy(self, complex_signal):
        """CUDA EnvCorr should match numpy EnvCorr exactly (both float64)."""
        from hypyp.sync.envelope_corr import EnvCorr

        n_samp = complex_signal.shape[3]
        result_np = EnvCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = EnvCorr(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
        result_np = PowCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_numba = PowCorr(optimization="numba").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_numba, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
    def test_powcorr_torch_vs_numpy(self, complex_signal):
        """Torch PowCorr should match numpy PowCorr within MPS tolerance."""
        from hypyp.sync.pow_corr import PowCorr

        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_torch = PowCorr(optimization="torch")
        result_torch = metric_torch.compute(complex_signal, n_samp, self.TRANSPOSE_AXES)
        if metric_torch._device == "mps":
            np.testing.assert_allclose(
                result_torch, result_np, rtol=self.MPS_TOL, atol=self.MPS_TOL
            )
        else:
            np.testing.assert_allclose(result_torch, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_powcorr_cuda_vs_numpy(self, complex_signal):
        """CUDA PowCorr should match numpy PowCorr exactly (both float64)."""
        from hypyp.sync.pow_corr import PowCorr

        n_samp = complex_signal.shape[3]
        result_np = PowCorr(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = PowCorr(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_wpli_metal_vs_numpy(self, complex_signal):
        """Metal wPLI should match numpy wPLI within float32 tolerance."""
        from hypyp.sync.wpli import WPLI

        n_samp = complex_signal.shape[3]
        result_np = WPLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_metal = WPLI(optimization="metal")
        assert metric_metal._backend == "metal"
        with spy_on_kernel("metal_phase", "wpli_metal") as spy:
            result_metal = metric_metal.compute(
                complex_signal, n_samp, self.TRANSPOSE_AXES
            )
        assert spy.call_count == 1
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_wpli_cuda_vs_numpy(self, complex_signal):
        """CUDA wPLI should match numpy wPLI exactly (both float64)."""
        from hypyp.sync.wpli import WPLI

        n_samp = complex_signal.shape[3]
        result_np = WPLI(optimization=None).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = WPLI(optimization="cuda_kernel").compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
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
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        metric_metal = ACCorr(optimization="metal", show_progress=False)
        assert metric_metal._backend == "metal"
        with spy_on_kernel("metal_accorr", "accorr_metal") as spy:
            result_metal = metric_metal.compute(
                complex_signal, n_samp, self.TRANSPOSE_AXES
            )
        assert spy.call_count == 1
        np.testing.assert_allclose(result_metal, result_np, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_accorr_cuda_vs_numpy(self, complex_signal):
        """CUDA ACCorr should match numpy exactly (both float64)."""
        from hypyp.sync.accorr import ACCorr

        n_samp = complex_signal.shape[3]
        result_np = ACCorr(optimization=None, show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        result_cuda = ACCorr(optimization="cuda_kernel", show_progress=False).compute(
            complex_signal, n_samp, self.TRANSPOSE_AXES
        )
        np.testing.assert_allclose(result_cuda, result_np, rtol=1e-9, atol=1e-10)


class TestAccorrErrorHandling:
    """Error handling and fallback behavior."""

    def test_invalid_optimization(self):
        with pytest.raises(ValueError, match="Unknown optimization"):
            ACCorr(optimization="invalid_option")

    def test_numba_fallback_warning(self):
        """When numba is unavailable, optimization='numba' warns and falls back to numpy."""
        with patch("hypyp.sync.base.NUMBA_AVAILABLE", False):
            with pytest.warns(UserWarning, match="numba not installed"):
                metric = ACCorr(optimization="numba")
            assert metric._backend == "numpy"

    def test_torch_fallback_warning(self):
        """When torch is unavailable, optimization='torch' warns and falls back to numpy."""
        with patch("hypyp.sync.base.TORCH_AVAILABLE", False):
            with pytest.warns(UserWarning, match="torch not installed"):
                metric = ACCorr(optimization="torch")
            assert metric._backend == "numpy"

    def test_auto_resolves(self):
        """optimization='auto' should resolve without error."""
        metric = ACCorr(optimization="auto")
        assert metric._backend in ("numpy", "numba", "torch", "metal", "cuda_kernel")


class TestAutoDispatch:
    """Benchmark-driven 'auto' dispatch per metric and platform."""

    def test_auto_all_metrics_resolve(self):
        """optimization='auto' resolves for every metric without error."""
        for metric_name in AUTO_PRIORITY:
            m = get_metric(metric_name, optimization="auto")
            assert m._backend in ("numpy", "numba", "torch", "metal", "cuda_kernel")

    @pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
    def test_auto_einsum_prefers_torch_on_mps(self):
        """Einsum metrics should prefer torch on Apple Silicon."""
        for metric_name in ["plv", "ccorr", "coh", "imcoh", "envcorr", "powcorr"]:
            m = get_metric(metric_name, optimization="auto")
            assert m._backend == "torch" and m._device == "mps", (
                f"{metric_name} auto: expected torch/mps, got {m._backend}/{m._device}"
            )

    @pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_auto_sign_prefers_metal_on_mps(self):
        """PLI/wPLI/ACCorr should prefer Metal on Apple Silicon."""
        for metric_name in ["pli", "wpli", "accorr"]:
            m = get_metric(metric_name, optimization="auto")
            assert m._backend == "metal", (
                f"{metric_name} auto: expected metal, got {m._backend}"
            )

    def test_auto_priority_override(self):
        """Custom priority overrides the AUTO_PRIORITY table."""
        m = get_metric("plv", optimization="auto", priority=["numba"])
        if NUMBA_AVAILABLE:
            assert m._backend == "numba"
        else:
            assert m._backend == "numpy"

    def test_auto_priority_skips_unavailable(self):
        """Priority list gracefully skips unavailable backends."""
        with (
            patch("hypyp.sync.base.METAL_AVAILABLE", False),
            patch("hypyp.sync.base.CUPY_AVAILABLE", False),
        ):
            m = get_metric(
                "pli", optimization="auto", priority=["metal", "cuda_kernel", "numba"]
            )
            if NUMBA_AVAILABLE:
                assert m._backend == "numba"
            else:
                assert m._backend == "numpy"

    def test_auto_fallback_cpu_only(self):
        """On CPU-only machines, auto warns and falls back to numba or numpy."""
        with (
            patch("hypyp.sync.base.MPS_AVAILABLE", False),
            patch("hypyp.sync.base.CUDA_AVAILABLE", False),
        ):
            with pytest.warns(UserWarning, match="No GPU available"):
                m = get_metric("plv", optimization="auto")
            if NUMBA_AVAILABLE:
                assert m._backend == "numba"
            else:
                assert m._backend == "numpy"

    def test_priority_parameter_propagated_via_get_metric(self):
        """get_metric passes priority through to the metric class."""
        m = get_metric("accorr", optimization="auto", priority=["numba"])
        assert m._priority == ["numba"]


class TestBackendCapability:
    """
    A requested backend must either run, or degrade with a warning.

    Only PLI, wPLI and ACCorr have Metal kernels — torch/MPS is the intended
    GPU path for the six einsum metrics (see hypyp/sync/base.py AUTO_PRIORITY
    rationale and the support matrix in hypyp/sync/README.md). Requesting
    'metal' for a metric that has no Metal kernel must therefore be reported,
    never silently answered with numpy.

    These tests patch the availability flags instead of gating on hardware, so
    the capability contract is verified on any machine including CI.
    """

    METAL_CAPABLE = {"pli", "wpli", "accorr"}

    def test_capability_matrix(self):
        """supports() must answer exactly the hand-written support matrix."""
        assert set(METRICS) == set(EXPECTED_BACKENDS)
        for mode, cls in METRICS.items():
            actual = {b for b in ALL_BACKENDS if cls.supports(b)}
            assert actual == EXPECTED_BACKENDS[mode], mode
            assert cls.supports("not_a_backend") is False

    def test_supports_reflects_the_implemented_methods(self):
        """supports() must be derived from the code, not a hand-kept list."""
        for mode, cls in METRICS.items():
            assert cls.supports("metal") == hasattr(cls, "_compute_metal")
            assert cls.supports("numpy") is True
            assert cls.supports("numba") == hasattr(cls, "_compute_numba")

    def test_metal_capability_matches_documented_support_matrix(self):
        """Exactly PLI/wPLI/ACCorr expose a Metal kernel."""
        actual = {mode for mode, cls in METRICS.items() if cls.supports("metal")}
        assert actual == self.METAL_CAPABLE

    @pytest.mark.parametrize("mode", sorted(METRICS))
    def test_metal_request_never_silently_degrades(self, mode, complex_signal):
        """
        optimization='metal' either resolves to metal, or warns and uses numpy.

        Regression test: before the capability check, _resolve_optimization
        granted ('metal', 'mps') to every metric, and compute() then fell
        through its if/elif chain into _compute_numpy — so six metrics returned
        a numpy result while reporting _backend == 'metal', with no warning.
        """
        with patch("hypyp.sync.base.METAL_AVAILABLE", True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metric = get_metric(mode, optimization="metal")

        if mode in self.METAL_CAPABLE:
            assert metric._backend == "metal"
        else:
            assert metric._backend == "numpy", (
                f"{mode}: asked for metal, resolved to {metric._backend!r} "
                f"but has no Metal kernel"
            )
            messages = [
                str(w.message) for w in caught if issubclass(w.category, UserWarning)
            ]
            assert any("Metal" in m for m in messages), (
                f"{mode}: degraded to numpy without warning (messages: {messages})"
            )
            # The fallback must also compute: same result as a plain numpy
            # metric, through the numpy method.
            n_samp = complex_signal.shape[3]
            axes = (0, 1, 3, 2)
            expected = get_metric(mode).compute(complex_signal, n_samp, axes)
            with spy_on(type(metric), "_compute_numpy") as spy:
                result = metric.compute(complex_signal, n_samp, axes)
            assert spy.call_count == 1
            np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "mode, backend",
        [
            (mode, backend)
            for mode in sorted(EXPECTED_BACKENDS)
            for backend in sorted(EXPECTED_BACKENDS[mode])
        ],
    )
    def test_compute_routes_to_the_method_of_the_backend(self, mode, backend):
        """
        compute() must call the ``_compute_*`` method of the resolved backend,
        with its arguments, and return its result, for every backend each
        metric implements. No hardware is needed: the method is replaced by a
        stub.
        """
        cls = METRICS[mode]
        method_name = EXPECTED_METHODS[backend]
        metric = cls()
        metric._backend = backend
        sentinel = object()
        with patch.object(cls, method_name, autospec=True) as stub:
            stub.return_value = sentinel
            result = metric.compute("signal", 7, (0, 1, 3, 2))
        assert result is sentinel
        stub.assert_called_once_with(metric, "signal", 7, (0, 1, 3, 2))

    @pytest.mark.parametrize("mode", sorted(set(METRICS) - {"pli", "wpli", "accorr"}))
    @pytest.mark.parametrize("priority", [["metal", "torch"], ["metal"]])
    def test_auto_priority_unsupported_backend_stays_on_numpy(self, mode, priority):
        """
        A priority list that reaches an available backend the metric cannot
        run keeps computing in numpy, as before, but now says so.

        The 0.6 series changes no computed value: moving on to the next
        backend of the list (torch, or the numba fallback) would.
        """
        with (
            patch("hypyp.sync.base.METAL_AVAILABLE", True),
            patch("hypyp.sync.base.TORCH_AVAILABLE", True),
            patch("hypyp.sync.base.MPS_AVAILABLE", True),
            patch("hypyp.sync.base.NUMBA_AVAILABLE", True),
        ):
            with pytest.warns(UserWarning, match="no Metal implementation"):
                metric = get_metric(mode, optimization="auto", priority=priority)
        assert (metric._backend, metric._device) == ("numpy", "cpu"), (
            f"{mode}: priority={priority} resolved to {metric._backend!r}"
        )

    def test_unknown_backend_fails_closed(self, complex_signal):
        """
        An unrecognised _backend must raise, not quietly compute in numpy.

        This is the fail-closed guarantee: the original if/elif chains ended in
        a bare `return self._compute_numpy(...)`, so any unhandled backend value
        became indistinguishable from the default.
        """
        from hypyp.sync.plv import PLV

        n_samp = complex_signal.shape[3]
        metric = PLV()
        metric._backend = "not_a_backend"
        with pytest.raises(ValueError) as excinfo:
            metric.compute(complex_signal, n_samp, (0, 1, 3, 2))
        # The error must be usable: it names the metric, the offending
        # backend and the backends that do exist for this metric.
        message = str(excinfo.value)
        assert "'plv'" in message
        assert "'not_a_backend'" in message
        assert "numpy" in message

    def test_unimplemented_backend_computes_in_numpy_with_a_warning(
        self, complex_signal
    ):
        """
        A known backend the metric does not implement computes in numpy, as
        before, but says so: the 0.6 series changes no computed value.
        """
        from hypyp.sync.plv import PLV

        n_samp = complex_signal.shape[3]
        axes = (0, 1, 3, 2)
        metric = PLV()
        metric._backend = "metal"
        with pytest.warns(UserWarning, match="'plv' has no Metal implementation"):
            result = metric.compute(complex_signal, n_samp, axes)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(
            result, PLV()._compute_numpy(complex_signal, n_samp, axes)
        )

    def test_priority_fallback_warning_names_the_skipped_backend(self):
        """
        When the only backend of a priority list has no implementation and
        cannot run on the machine either, the fallback warning must give the
        first reason.

        Before, priority=['metal'] on an einsum metric warned "No GPU backend
        available" on a CUDA machine, where a GPU backend was available,
        without mentioning that the metric has no Metal kernel.
        """
        with (
            patch("hypyp.sync.base.METAL_AVAILABLE", False),
            patch("hypyp.sync.base.TORCH_AVAILABLE", True),
            patch("hypyp.sync.base.MPS_AVAILABLE", False),
            patch("hypyp.sync.base.CUDA_AVAILABLE", True),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metric = get_metric("plv", optimization="auto", priority=["metal"])

        assert metric._backend in ("numba", "numpy")
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("'plv' has no Metal implementation" in m for m in messages), (
            f"fallback warning does not explain the skip (messages: {messages})"
        )
        assert not any("No GPU backend available" in m for m in messages)
        # The message must stay true when the CPU fallback is numba: it speaks
        # of GPU backends only.
        assert any("no GPU backend of the priority list" in m for m in messages)

    @staticmethod
    def _legacy_metric(with_helper=False):
        """A third-party metric written against the pre-0.6.2 contract: it
        overrides ``compute``, branches on ``self._backend`` itself and has no
        ``_compute_*`` method."""
        from hypyp.sync.base import BaseMetric

        class LegacyMetric(BaseMetric):
            name = "legacy"

            def compute(self, complex_signal, n_samp, transpose_axes):
                base_result = super().compute(complex_signal, n_samp, transpose_axes)
                return self._backend, base_result

        class LegacyWithHelper(LegacyMetric):
            # Same contract, but the author happened to name a helper like the
            # methods of the current contract. Still its own dispatch.
            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                return "helper"

        return LegacyWithHelper if with_helper else LegacyMetric

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (dict(optimization=None), "numpy"),
            (dict(optimization="numba"), "numba"),
            (dict(optimization="torch"), "torch"),
            (dict(optimization="metal"), "metal"),
            (dict(optimization="auto", priority=["metal", "torch"]), "metal"),
            (dict(optimization="auto", priority=["torch"]), "torch"),
        ],
    )
    @pytest.mark.parametrize("with_helper", [False, True])
    def test_legacy_subclass_keeps_its_own_dispatch(
        self, kwargs, expected, with_helper
    ):
        """
        A subclass of the earlier contract is granted the backend it asks for,
        as before the capability check, and without a "no implementation"
        warning: the base class cannot see inside its ``compute``.
        """
        legacy_cls = self._legacy_metric(with_helper)
        with (
            patch("hypyp.sync.base.METAL_AVAILABLE", True),
            patch("hypyp.sync.base.TORCH_AVAILABLE", True),
            patch("hypyp.sync.base.MPS_AVAILABLE", True),
            patch("hypyp.sync.base.NUMBA_AVAILABLE", True),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metric = legacy_cls(**kwargs)
        assert metric._backend == expected
        assert not any("implementation" in str(w.message) for w in caught)
        # super().compute() used to be an abstract method with an empty body:
        # it returned None and must still do so, not raise.
        assert metric.compute(None, 0, None) == (expected, None)

    @pytest.mark.parametrize("gpu", [True, False])
    def test_numpy_only_metric_never_gets_numba(self, gpu, complex_signal):
        """
        A metric of the current contract that implements numpy alone must
        resolve to numpy under 'auto', even when numba is installed: the CPU
        fallback has to respect capability like every other path.
        """
        from hypyp.sync.base import BaseMetric

        class NumpyOnly(BaseMetric):
            name = "numpy_only"

            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                return np.ones(1)

        with (
            patch("hypyp.sync.base.NUMBA_AVAILABLE", True),
            patch("hypyp.sync.base.TORCH_AVAILABLE", gpu),
            patch("hypyp.sync.base.MPS_AVAILABLE", gpu),
            patch("hypyp.sync.base.CUDA_AVAILABLE", False),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = NumpyOnly(optimization="auto")
        assert metric._backend == "numpy"
        n_samp = complex_signal.shape[3]
        assert metric.compute(complex_signal, n_samp, (0, 1, 3, 2)).shape == (1,)

    def test_subclass_of_builtin_with_its_own_backend(self, complex_signal):
        """
        A third-party subclass of a built-in metric that handles a backend in
        its own ``compute`` and delegates the rest to its parent keeps working:
        the backend is granted without warning, its own branch runs, and the
        delegation still computes.
        """
        from hypyp.sync.plv import PLV

        class CustomPLV(PLV):
            def compute(self, complex_signal, n_samp, transpose_axes):
                if self._backend == "metal":
                    return "custom Metal"
                return super().compute(complex_signal, n_samp, transpose_axes)

        with patch("hypyp.sync.base.METAL_AVAILABLE", True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metric = CustomPLV(optimization="metal")
        assert metric._backend == "metal"
        assert not caught
        assert metric.compute(None, 0, None) == "custom Metal"

        n_samp = complex_signal.shape[3]
        axes = (0, 1, 3, 2)
        delegated = CustomPLV().compute(complex_signal, n_samp, axes)
        # Compared with the numpy method itself, and checked to be an array:
        # two None results would otherwise compare equal.
        assert isinstance(delegated, np.ndarray)
        np.testing.assert_array_equal(
            delegated, PLV()._compute_numpy(complex_signal, n_samp, axes)
        )
        # The built-in parent itself stays capability-checked.
        assert PLV.supports("metal") is False

    def test_dispatch_flag_set_by_a_mixin(self):
        """
        The class that sets ``_dispatch_via_table`` need not define
        ``compute``: a mixin can carry the flag. The metric is then
        capability-checked like its built-in parent, and does not crash.
        """
        from hypyp.sync.plv import PLV

        class DispatchPolicy:
            _dispatch_via_table = True

        class MixedPLV(DispatchPolicy, PLV):
            pass

        assert MixedPLV.supports("numpy") is True
        assert MixedPLV.supports("metal") is False
        with patch("hypyp.sync.base.METAL_AVAILABLE", True):
            with pytest.warns(UserWarning, match="no Metal implementation"):
                metric = MixedPLV(optimization="metal")
        assert metric._backend == "numpy"

    def test_rebinding_the_parent_compute_keeps_the_capability_check(self):
        """
        ``compute = PLV.compute`` in a subclass is the same function as the
        one the flag of PLV vouches for, not a dispatch of its own: the
        subclass is still capability-checked.
        """
        from hypyp.sync.plv import PLV

        class Alias(PLV):
            compute = PLV.compute

        assert Alias.supports("numpy") is True
        assert Alias.supports("metal") is False
        with patch("hypyp.sync.base.METAL_AVAILABLE", True):
            with pytest.warns(UserWarning, match="no Metal implementation"):
                metric = Alias(optimization="metal")
        assert metric._backend == "numpy"

    def test_auto_keeps_numba_handled_inside_compute(self, complex_signal):
        """
        A descendant of a built-in metric that hides ``_compute_numba`` and
        handles numba inside its own ``compute`` still gets numba from the
        CPU fallback of ``'auto'``, as before the capability check, and its
        own branch runs.
        """
        from hypyp.sync.plv import PLV

        class OwnNumba(PLV):
            _compute_numba = None

            def compute(self, complex_signal, n_samp, transpose_axes):
                if self._backend == "numba":
                    return "own numba"
                return super().compute(complex_signal, n_samp, transpose_axes)

        n_samp = complex_signal.shape[3]
        axes = (0, 1, 3, 2)
        with (
            patch("hypyp.sync.base.NUMBA_AVAILABLE", True),
            patch("hypyp.sync.base.TORCH_AVAILABLE", False),
            patch("hypyp.sync.base.MPS_AVAILABLE", False),
            patch("hypyp.sync.base.CUDA_AVAILABLE", False),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                automatic = OwnNumba(optimization="auto")
            requested = OwnNumba(optimization="numba")
        assert automatic._backend == "numba"
        assert automatic.compute(complex_signal, n_samp, axes) == "own numba"
        assert requested._backend == "numba"
        assert requested.compute(complex_signal, n_samp, axes) == "own numba"

    @pytest.mark.parametrize("delegates", [False, True])
    def test_backend_method_added_by_a_subclass_needs_the_flag(
        self, complex_signal, delegates
    ):
        """
        Before the table dispatch, a ``_compute_metal`` added to a subclass of
        PLV was never called: the request computed in numpy. That result is
        kept, with a warning, whether the subclass inherits ``compute`` or
        overrides it only to delegate. The added method is used once the
        subclass sets ``_dispatch_via_table`` itself.
        """
        from hypyp.sync.plv import PLV

        class AddsMetal(PLV):
            def _compute_metal(self, complex_signal, n_samp, transpose_axes):
                return "added Metal"

            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                # An overridden method of the parent is still honoured.
                return 2 * super()._compute_numpy(
                    complex_signal, n_samp, transpose_axes
                )

        if delegates:

            class AddsMetal(AddsMetal):  # noqa: F811
                def compute(self, complex_signal, n_samp, transpose_axes):
                    return super().compute(complex_signal, n_samp, transpose_axes)

        class Migrated(AddsMetal):
            _dispatch_via_table = True

        n_samp = complex_signal.shape[3]
        axes = (0, 1, 3, 2)
        expected = 2 * PLV()._compute_numpy(complex_signal, n_samp, axes)
        with patch("hypyp.sync.base.METAL_AVAILABLE", True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metric = AddsMetal(optimization="metal")
                result = metric.compute(complex_signal, n_samp, axes)
            migrated = Migrated(optimization="metal")
        assert any("no Metal implementation" in str(w.message) for w in caught)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        assert Migrated.supports("metal") is True
        assert migrated._backend == "metal"
        assert migrated.compute(complex_signal, n_samp, axes) == "added Metal"

    def test_dispatch_flag_set_by_a_mixin_listed_after_the_base(self):
        """
        A mixin that carries the flag can come after ``BaseMetric`` in the
        bases, where no class defines ``compute`` any more. A metric with its
        own ``compute`` is then trusted with the backend it requests, and one
        without is capability-checked; neither crashes.
        """

        class DispatchPolicy:
            _dispatch_via_table = True

        class OwnDispatch(BaseMetric, DispatchPolicy):
            name = "own_dispatch"

            def compute(self, complex_signal, n_samp, transpose_axes):
                return self._backend

        class TableOnly(BaseMetric, DispatchPolicy):
            name = "table_only"

            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                return "numpy"

        with patch("hypyp.sync.base.NUMBA_AVAILABLE", True):
            assert OwnDispatch.supports("numba") is True
            metric = OwnDispatch(optimization="numba")
            assert metric.compute(None, 0, None) == "numba"
            assert TableOnly.supports("numpy") is True
            assert TableOnly.supports("numba") is False

    def test_delegating_descendant_of_numpy_only_metric_still_computes(self):
        """
        A descendant that overrides ``compute`` only to delegate is trusted
        with numba by the automatic CPU fallback, like any class with its own
        ``compute``. No numba method exists, so the dispatch computes in numpy
        and warns instead of failing.
        """
        from hypyp.sync.base import BaseMetric

        class NumpyOnly(BaseMetric):
            name = "numpy_only"
            _dispatch_via_table = True

            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                return "numpy result"

        class Delegating(NumpyOnly):
            def compute(self, complex_signal, n_samp, transpose_axes):
                return super().compute(complex_signal, n_samp, transpose_axes)

        with (
            patch("hypyp.sync.base.NUMBA_AVAILABLE", True),
            patch("hypyp.sync.base.TORCH_AVAILABLE", False),
            patch("hypyp.sync.base.MPS_AVAILABLE", False),
            patch("hypyp.sync.base.CUDA_AVAILABLE", False),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric = Delegating(optimization="auto")
        assert metric._backend == "numba"
        with pytest.warns(UserWarning, match="no numba implementation"):
            assert metric.compute(None, 0, None) == "numpy result"

    def test_classmethod_implementation_is_recognised(self):
        """A ``_compute_*`` method declared as a classmethod is an
        implementation like any other."""
        from hypyp.sync.base import BaseMetric

        class ClassLevel(BaseMetric):
            name = "class_level"
            _dispatch_via_table = True

            @classmethod
            def _compute_numpy(cls, complex_signal, n_samp, transpose_axes):
                return "numpy result"

        assert ClassLevel.supports("numpy") is True
        assert ClassLevel().compute(None, 0, None) == "numpy result"

    def test_supports_ignores_placeholders(self):
        """supports() must not count a non-callable attribute, nor the default
        ``_compute_numpy`` of the base class, as an implementation."""
        from hypyp.sync.base import BaseMetric

        class Placeholder(BaseMetric):
            name = "placeholder"
            _compute_torch = None

            def _compute_numpy(self, complex_signal, n_samp, transpose_axes):
                return np.ones(1)

        class EmptyMetric(BaseMetric):
            name = "empty"

        assert Placeholder.supports("numpy") is True
        assert Placeholder.supports("torch") is False
        assert EmptyMetric.supports("numpy") is False
        assert BaseMetric.supports("numpy") is False

    def test_torch_only_metric_reports_its_backends(self):
        """A metric of the current contract without numpy runs the backend it
        has and names what is missing otherwise."""
        from hypyp.sync.base import BaseMetric

        class TorchOnly(BaseMetric):
            name = "torch_only"

            def _compute_torch(self, complex_signal, n_samp, transpose_axes):
                return "torch result"

        metric = TorchOnly()
        with pytest.raises(NotImplementedError, match="_compute_numpy"):
            metric.compute(None, 0, None)
        metric._backend = "torch"
        assert metric.compute(None, 0, None) == "torch result"
        metric._backend = "metal"
        with pytest.raises(
            ValueError, match=r"implemented for this metric: \['torch'\]"
        ):
            metric.compute(None, 0, None)

    def test_missing_numpy_implementation_is_reported(self, complex_signal):
        """A metric with neither ``compute`` nor ``_compute_numpy`` says so."""
        from hypyp.sync.base import BaseMetric

        class EmptyMetric(BaseMetric):
            name = "empty"

        n_samp = complex_signal.shape[3]
        with pytest.raises(NotImplementedError, match="_compute_numpy"):
            EmptyMetric().compute(complex_signal, n_samp, (0, 1, 3, 2))


# ---------------------------------------------------------------------------
# Robustness of the optional-dependency probes and of the metric registry
# ---------------------------------------------------------------------------


def _import_hypyp_with_fake_package(tmp_path, module, flag, init_source):
    """Import hypyp in a fresh interpreter where `module` resolves to a fake
    package whose `__init__` is `init_source`. Returns the availability flag
    and the messages of the warnings that name the module."""
    import json
    import os
    import subprocess
    import sys

    package = tmp_path / module
    package.mkdir()
    (package / "__init__.py").write_text(init_source)
    code = (
        "import json, warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import hypyp.analyses\n"
        "    from hypyp.sync import base, kernels\n"
        f"messages = [str(w.message) for w in caught if '{module}' in str(w.message)]\n"
        f"print('RESULT', json.dumps([bool({flag}), messages]))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


OPTIONAL_PACKAGES = [
    ("torch", "base.TORCH_AVAILABLE"),
    ("numba", "base.NUMBA_AVAILABLE"),
    ("cupy", "kernels.CUPY_AVAILABLE"),
    ("Metal", "kernels.METAL_AVAILABLE"),
]


@pytest.mark.parametrize("module, flag", OPTIONAL_PACKAGES)
@pytest.mark.parametrize(
    "error",
    [
        'OSError("simulated broken install: missing shared library")',
        'ImportError("simulated broken install: cannot load a symbol")',
        'ModuleNotFoundError("simulated broken install", name="a_dependency")',
        'RuntimeError("simulated broken install: version mismatch")',
    ],
)
def test_broken_optional_dependency_does_not_break_import(
    tmp_path, module, flag, error
):
    """An optional package that is installed but fails to load must leave
    hypyp importable, with the backend reported as unavailable and a warning
    that names the package and gives the original error."""
    available, messages = _import_hypyp_with_fake_package(
        tmp_path, module, flag, f"raise {error}\n"
    )
    assert available is False
    assert len(messages) == 1
    assert "simulated broken install" in messages[0]


@pytest.mark.parametrize("module, flag", OPTIONAL_PACKAGES)
def test_absent_optional_dependency_stays_silent(tmp_path, module, flag):
    """A package that is simply not installed disables its backend without
    any warning, as before."""
    available, messages = _import_hypyp_with_fake_package(
        tmp_path,
        module,
        flag,
        f'raise ModuleNotFoundError("No module named {module!r}", name="{module}")\n',
    )
    assert available is False
    assert messages == []


@pytest.mark.parametrize(
    "alias, mode",
    [("envelope_corr", "envcorr"), ("pow_corr", "powcorr"), ("imaginary_coh", "imcoh")],
)
def test_get_metric_accepts_the_aliases_of_compute_sync(alias, mode):
    assert type(get_metric(alias)) is METRICS[mode]
    assert type(get_metric(alias.upper())) is METRICS[mode]


def test_get_metric_registered_name_wins_over_alias():
    """A metric that a user registered under the name of an alias is still
    the one returned."""

    class Custom(METRICS["plv"]):
        pass

    with patch.dict(METRICS, {"pow_corr": Custom}):
        assert type(get_metric("pow_corr")) is Custom
    assert type(get_metric("pow_corr")) is METRICS["powcorr"]


def test_get_metric_unknown_mode_still_raises():
    with pytest.raises(ValueError, match="Unknown metric mode 'nope'"):
        get_metric("nope")


@pytest.mark.parametrize("backend", ["numba", "torch"])
def test_missing_backend_hint_names_the_pip_extra(backend):
    flag = f"hypyp.sync.base.{backend.upper()}_AVAILABLE"
    with patch(flag, False):
        with pytest.warns(UserWarning) as record:
            metric = get_metric("plv", optimization=backend)
    assert metric._backend == "numpy"
    messages = " ".join(str(w.message) for w in record)
    assert f'pip install "hypyp[{backend}]"' in messages
    assert "poetry" not in messages
