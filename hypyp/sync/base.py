#!/usr/bin/env python
# coding=utf-8

"""
Base classes, einsum helpers, optional-dependency probing, and the
AUTO_PRIORITY benchmark dispatch table for the connectivity metrics.

This module is shared by every concrete metric in ``hypyp.sync``. It
exposes:

- ``BaseMetric`` — base class. Concrete metrics implement the
  ``_compute_*`` methods (``_compute_numpy`` at least) and rely on the
  shared backend-resolution, warning-fallback and dispatch logic.
- ``multiply_conjugate``, ``multiply_conjugate_time``,
  ``multiply_product`` — vectorised einsum kernels (numpy).
- ``multiply_conjugate_torch``, ``multiply_conjugate_time_torch`` —
  torch equivalents (only resolvable if torch is installed).
- ``AUTO_PRIORITY`` — benchmark-driven backend lookup table per
  ``{metric_name: {platform: [gpu_backend, fallback]}}``.
- Capability flags ``TORCH_AVAILABLE``, ``MPS_AVAILABLE``,
  ``CUDA_AVAILABLE``, ``NUMBA_AVAILABLE``, ``METAL_AVAILABLE``,
  ``CUPY_AVAILABLE`` — probed at import time so concrete metric classes
  don't have to retry.

Design note
-----------
``AUTO_PRIORITY`` is intentionally kept as a Python dict (not
externalised to a YAML file) for three reasons:

1. The values are not user-tunable — they are derived from benchmarks
   on Mac M4 Max (131 rows) and Narval A100 (111 rows) and need
   re-derivation if the kernels change. Putting them in YAML would
   wrongly suggest they are configuration knobs.
2. The per-call ``priority=`` kwarg on ``get_metric`` already provides
   the override path users actually need.
3. The table is short (9 entries) and sits next to its rationale
   comment block; a YAML file would split the explanation from the
   data.

If a future benchmark sweep changes the optimal backend, the change
should be a code edit (with a tests/benchmarks update) — not a config
change.
"""

import warnings
from abc import ABC
from typing import Optional

import numpy as np


# Check optional dependency availability
try:
    import torch

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Custom kernel backends
from .kernels import METAL_AVAILABLE, CUPY_AVAILABLE


# ---------------------------------------------------------------------------
# Benchmark-driven GPU backend priority for optimization='auto'
# ---------------------------------------------------------------------------
# Compiled from Mac M4 Max (131 rows) and Narval A100 (111 rows) benchmarks.
# Format: {metric_name: {platform: [gpu_backend_1, gpu_backend_2]}}
# First available GPU backend in the list wins.
#
# 'auto' selects the best GPU backend only. Users choose CPU strategies
# explicitly: optimization=None (numpy) or optimization='numba'.
#
# The priority can be overridden per-call via the `priority` parameter:
#   get_metric('plv', optimization='auto', priority=['metal', 'torch'])
#
# Rationale:
#   MPS — einsum metrics: torch wins (batched matrix ops via Apple MPS).
#          sign-based/accorr: Metal custom kernels win (sign() and circular
#          correlation are not vectorizable; torch OOMs at ≥512ch for PLI/wPLI).
#          No Metal kernels for einsum metrics (torch_mps dominates at all scales).
#   CUDA — cuda_kernel first for all metrics: torch_cuda is faster at small/
#          medium scale but OOMs at realistic_hd (512ch) due to large
#          intermediate tensors. cuda_kernel computes pairwise without
#          materializing the full output tensor.
AUTO_PRIORITY = {
    # einsum metrics — torch wins on MPS, cuda_kernel safe-first on CUDA
    # (torch OOMs at ≥512ch on CUDA; cuda_kernel computes pairwise)
    "plv": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    "ccorr": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    "coh": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    "imcoh": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    "envcorr": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    "powcorr": {"mps": ["torch"], "cuda": ["cuda_kernel", "torch"]},
    # sign-based — custom kernels beat torch on both platforms
    "pli": {"mps": ["metal", "torch"], "cuda": ["cuda_kernel", "torch"]},
    "wpli": {"mps": ["metal", "torch"], "cuda": ["cuda_kernel", "torch"]},
    # accorr — Metal wins on MPS (circular correlation), cuda_kernel safe on CUDA
    "accorr": {"mps": ["metal", "torch"], "cuda": ["cuda_kernel", "torch"]},
}


def multiply_conjugate(
    real: np.ndarray, imag: np.ndarray, transpose_axes: tuple
) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate efficiently.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate

    Notes
    -----
    Implements: product = (real x real.T + imag x imag.T) - i(real x imag.T - imag x real.T)
    """
    formula = "jilm,jimk->jilk"
    product = (
        np.einsum(formula, real, real.transpose(transpose_axes))
        + np.einsum(formula, imag, imag.transpose(transpose_axes))
        - 1j
        * (
            np.einsum(formula, real, imag.transpose(transpose_axes))
            - np.einsum(formula, imag, real.transpose(transpose_axes))
        )
    )

    return product


def multiply_conjugate_time(
    real: np.ndarray, imag: np.ndarray, transpose_axes: tuple
) -> np.ndarray:
    """
    Computes the product of a complex array and its conjugate without collapsing time dimension.

    Similar to multiply_conjugate, but preserves the time dimension, which is
    needed for certain connectivity metrics like wPLI.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array and its complex conjugate with time dimension preserved
    """
    formula = "jilm,jimk->jilkm"
    product = (
        np.einsum(formula, real, real.transpose(transpose_axes))
        + np.einsum(formula, imag, imag.transpose(transpose_axes))
        - 1j
        * (
            np.einsum(formula, real, imag.transpose(transpose_axes))
            - np.einsum(formula, imag, real.transpose(transpose_axes))
        )
    )

    return product


def multiply_product(
    real: np.ndarray, imag: np.ndarray, transpose_axes: tuple
) -> np.ndarray:
    """
    Computes the product of two complex arrays (not conjugate) efficiently.

    Unlike multiply_conjugate, this computes z1 * z2 instead of z1 * conj(z2).
    Used in the adjusted circular correlation (accorr) metric.

    Parameters
    ----------
    real : np.ndarray
        Real part of the complex array
    imag : np.ndarray
        Imaginary part of the complex array
    transpose_axes : tuple
        Axes to transpose for matrix multiplication

    Returns
    -------
    product : np.ndarray
        Product of the array with itself (non-conjugate)
    """
    formula = "jilm,jimk->jilk"
    product = (
        np.einsum(formula, real, real.transpose(transpose_axes))
        - np.einsum(formula, imag, imag.transpose(transpose_axes))
        + 1j
        * (
            np.einsum(formula, real, imag.transpose(transpose_axes))
            + np.einsum(formula, imag, real.transpose(transpose_axes))
        )
    )

    return product


def multiply_conjugate_torch(c, s):
    """
    Compute z * conj(z) using torch tensors, collapsing time dimension.

    Torch equivalent of :func:`multiply_conjugate`. Uses the einsum convention
    ``e=epoch, f=freq, i=ch_row, j=ch_col, t=time``.

    Parameters
    ----------
    c : torch.Tensor
        Real part, shape (E, F, C, T).
    s : torch.Tensor
        Imaginary part, shape (E, F, C, T).

    Returns
    -------
    torch.Tensor
        Complex product, shape (E, F, C, C).
    """
    formula = "efit,efjt->efij"
    import torch

    return (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * (
        torch.einsum(formula, c, s) - torch.einsum(formula, s, c)
    )


def multiply_conjugate_time_torch(c, s):
    """
    Compute z * conj(z) using torch tensors, preserving time dimension.

    Torch equivalent of :func:`multiply_conjugate_time`. Produces a 5D tensor
    ``(E, F, C, C, T)`` — can be very large for high channel counts.

    Parameters
    ----------
    c : torch.Tensor
        Real part, shape (E, F, C, T).
    s : torch.Tensor
        Imaginary part, shape (E, F, C, T).

    Returns
    -------
    torch.Tensor
        Complex product, shape (E, F, C, C, T).
    """
    formula = "efit,efjt->efijt"
    import torch

    return (torch.einsum(formula, c, c) + torch.einsum(formula, s, s)) - 1j * (
        torch.einsum(formula, c, s) - torch.einsum(formula, s, c)
    )


class BaseMetric(ABC):
    """
    Abstract base class for connectivity metrics.

    All connectivity metrics should inherit from this class and implement
    the compute method.

    Parameters
    ----------
    optimization : str, optional
        Optimization strategy for computation. Options:
        - None: standard numpy (default)
        - 'auto': best available (torch > numba > numpy)
        - 'numba': numba JIT compilation (falls back to numpy if unavailable)
        - 'torch': PyTorch with auto-detected GPU (falls back gracefully)

    Attributes
    ----------
    optimization : str or None
        The requested optimization.
    name : str
        The name of the metric (class attribute to be defined by subclasses).
    """

    name: str = "base"

    #: Maps a backend name to the method implementing it. This table is the
    #: single source of truth for dispatch: ``compute`` looks the backend up
    #: here, so an unrecognised backend raises ``ValueError`` instead of silently
    #: falling through to numpy, and ``supports`` derives capability from the
    #: methods a subclass actually defines rather than from a hand-kept list.
    _BACKEND_METHODS = {
        "numpy": "_compute_numpy",
        "numba": "_compute_numba",
        "torch": "_compute_torch",
        "metal": "_compute_metal",
        "cuda_kernel": "_compute_cuda",
    }

    #: Human-readable backend names, used in fallback warnings.
    _BACKEND_LABELS = {
        "numpy": "numpy",
        "numba": "numba",
        "torch": "torch",
        "metal": "Metal",
        "cuda_kernel": "CUDA",
    }

    def __init__(
        self, optimization: Optional[str] = None, priority: Optional[list] = None
    ):
        self.optimization = optimization
        self._priority = priority
        self._backend, self._device = self._resolve_optimization(optimization, priority)

    @classmethod
    def supports(cls, backend: str) -> bool:
        """
        Whether this metric implements ``backend``.

        Capability is derived from the presence of the corresponding
        ``_compute_*`` method, so it cannot drift out of sync with the code.
        Not every metric has every backend — Metal kernels exist only for the
        sign-based metrics and ACCorr, because torch on MPS is faster for the
        einsum metrics at every channel count (see ``AUTO_PRIORITY``).

        A subclass written against the earlier contract, which overrides
        ``compute`` and branches on ``self._backend`` itself without defining
        ``_compute_numpy``, cannot be inspected this way. It is trusted with
        every known backend, exactly as before the capability check existed.

        Parameters
        ----------
        backend : str
            One of ``'numpy'``, ``'numba'``, ``'torch'``, ``'metal'``,
            ``'cuda_kernel'``. An unknown name returns ``False``.

        Returns
        -------
        bool
            True if the metric can run on ``backend``.

        Examples
        --------
        >>> from hypyp.sync import PLI, PLV
        >>> PLI.supports('metal'), PLV.supports('metal')
        (True, False)
        """
        method = cls._BACKEND_METHODS.get(backend)
        if method is None:
            return False
        if not cls._uses_table_dispatch():
            return cls._dispatches_itself()
        return callable(getattr(cls, method, None))

    @classmethod
    def _uses_table_dispatch(cls) -> bool:
        """Whether the class follows the ``_compute_*`` contract.

        The marker is its own ``_compute_numpy``: the reference implementation
        is the one method every metric of the current contract provides.
        """
        return cls._compute_numpy is not BaseMetric._compute_numpy

    @classmethod
    def _dispatches_itself(cls) -> bool:
        """Whether the class follows the earlier contract: its own ``compute``
        and no ``_compute_numpy``."""
        return not cls._uses_table_dispatch() and cls.compute is not BaseMetric.compute

    @classmethod
    def _cpu_fallback(cls) -> tuple:
        """CPU backend used when no GPU backend can be selected: numba when it
        is installed and the metric implements it, numpy otherwise."""
        if NUMBA_AVAILABLE and cls.supports("numba"):
            return "numba", "cpu"
        return "numpy", "cpu"

    @classmethod
    def _resolve_optimization(
        cls, optimization: Optional[str] = None, priority: Optional[list] = None
    ) -> tuple:
        """
        Resolves an optimization value to (backend, device).

        Implements fallback logic with warnings when requested
        optimization is not available.

        Parameters
        ----------
        optimization : str or None
            Requested optimization strategy:

            - ``None``: standard numpy, no acceleration (default).
            - ``'auto'``: best available backend, selected per-metric from
              the ``AUTO_PRIORITY`` table (compiled from benchmarks).
              See ``_resolve_auto`` for details.
            - ``'numba'``: JIT-compiled loops via numba. Falls back to numpy
              with a UserWarning if numba is not installed.
            - ``'torch'``: PyTorch tensors with auto-detected GPU (see
              ``_resolve_torch`` for device priority). Falls back to numpy
              with a UserWarning if torch is not installed.
            - ``'metal'``: Apple Metal compute shaders. Falls back to numpy
              with a UserWarning if PyObjC Metal is not available.
            - ``'cuda_kernel'``: Custom CUDA kernels via CuPy. Falls back
              to numpy with a UserWarning if CuPy is not available.
        priority : list of str, optional
            Custom backend priority list for ``'auto'`` mode. Overrides
            the default ``AUTO_PRIORITY`` table for this call.
            Example: ``['metal', 'torch', 'numba']``.

        Returns
        -------
        backend : str
            One of ``'numpy'``, ``'numba'``, ``'torch'``, ``'metal'``,
            ``'cuda_kernel'``.
        device : str
            One of ``'cpu'``, ``'mps'``, ``'cuda'``.

        Notes
        -----
        Fallback cascade for ``'auto'`` (per-metric, per-platform):
            Iterates ``AUTO_PRIORITY[metric][platform]`` and returns the
            first available backend. Falls back to numba → numpy if no
            GPU backend is available.

        Fallback cascade for explicit backends when unavailable:
            requested backend → numpy (with UserWarning)
        """
        if optimization is None:
            return "numpy", "cpu"

        if optimization == "auto":
            return cls._resolve_auto(priority)

        if optimization not in ("numba", "torch", "metal", "cuda_kernel"):
            raise ValueError(
                f"Unknown optimization '{optimization}'. "
                f"Options: None, 'auto', 'numba', 'torch', 'metal', 'cuda_kernel'"
            )

        # Capability before availability: a backend the machine can run is
        # still useless if this metric has no implementation for it. Without
        # this check the backend was accepted and dispatch quietly returned a
        # numpy result — the caller believed they were on the GPU.
        if not cls.supports(optimization):
            label = cls._BACKEND_LABELS[optimization]
            warnings.warn(
                f"{cls.name!r} has no {label} implementation, falling back to "
                f"numpy. Use optimization='auto' to select the best backend "
                f"available for this metric.",
                UserWarning,
                stacklevel=3,
            )
            return "numpy", "cpu"

        if optimization == "numba":
            if NUMBA_AVAILABLE:
                return "numba", "cpu"
            warnings.warn(
                "numba not installed, falling back to numpy. "
                "Install with: poetry install --with optim_numba",
                UserWarning,
                stacklevel=3,
            )
            return "numpy", "cpu"

        if optimization == "torch":
            if TORCH_AVAILABLE:
                return cls._resolve_torch()
            warnings.warn(
                "torch not installed, falling back to numpy. "
                "Install with: poetry install --with optim_torch",
                UserWarning,
                stacklevel=3,
            )
            return "numpy", "cpu"

        if optimization == "metal":
            if METAL_AVAILABLE:
                return "metal", "mps"
            warnings.warn(
                "PyObjC Metal not available, falling back to numpy. "
                "Install with: pip install pyobjc-framework-Metal",
                UserWarning,
                stacklevel=3,
            )
            return "numpy", "cpu"

        if optimization == "cuda_kernel":
            if CUPY_AVAILABLE:
                return "cuda_kernel", "cuda"
            warnings.warn(
                "CuPy not available, falling back to numpy. "
                "Install with: pip install cupy-cuda12x",
                UserWarning,
                stacklevel=3,
            )
            return "numpy", "cpu"

        # Unreachable: the membership test above already rejected any other
        # value. Kept as a guard in case a backend is added to that tuple
        # without a matching branch here.
        raise ValueError(
            f"Unknown optimization '{optimization}'. "
            f"Options: None, 'auto', 'numba', 'torch', 'metal', 'cuda_kernel'"
        )

    @classmethod
    def _resolve_auto(cls, priority: Optional[list] = None) -> tuple:
        """
        Benchmark-driven backend selection, per metric and platform.

        Uses the ``AUTO_PRIORITY`` table compiled from Mac M4 Max and
        Narval A100 benchmarks. Iterates the priority list and returns
        the first available backend.

        Parameters
        ----------
        priority : list of str, optional
            Custom priority list overriding ``AUTO_PRIORITY`` for this call.

        Returns
        -------
        backend : str
            Selected backend name.
        device : str
            Associated device (``'cpu'``, ``'mps'``, or ``'cuda'``).

        Notes
        -----
        Platform detection: MPS → 'mps', CUDA → 'cuda', else 'cpu'.
        On CPU-only machines, warns and falls back to numba → numpy.
        """
        if MPS_AVAILABLE:
            platform = "mps"
        elif CUDA_AVAILABLE:
            platform = "cuda"
        else:
            # No GPU — warn and fall back to CPU
            warnings.warn(
                "No GPU available. optimization='auto' selects the best GPU "
                "backend. Use optimization='numba' for CPU parallelism or "
                "optimization=None for numpy.",
                UserWarning,
                stacklevel=4,
            )
            return cls._cpu_fallback()

        if priority is None:
            priority = AUTO_PRIORITY.get(cls.name, {}).get(platform, [])

        # Backends of the priority list this metric has no implementation for,
        # remembered so the fallback warning can give the real reason.
        unimplemented = []
        for backend in priority:
            # Skip a backend this metric has no implementation for, so a
            # priority list falls through to the next candidate instead of
            # selecting a backend that would degrade to numpy at dispatch.
            if not cls.supports(backend):
                if backend in cls._BACKEND_LABELS:
                    unimplemented.append(cls._BACKEND_LABELS[backend])
                continue
            if backend == "torch" and TORCH_AVAILABLE:
                return cls._resolve_torch()
            if backend == "metal" and METAL_AVAILABLE:
                return "metal", "mps"
            if backend == "cuda_kernel" and CUPY_AVAILABLE:
                return "cuda_kernel", "cuda"

        # No GPU backend from priority list available — fall back. When a
        # backend was skipped for lack of an implementation, say so: "no GPU
        # backend available" would be false on a machine that has one.
        if unimplemented:
            reason = (
                f"{cls.name!r} has no {' or '.join(unimplemented)} "
                f"implementation, and no GPU backend of the priority list "
                f"is available on platform '{platform}'."
            )
        else:
            reason = (
                f"No GPU backend available for {cls.name!r} on platform '{platform}'."
            )
        warnings.warn(
            f"{reason} Falling back to CPU.",
            UserWarning,
            stacklevel=4,
        )
        return cls._cpu_fallback()

    @staticmethod
    def _resolve_torch() -> tuple:
        """
        Resolves the best available torch device, with warnings if no GPU found.

        Device priority: MPS > CUDA > CPU.

        MPS (Metal Performance Shaders) is Apple Silicon's GPU backend and is
        checked first. CUDA is checked second for NVIDIA GPUs on Linux/Windows.
        The two are mutually exclusive — a machine will have one or the other,
        never both. If neither is available, torch runs on CPU with a warning.

        Returns
        -------
        backend : str
            Always ``'torch'``.
        device : str
            One of ``'mps'``, ``'cuda'``, or ``'cpu'``.

        Notes
        -----
        MPS uses 32-bit float precision (``torch.float32``), so numerical
        results may differ from CPU/CUDA (64-bit) by up to ~1e-5. Tests
        should apply a looser tolerance when MPS is the active device.
        """
        if MPS_AVAILABLE:
            return "torch", "mps"
        if CUDA_AVAILABLE:
            return "torch", "cuda"
        warnings.warn("No GPU found, using torch on CPU", UserWarning, stacklevel=4)
        return "torch", "cpu"

    def compute(
        self, complex_signal: np.ndarray, n_samp: int, transpose_axes: tuple
    ) -> np.ndarray:
        """
        Compute the connectivity metric on the resolved backend.

        Dispatch is table-driven via ``_BACKEND_METHODS``: the backend chosen at
        construction selects the ``_compute_*`` method to run. Subclasses
        normally implement those methods and leave this one alone; a subclass
        that overrides ``compute`` itself bypasses the dispatch and is
        responsible for honouring ``self._backend``.

        Parameters
        ----------
        complex_signal : np.ndarray
            Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times).
        n_samp : int
            Number of time samples.
        transpose_axes : tuple
            Axes to transpose for matrix multiplication.

        Returns
        -------
        con : np.ndarray
            Connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).

        Raises
        ------
        NotImplementedError
            If the metric provides neither ``_compute_numpy`` nor its own
            ``compute``.
        ValueError
            If ``self._backend`` is not a backend this metric implements. The
            message names the metric and the backends it does implement. This
            is deliberate: an earlier hand-written ``if/elif`` chain per metric
            ended in a bare ``return self._compute_numpy(...)``, so an
            unhandled backend was indistinguishable from the numpy default and
            failed silently. Dispatching through the table fails loudly instead.

        Notes
        -----
        Output precision follows the backend and the input: the Metal kernels
        and torch on MPS compute in ``float32`` whatever the input, while the
        other backends follow the precision of ``complex_signal``.

        A subclass of the earlier contract may call ``super().compute(...)``
        from its own ``compute``. The base method was then abstract with an
        empty body and returned ``None``; it still does for such a subclass.
        """
        if not self._uses_table_dispatch():
            if self._dispatches_itself():
                return None
            raise NotImplementedError(
                f"{type(self).__name__} must implement _compute_numpy "
                f"(or override compute)."
            )
        if not self.supports(self._backend):
            implemented = [b for b in self._BACKEND_METHODS if self.supports(b)]
            raise ValueError(
                f"{self.name!r} cannot run on backend {self._backend!r}. "
                f"Backends implemented for this metric: {implemented}."
            )
        method = getattr(self, self._BACKEND_METHODS[self._backend])
        return method(complex_signal, n_samp, transpose_axes)

    def _compute_numpy(
        self, complex_signal: np.ndarray, n_samp: int, transpose_axes: tuple
    ) -> np.ndarray:
        """
        Reference implementation, in numpy. Always available.

        Every metric should provide this: it is the correctness oracle the
        accelerated backends are validated against, and the fallback target
        whenever a requested backend is unavailable or unimplemented. It is
        deliberately not an abstract method, so that a subclass written
        against the earlier contract (overriding ``compute`` only) can still
        be instantiated; this default raises ``NotImplementedError``.

        Parameters
        ----------
        complex_signal : np.ndarray
            Complex analytic signals with shape (n_epochs, n_freq, 2*n_channels, n_times).
        n_samp : int
            Number of time samples.
        transpose_axes : tuple
            Axes to transpose for matrix multiplication.

        Returns
        -------
        con : np.ndarray
            Connectivity matrix with shape (n_epoch, n_freq, 2*n_ch, 2*n_ch).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compute_numpy "
            f"(or override compute)."
        )
