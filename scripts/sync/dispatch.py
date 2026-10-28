"""Edge cases of BaseMetric._resolve_auto with patched availability flags."""
import warnings
from unittest.mock import patch

from hypyp.sync import get_metric

B = "hypyp.sync.base."


def resolve(mode, flags, **kw):
    with patch.multiple(B.rstrip("."), **flags):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = get_metric(mode, optimization="auto", **kw)
    return (m._backend, m._device), [str(x.message)[:90] for x in w]


GPU = dict(TORCH_AVAILABLE=True, MPS_AVAILABLE=True, CUDA_AVAILABLE=False,
           METAL_AVAILABLE=True, NUMBA_AVAILABLE=True)
CPU = dict(TORCH_AVAILABLE=False, MPS_AVAILABLE=False, CUDA_AVAILABLE=False,
           METAL_AVAILABLE=False, NUMBA_AVAILABLE=True)
METAL_ONLY = dict(TORCH_AVAILABLE=False, MPS_AVAILABLE=False, CUDA_AVAILABLE=False,
                  METAL_AVAILABLE=True, NUMBA_AVAILABLE=False)

print("1. GPU machine, priority=['numba', 'torch'] (user wants numba first):")
print("   ", resolve("plv", GPU, priority=["numba", "torch"]))
print("2. GPU machine, priority=['numpy'] (user wants plain numpy):")
print("   ", resolve("plv", GPU, priority=["numpy"]))
print("3. GPU machine, priority=['numba'] alone:")
print("   ", resolve("plv", GPU, priority=["numba"]))
print("4. CPU-only machine, priority=['numpy'] (ignored?):")
print("   ", resolve("plv", CPU, priority=["numpy"]))
print("5. Metal installed, torch NOT installed (pip install hypyp[metal] only), pli auto:")
print("   ", resolve("pli", METAL_ONLY))
print("6. GPU machine, default auto for pli (metal first):")
print("   ", resolve("pli", GPU))
print("7. GPU machine, default auto for plv (torch mps -> float32):")
print("   ", resolve("plv", GPU))
