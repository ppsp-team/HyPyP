# `hypyp.sync` — Connectivity Metrics Reference

This module provides modular, backend-optimizable connectivity metrics for
inter-brain synchrony analysis. All metrics are accessible via `compute_sync()`
in `hypyp.analyses` or by direct class instantiation.

---

## Metrics

### Phase Locking Value (`plv`)

**Formula:**
```
PLV = |⟨e^{i·Δφ(t)}⟩_t|
```
where `Δφ(t) = φ₁(t) − φ₂(t)` is the instantaneous phase difference.

Ranges from 0 (no synchrony) to 1 (perfect phase locking).

**Reference:** Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194–208.
https://doi.org/10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C

---

### Circular Correlation (`ccorr`)

**Formula:**
```
CCorr = Σ sin(α₁ − ᾱ₁) · sin(α₂ − ᾱ₂)
        ─────────────────────────────────────────────────
        √[ Σ sin²(α₁ − ᾱ₁) · Σ sin²(α₂ − ᾱ₂) ]
```
where `ᾱ` is the circular mean of `α`.

**Reference:** Fisher, N. I. (1995). *Statistical analysis of circular data*.
Cambridge University Press.

---

### Adjusted Circular Correlation (`accorr`)

**Formula:** Same structure as CCorr, but the centering values `m_adj` and `n_adj`
are optimized *per channel pair* rather than using the global circular mean:

```
n_adj = −(mean_diff − mean_sum) / 2
m_adj = mean_diff + n_adj
```
where `mean_diff = ∠⟨e^{i(α₁−α₂)}⟩` and `mean_sum = ∠⟨e^{i(α₁+α₂)}⟩`.

This per-pair adjustment corrects for biases introduced by non-uniform phase
distributions, providing a more accurate inter-brain synchrony estimate.

**Reference:** Zimmermann, M., Schultz-Nielsen, K., Dumas, G., & Konvalinka, I. (2024).
Arbitrary methodological decisions skew inter-brain synchronization estimates
in hyperscanning-EEG studies. *Imaging Neuroscience*, 2.
https://doi.org/10.1162/imag_a_00350

---

### Coherence (`coh`)

**Formula:**
```
Coh(f) = |⟨S₁₂(f)⟩|²
          ─────────────────────────────
          ⟨|S₁(f)|²⟩ · ⟨|S₂(f)|²⟩
```
where `S₁₂(f)` is the cross-spectrum.

Ranges from 0 to 1. Sensitive to volume conduction.

**Reference:** Nunez, P. L., & Srinivasan, R. (2006). *Electric fields of the brain:
the neurophysics of EEG*. Oxford University Press.

---

### Imaginary Coherence (`imaginary_coh` / `imcoh`)

**Formula:**
```
ImCoh(f) = Im(⟨S₁₂(f)⟩)
           ─────────────────────────────────────
           √[ ⟨|S₁(f)|²⟩ · ⟨|S₂(f)|²⟩ ]
```

Volume-conduction resistant: zero-lag (instantaneous) coupling contributes
only to the real part of coherence, so the imaginary part reflects true
time-lagged interactions.

**Reference:** Nolte, G., et al. (2004). See Coherence above.

---

### Phase Lag Index (`pli`)

**Formula:**
```
PLI = |⟨sign(Im(S₁₂(f)))⟩|
```

Ranges from 0 (no coupling or symmetric phase distribution) to 1
(perfectly asymmetric phase distribution). Insensitive to volume conduction.

**Reference:** Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).
Phase lag index: assessment of functional connectivity from multi channel EEG and
MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178–1193.
https://doi.org/10.1002/hbm.20346

---

### Weighted Phase Lag Index (`wpli`)

**Formula:**
```
WPLI = |⟨|Im(S₁₂)| · sign(Im(S₁₂))⟩|
       ─────────────────────────────────
              ⟨|Im(S₁₂)|⟩
```

Weighted version of PLI that reduces the impact of spectral noise while
maintaining insensitivity to zero-lag coupling.

**Reference:** Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., &
Pennartz, C. M. A. (2011). An improved index of phase-synchronization for
electrophysiological data in the presence of volume-conduction, noise and
sample-size bias. *NeuroImage*, 55(4), 1548–1565.
https://doi.org/10.1016/j.neuroimage.2011.01.055

---

### Envelope Correlation (`envelope_corr` / `envcorr`)

**Formula:**
```
EnvCorr = Pearson r( |s₁(t)|, |s₂(t)| )
```
where `|s(t)|` is the analytic amplitude (envelope) of the signal.

Captures low-frequency amplitude co-fluctuations, often reflecting
slow network dynamics.

**Reference:** Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., &
Engel, A. K. (2012). Large-scale cortical correlation structure of spontaneous
oscillatory activity. *Nature Neuroscience*, 15(6), 884–890.
https://doi.org/10.1038/nn.3101

---

### Power Correlation (`pow_corr` / `powcorr`)

**Formula:**
```
PowCorr = Pearson r( |s₁(t)|², |s₂(t)|² )
```
Similar to envelope correlation but uses instantaneous power rather than
amplitude. More sensitive to high-amplitude bursts.

---

## Optimization Backends

All 9 metrics support multiple computational backends via the `optimization`
parameter in `compute_sync()` or the class constructor.

### Backend Support Matrix

| Metric | numpy | numba | torch | metal | cuda_kernel |
|--------|:-----:|:-----:|:-----:|:-----:|:-----------:|
| PLV    |   x   |   x   |   x   |   --  |      x      |
| CCorr  |   x   |   x   |   x   |   --  |      x      |
| Coh    |   x   |   x   |   x   |   --  |      x      |
| ImCoh  |   x   |   x   |   x   |   --  |      x      |
| EnvCorr|   x   |   x   |   x   |   --  |      x      |
| PowCorr|   x   |   x   |   x   |   --  |      x      |
| PLI    |   x   |   x   |   x   |   x   |      x      |
| wPLI   |   x   |   x   |   x   |   x   |      x      |
| ACCorr |   x   |   x   |   x   |   x   |      x      |

### Backend Descriptions

| Value | Backend | Device | Notes |
|-------|---------|--------|-------|
| `None` (default) | NumPy | CPU | Standard, no extra dependencies |
| `'auto'` | Best available | Auto | Selects best GPU backend per metric and platform |
| `'numba'` | Numba JIT | CPU | Fused single-pass kernels with `prange` parallelism |
| `'torch'` | PyTorch | GPU/CPU | Batched einsum; MPS (Apple) / CUDA (NVIDIA) / CPU |
| `'metal'` | Metal shaders | Apple GPU | Custom compute shaders for PLI, wPLI, ACCorr only |
| `'cuda_kernel'` | CuPy RawKernel | NVIDIA GPU | Custom CUDA kernels; float64 precision |

### `optimization='auto'` — Benchmark-Driven Dispatch

The `'auto'` mode selects the best GPU backend for each metric based on
benchmark data compiled from Mac M4 Max (131 runs) and Narval A100 (111 runs).

**MPS (Apple Silicon):**
- Einsum metrics (PLV, CCorr, Coh, ImCoh, EnvCorr, PowCorr): torch (batched BLAS)
- Sign-based (PLI, wPLI) + ACCorr: Metal custom kernels

**CUDA (NVIDIA):**
- All metrics: `cuda_kernel` first (pairwise computation, OOM-safe at 512+ channels),
  with torch as fallback.

The priority can be overridden per-call:
```python
get_metric('plv', optimization='auto', priority=['torch', 'cuda_kernel'])
```

If no GPU backend is available, `'auto'` falls back to numba, then numpy.

### Precision

- **CPU / CUDA (`float64`):** reference precision, `rtol=1e-9, atol=1e-10`
- **MPS / Metal (`float32`):** up to ~1e-5 difference vs CPU reference.
  Sign-based metrics (PLI, wPLI) may show larger differences (`rtol=1e-2`)
  near the sign discontinuity at zero.

---

## Architecture

```
hypyp/sync/
├── __init__.py          # Registry, get_metric(), exports
├── base.py              # BaseMetric, AUTO_PRIORITY, helpers
├── plv.py ... wpli.py   # One file per metric (9 files)
└── kernels/             # Custom GPU kernels
    ├── __init__.py      # METAL_AVAILABLE, CUPY_AVAILABLE flags
    ├── _metal_dispatch.py   # Shared Metal pairwise dispatch
    ├── _cuda_dispatch.py    # Shared CUDA pairwise dispatch
    ├── metal_phase.py       # PLI, wPLI Metal shaders
    ├── metal_accorr.py      # ACCorr Metal shader
    ├── cuda_phase.py        # PLI, wPLI, PLV, CCorr CUDA kernels
    ├── cuda_amplitude.py    # Coh, ImCoh, EnvCorr, PowCorr CUDA kernels
    └── cuda_accorr.py       # ACCorr CUDA kernel
```

Each metric class inherits from `BaseMetric` and implements:
- `_compute_numpy()` — always available (reference implementation)
- `_compute_numba()` — fused loop with `numba.prange` parallelism
- `_compute_torch()` — batched einsum on auto-detected device
- `_compute_metal()` — Metal shader dispatch (PLI, wPLI, ACCorr only)
- `_compute_cuda()` — CUDA RawKernel dispatch

Backend selection happens at `__init__()`, dispatch at `compute()`.

---

## Installation

```bash
# Core (numpy backend always available)
pip install hypyp

# CPU parallelism
pip install "hypyp[numba]"

# GPU acceleration (PyTorch)
pip install "hypyp[torch]"

# Apple Silicon Metal shaders (PLI, wPLI, ACCorr)
pip install "hypyp[metal]"

# NVIDIA CUDA kernels (all metrics, requires CUDA 12.x)
pip install "hypyp[cupy]"
```

---

## Usage

```python
from hypyp.analyses import compute_sync

# Standard (numpy)
con = compute_sync(complex_signal, 'plv')

# Best available GPU backend
con = compute_sync(complex_signal, 'plv', optimization='auto')

# Specific backend
con = compute_sync(complex_signal, 'pli', optimization='metal')

# Custom priority
con = compute_sync(complex_signal, 'coh', optimization='auto',
                   priority=['torch', 'cuda_kernel'])

# Direct class instantiation
from hypyp.sync import get_metric
metric = get_metric('accorr', optimization='auto')
con = metric.compute(complex_signal_internal, n_samp, transpose_axes)
```
