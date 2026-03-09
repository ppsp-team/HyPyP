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

**Note:** ACCorr supports hardware acceleration via `optimization` parameter.
See [Optimization Backends](#optimization-backends) below.

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

ACCorr supports three computational backends via the `optimization` parameter
in `compute_sync()` or the class constructor:

| Value | Backend | Device | Notes |
|-------|---------|--------|-------|
| `None` (default) | NumPy | CPU | Standard, no extra dependencies |
| `'auto'` | Best available | Auto | torch → numba → numpy |
| `'numba'` | Numba JIT | CPU | ~2× speedup; install: `poetry install --with optim_numba` |
| `'torch'` | PyTorch | GPU/CPU | ~20× speedup on GPU; install: `poetry install --with optim_torch` |

**Device priority for `'torch'` and `'auto'`:** MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.
MPS and CUDA are mutually exclusive; the best available device is selected automatically.

**Precision note:** MPS uses `float32`, which may introduce numerical differences
of up to ~1e-5 compared to CPU/CUDA (`float64`).

All other metrics currently use numpy only (`optimization` parameter is accepted
but ignored for non-ACCorr metrics).

---

## Usage

```python
from hypyp.analyses import compute_sync

# Standard (numpy)
con = compute_sync(complex_signal, 'accorr')

# With GPU acceleration
con = compute_sync(complex_signal, 'accorr', optimization='torch')

# Direct class instantiation
from hypyp.sync import ACCorr
metric = ACCorr(optimization='auto', show_progress=True)
con = metric.compute(complex_signal_internal, n_samp, transpose_axes)
```
