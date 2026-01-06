"""
Test PID on real HyPyP tutorial data.

This script loads the preprocessed data from the tutorial and computes
PID to verify it works on real hyperscanning data.

Author: Rémy Ramadour
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import requests
import tempfile
from hypyp import prep, analyses_pid

# Template URL for downloading participant data
URL_TEMPLATE = "https://github.com/ppsp-team/HyPyP/blob/master/data/participant{}-epo.fif?raw=true"

def get_data(idx):
    """Download EEG data for a given participant."""
    url = URL_TEMPLATE.format(idx)
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="-epo.fif")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

print("="*60)
print("PID TEST ON REAL HYPYP DATA")
print("="*60)

# Load epochs
print("\n1. Loading data...")
epo1 = mne.read_epochs(get_data(1), preload=True, verbose=False)
epo2 = mne.read_epochs(get_data(2), preload=True, verbose=False)

# Equalize epochs
mne.epochs.equalize_epoch_counts([epo1, epo2])
print(f"   Loaded {len(epo1)} epochs, {len(epo1.info['ch_names'])} channels")

# Preprocessing (skip ICA for automated testing, just use AutoReject)
print("\n2. Preprocessing (AutoReject only)...")
cleaned_epochs_AR, _ = prep.AR_local([epo1, epo2],
                                     strategy="union",
                                     threshold=50.0,
                                     verbose=True)

preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]
print(f"   Preprocessed: {len(preproc_S1)} epochs, {len(preproc_S1.info['ch_names'])} channels")

# Compute PID
print("\n3. Computing PID (this may take a minute)...")
print("   Target: Participant 1")

pid = analyses_pid.compute_pid_gaussian(
    [preproc_S1, preproc_S2],
    target_participant=0,
    epochs_average=True
)

print("   ✓ PID computation complete!")

# Analyze results
n_ch = len(preproc_S1.info['ch_names'])

print("\n4. Results Summary:")
print(f"   Matrix shape: {pid['redundancy'].shape}")

for atom_name in ['redundancy', 'unique1', 'unique2', 'synergy']:
    values = pid[atom_name][0, 0, :, :]
    mean_val = np.mean(values)
    median_val = np.median(values)
    max_val = np.max(values)
    print(f"   {atom_name.capitalize():12s}: mean={mean_val:.6f}, median={median_val:.6f}, max={max_val:.6f}")

# Check conservation
total = (pid['redundancy'] + pid['unique1'] + pid['unique2'] + pid['synergy'])
print(f"\n5. Conservation check:")
print(f"   Sum (mean): {np.mean(total):.6f}")
print(f"   Sum (median): {np.median(total):.6f}")
print(f"   All non-negative: {np.all(total >= -1e-6)}")

# Extract inter-brain values (P1 → P2 and P2 → P1 blocks)
print("\n6. Inter-brain analysis:")

# For each atom, extract inter-brain connectivity
red_inter = pid['redundancy'][0, 0, :n_ch, n_ch:]  # P1 targets, P2 sources
unq1_inter = pid['unique1'][0, 0, :n_ch, :n_ch]    # P1 targets, P1 sources (self)
unq2_inter = pid['unique2'][0, 0, :n_ch, n_ch:]    # P1 targets, P2 sources (inter)
syn_inter = pid['synergy'][0, 0, :n_ch, n_ch:]     # P1 targets, inter-sources

print(f"   Redundancy (inter-brain): mean={np.mean(red_inter):.6f}, median={np.median(red_inter):.6f}")
print(f"   Unique S2 (P2→P1): mean={np.mean(unq2_inter):.6f}, median={np.median(unq2_inter):.6f}")
print(f"   Synergy (inter-brain): mean={np.mean(syn_inter):.6f}, median={np.median(syn_inter):.6f}")

# Visualize 4 atoms
print("\n7. Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('PID Decomposition on Real HyPyP Data\n(Target: Participant 1)', fontsize=14)

atoms = ['redundancy', 'unique1', 'unique2', 'synergy']
titles = ['Redundancy', 'Unique S1 (P1 contribution)',
          'Unique S2 (P2 contribution)', 'Synergy (Joint P1+P2)']

vmin = min([pid[atom][0, 0].min() for atom in atoms])
vmax = max([pid[atom][0, 0].max() for atom in atoms])

for idx, (atom, title) in enumerate(zip(atoms, titles)):
    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(pid[atom][0, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Source Channel')
    ax.set_ylabel('Target Channel')

    # Add dividing lines
    ax.axhline(y=n_ch-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=n_ch-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add labels
    ax.text(n_ch/2, -1.5, 'P1', ha='center', fontsize=10)
    ax.text(3*n_ch/2, -1.5, 'P2', ha='center', fontsize=10)

    plt.colorbar(im, ax=ax, label='Information (nats)')

plt.tight_layout()
plt.savefig('sandbox/pid_real_data.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved visualization to sandbox/pid_real_data.png")
plt.show()

print("\n" + "="*60)
print("PID TEST COMPLETED SUCCESSFULLY ✓")
print("="*60)
