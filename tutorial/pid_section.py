"""
PID Section for Information Theory Demo Notebook

This file contains the code for the PID section to be added to the notebook.
Copy-paste the cells below into the notebook after the TE section.

Author: Rémy Ramadour
Date: January 2026
"""

# =============================================================================
# CELL 34 (Markdown): PID Introduction
# =============================================================================
"""
## Information Theory Analysis: Partial Information Decomposition (PID)

**Partial Information Decomposition (PID)** goes beyond MI and TE by decomposing the information that two sources (P1, P2) provide about a target into **4 fundamental atoms**:

1. **Redundancy (Red)**: Information that BOTH sources provide about the target
   - Both brains carry the same predictive information
   - Example: Both participants see the same stimulus

2. **Unique S1 (Unq₁)**: Information ONLY source 1 (P1) provides about the target
   - Unique contribution of participant 1's brain activity
   - Information that P2 doesn't have

3. **Unique S2 (Unq₂)**: Information ONLY source 2 (P2) provides about the target
   - Unique contribution of participant 2's brain activity
   - Information that P1 doesn't have

4. **Synergy (Syn)**: Information created ONLY by the COMBINATION of both sources
   - Emergent information from the interaction
   - Cannot be predicted from either source alone
   - Example: XOR-like relationships

**Conservation Property**:
```
MI(S1, S2; T) = Red + Unq₁ + Unq₂ + Syn
```

**Key properties**:
- Decomposes total information into interpretable components
- Temporal domain (like MI and TE)
- Non-linear (Gaussian estimator)
- Reveals interaction dynamics beyond simple coupling
"""

# =============================================================================
# CELL 35 (Code): Compute PID
# =============================================================================
# Compute PID with target = Participant 1
# We decompose: how do P1 and P2's channels jointly inform P1's activity?
print("Computing PID (this may take a minute)...")

pid = analyses_pid.compute_pid_gaussian(
    [preproc_S1, preproc_S2],
    target_participant=0,  # Target from P1
    epochs_average=True
)

print(f"PID computation complete!")
print(f"Matrix shape: {pid['redundancy'].shape}")

# Summary statistics
for atom_name in ['redundancy', 'unique1', 'unique2', 'synergy']:
    values = pid[atom_name][0, 0, :, :]
    mean_val = np.mean(values)
    median_val = np.median(values)
    max_val = np.max(values)
    print(f"{atom_name.capitalize():12s}: mean={mean_val:.6f}, median={median_val:.6f}, max={max_val:.6f} nats")

# Check conservation
total = (pid['redundancy'] + pid['unique1'] + pid['unique2'] + pid['synergy'])
print(f"\nConservation check (sum of atoms): mean={np.mean(total):.6f} nats")

# =============================================================================
# CELL 36 (Markdown): Interpret Results
# =============================================================================
"""
### Interpreting PID Results

The PID matrices show how information is distributed across participants:

- **Matrix indices**: `[target_channel, source_channel]`
- **Target**: Channels from P1 (target_participant=0)
- **Sources**: All channels from both P1 and P2

**Key blocks to examine**:
1. **Upper-left (P1→P1)**: Intra-brain dependencies
2. **Upper-right (P2→P1)**: Inter-brain dependencies
3. **Diagonal values**: Self-prediction (high unique values expected)
4. **Off-diagonal**: Cross-channel interactions
"""

# =============================================================================
# CELL 37 (Code): Extract Inter-brain Values
# =============================================================================
# Extract inter-brain information (P1 and P2 jointly predicting P1)
red_inter = pid['redundancy'][0, 0, :n_ch, n_ch:]   # Redundant info P1+P2 → P1
unq1_inter = pid['unique1'][0, 0, :n_ch, :n_ch]     # Unique P1 → P1 (self)
unq2_inter = pid['unique2'][0, 0, :n_ch, n_ch:]     # Unique P2 → P1 (inter)
syn_inter = pid['synergy'][0, 0, :n_ch, n_ch:]      # Synergistic P1+P2 → P1

print("Inter-brain PID analysis (P2 → P1 targets):")
print(f"  Redundancy (shared P1+P2): mean={np.mean(red_inter):.6f} nats")
print(f"  Unique P2 contribution: mean={np.mean(unq2_inter):.6f} nats")
print(f"  Synergy (emergent): mean={np.mean(syn_inter):.6f} nats")

# Compare with intra-brain
unq1_intra_diag = np.diag(pid['unique1'][0, 0, :n_ch, :n_ch])
print(f"\nIntra-brain (P1 self-prediction diagonal): mean={np.mean(unq1_intra_diag):.6f} nats")
print(f"  → Higher values expected (channels predict themselves)")

# =============================================================================
# CELL 38 (Code): Visualize 4 Atoms
# =============================================================================
# Visualize all 4 PID atoms as heatmaps
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('PID Decomposition: How P1 and P2 Jointly Inform P1', fontsize=16, y=0.995)

atoms = ['redundancy', 'unique1', 'unique2', 'synergy']
titles = [
    'Redundancy\n(Shared by P1 and P2)',
    'Unique S1\n(Only P1 knows)',
    'Unique S2\n(Only P2 knows)',
    'Synergy\n(Emergent from P1+P2 interaction)'
]

# Use same color scale for all atoms for comparison
vmin = min([pid[atom][0, 0].min() for atom in atoms])
vmax = max([pid[atom][0, 0].max() for atom in atoms])

for idx, (atom, title) in enumerate(zip(atoms, titles)):
    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(pid[atom][0, 0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Source Channel')
    ax.set_ylabel('Target Channel (P1)')

    # Add dividing lines between P1 and P2 sources
    ax.axhline(y=n_ch-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(x=n_ch-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

    # Add participant labels
    ax.text(n_ch/2, -2, 'P1 sources', ha='center', fontsize=10, fontweight='bold')
    ax.text(3*n_ch/2, -2, 'P2 sources', ha='center', fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Information (nats)')

plt.tight_layout()
plt.show()

print("\nNote: Red dashed lines separate P1 sources (left) from P2 sources (right)")
print("Upper-right quadrant shows inter-brain information decomposition")

# =============================================================================
# CELL 39 (Code): Compare Atoms
# =============================================================================
# Compare the magnitude of different atoms
fig, ax = plt.subplots(figsize=(10, 6))

atom_names = ['Redundancy', 'Unique P1', 'Unique P2', 'Synergy']
atom_means = [
    np.mean(pid['redundancy'][0, 0, :n_ch, n_ch:]),  # Inter-brain
    np.mean(np.diag(pid['unique1'][0, 0, :n_ch, :n_ch])),  # P1 diagonal
    np.mean(pid['unique2'][0, 0, :n_ch, n_ch:]),  # Inter-brain
    np.mean(pid['synergy'][0, 0, :n_ch, n_ch:])  # Inter-brain
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(atom_names, atom_means, color=colors, alpha=0.7, edgecolor='black')

ax.set_ylabel('Mean Information (nats)', fontsize=12)
ax.set_title('PID Atom Comparison: Inter-brain Information', fontsize=14, fontweight='bold')
ax.set_ylim(bottom=0)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, atom_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print(f"  Redundancy = {atom_means[0]:.4f} nats: Information shared by both brains")
print(f"  Unique P1 = {atom_means[1]:.4f} nats: P1's unique self-information")
print(f"  Unique P2 = {atom_means[2]:.4f} nats: P2's unique contribution to P1")
print(f"  Synergy = {atom_means[3]:.4f} nats: Emergent from P1+P2 interaction")

# =============================================================================
# CELL 40 (Markdown): PID vs MI/TE Comparison
# =============================================================================
"""
### PID vs MI and TE

**How PID relates to MI and TE**:

1. **PID decomposes MI**:
   - MI measures total information sharing
   - PID breaks it down: Red + Unq₁ + Unq₂ + Syn = MI(S1,S2; T)

2. **PID complements TE**:
   - TE measures directional information flow
   - PID reveals HOW that information is structured:
     * Is it redundant (both sources)?
     * Unique to one source?
     * Synergistic (emergent)?

3. **Example interpretation**:
   - **High MI + High Red**: Both brains share similar information about target
   - **High MI + High Syn**: Interaction creates emergent information
   - **High TE + High Unq**: Strong directional unique contribution

**Use cases**:
- **Redundancy**: Indicates common processing or shared environmental input
- **Unique**: Reveals individual contributions in joint tasks
- **Synergy**: Suggests true collaboration and emergent coordination
"""

# =============================================================================
# CELL 41 (Code): Relationship between MI and PID
# =============================================================================
# Verify conservation: sum of atoms should approximate MI
# Note: MI(S1,S2; T) computed differently than pairwise MI

print("Relationship between PID atoms and total information:")
print("\nConservation property: Red + Unq1 + Unq2 + Syn should equal total MI")

# For each target channel, sum the atoms
total_info_per_target = np.mean(total[0, 0, :n_ch, :], axis=1)  # Average across sources

print(f"Mean total information per target channel: {np.mean(total_info_per_target):.6f} nats")
print(f"This represents MI(P1_channels, P2_channels; Target)")

# Show distribution of atoms as percentage
total_atom_sum = sum(atom_means)
if total_atom_sum > 0:
    print("\nRelative contribution of each atom (inter-brain):")
    for name, value in zip(atom_names, atom_means):
        percentage = (value / total_atom_sum) * 100
        print(f"  {name}: {percentage:.1f}%")

print("\n✓ PID analysis complete!")
print("  PID provides a fine-grained decomposition of information sharing")
print("  revealing redundant, unique, and synergistic contributions.")
