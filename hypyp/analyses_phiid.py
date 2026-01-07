#!/usr/bin/env python
# coding=utf-8

"""
Integrated Information Decomposition (Φ-ID) for Hyperscanning

| Option | Description |
| ------ | ----------- |
| title           | analyses_phiid.py |
| authors         | Rémy Ramadour, Guillaume Dumas |
| date            | 2026-01-06 |

This module implements Integrated Information Decomposition (Φ-ID) following
the HyperIT approach, using the phyid package from Imperial-MIND-lab.

Φ-ID extends classical PID (Partial Information Decomposition) by decomposing
mutual information into 16 atoms instead of 4, providing a more detailed
analysis of information structure.

References
----------
.. [1] Mediano, P. A., et al. (2021). "Greater than the parts: A review of the
       information decomposition approach to causal emergence." Philosophical
       Transactions of the Royal Society A, 380(2227), 20210246.
.. [2] Luppi, A. I., et al. (2022). "A synergistic core for human brain evolution
       and cognition." Nature Neuroscience, 25(6), 771-782.
"""

import numpy as np
import mne
from typing import Union, Dict
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr


def _compute_phiid_raw(
    epochs: list,
    epochs_average: bool = True
) -> np.ndarray:
    """
    Compute Integrated Information Decomposition following HyperIT algorithm.

    This is the core computation function that exactly replicates HyperIT's
    approach to computing Φ-ID using the phyid package.

    Parameters
    ----------
    epochs : list of mne.Epochs
        List of 2 MNE Epochs objects, one per participant.
        Each Epochs object should have shape (n_epochs, n_channels, n_times).
    epochs_average : bool, optional
        If True, average across epochs before computing Φ-ID.
        If False, compute Φ-ID for each epoch separately.
        Default is True.

    Returns
    -------
    phiid_matrix : np.ndarray
        Φ-ID matrix with 16 atoms per channel pair.
        Shape: (2*n_channels, 2*n_channels, 16) if epochs_average=True
               (n_epochs, 2*n_channels, 2*n_channels, 16) if epochs_average=False

    Notes
    -----
    The 16 Φ-ID atoms are (in order):
        0. rtr: Redundant transfer
        1. xtr: Unique transfer from X
        2. ytr: Unique transfer from Y
        3. str: Synergistic transfer
        4. rtx: Redundant transfer to X
        5. rty: Redundant transfer to Y
        6. rts: Redundant transfer synergy
        7. rr: Redundant redundancy
        8. xtx: Unique X transfer to X
        9. xty: Unique X transfer to Y
        10. xts: Unique X transfer synergy
        11. ux: Unique X
        12. yty: Unique Y transfer to Y
        13. yts: Unique Y transfer synergy
        14. uy: Unique Y
        15. ss: Synergistic synergy

    This function follows HyperIT's implementation:
    - Line 497: Matrix initialization (n_channels, n_channels, 16)
    - Line 548: Call to calc_PhiID(s1, s2, tau, kind='gaussian')
    - Line 554: Average across epochs using np.mean(..., axis=1)
    - Line 610: Store 16-atom vector in matrix[i, j]
    """
    # Validate input
    if not isinstance(epochs, list) or len(epochs) != 2:
        raise ValueError("epochs must be a list of 2 mne.Epochs objects")

    if not all(isinstance(epo, mne.BaseEpochs) for epo in epochs):
        raise TypeError("All elements in epochs must be mne.Epochs objects")

    # Extract data
    data1 = epochs[0].get_data(copy=True)  # (n_epochs, n_channels, n_times)
    data2 = epochs[1].get_data(copy=True)

    # Validate shapes match
    if data1.shape != data2.shape:
        raise ValueError("Both epochs must have the same shape")

    n_epochs, n_channels, n_times = data1.shape

    # Concatenate participants (HyperIT approach for inter-brain analysis)
    # Shape: (n_epochs, 2*n_channels, n_times)
    data_combined = np.concatenate([data1, data2], axis=1)

    # For MVP, only support epochs_average=True
    if not epochs_average:
        raise NotImplementedError("epochs_average=False is not yet implemented for Φ-ID")

    # Initialize result matrix (following HyperIT line 497)
    phiid_matrix = np.zeros((2 * n_channels, 2 * n_channels, 16))

    # Compute Φ-ID for each pair of channels
    # Following HyperIT's __compute_pair_or_group and __build_matrix methods
    loop_range = 2 * n_channels

    for i in range(loop_range):
        for j in range(loop_range):
            # Skip self-connections (following HyperIT line 597)
            if i == j:
                continue

            # Extract time series for channels i and j
            # HyperIT uses shape (n_samples, n_epochs) for phyid
            # (HyperIT line 589: s1, s2 = self._it_data1[:, i, :].T)
            # data_combined[:, i, :] has shape (n_epochs, n_samples)
            # .T gives (n_samples, n_epochs)
            s1 = data_combined[:, i, :].T  # (n_times, n_epochs)
            s2 = data_combined[:, j, :].T  # (n_times, n_epochs)

            # Compute Φ-ID using phyid (HyperIT line 548)
            # tau=1 is the default delay for temporal dynamics
            try:
                atoms_results, _ = calc_PhiID(s1, s2, tau=1, kind='gaussian', redundancy='MMI')

                # Average across time points (HyperIT line 554)
                # atoms_results is a dict with 16 keys
                # Each value is array of shape (n_times-tau,) when input is (n_times, n_epochs)
                # We need to average across time to get a single scalar per atom
                calc_atoms = np.mean(
                    np.array([atoms_results[atom] for atom in PhiID_atoms_abbr]),
                    axis=1
                )  # Shape: (16,) - one value per atom

            except Exception as e:
                # If computation fails (e.g., identical signals), set to zero
                # (HyperIT line 552)
                print(f"Warning: Φ-ID computation failed for pair ({i}, {j}): {e}")
                calc_atoms = np.zeros(16)

            # Store in matrix
            phiid_matrix[i, j] = calc_atoms

    return phiid_matrix


def compute_phiid(
    epochs: list,
    epochs_average: bool = True,
    return_format: str = 'dict'
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute Integrated Information Decomposition (Φ-ID) for hyperscanning.

    This function computes Φ-ID using the same algorithm as HyperIT, but
    provides output formats compatible with HyPyP's stats and visualization
    functions.

    Φ-ID extends classical PID by decomposing mutual information into 16
    atoms instead of 4, providing detailed insights into information structure:
    redundancy, uniqueness, synergy, and their interactions.

    Parameters
    ----------
    epochs : list of mne.Epochs
        List of 2 MNE Epochs objects, one per participant.
        Each Epochs object should have shape (n_epochs, n_channels, n_times).
    epochs_average : bool, optional
        If True, average Φ-ID across epochs.
        If False, return Φ-ID per epoch.
        Default is True.
    return_format : {'dict', 'hyperit'}, optional
        Output format:
        - 'dict': Return dictionary with 16 keys (one per atom), each value
                  is a matrix of shape (1, 2*n_channels, 2*n_channels).
                  **HyPyP-compatible format** for stats.py and viz.py.
        - 'hyperit': Return array of shape (2*n_channels, 2*n_channels, 16).
                     **HyperIT-compatible format** for comparison/validation.
        Default is 'dict'.

    Returns
    -------
    results : dict or np.ndarray
        If return_format='dict':
            Dictionary with 16 keys (atom names from PhiID_atoms_abbr):
            - 'rtr': Redundant transfer (1, 2*n_channels, 2*n_channels)
            - 'xtr': Unique transfer from X (1, 2*n_channels, 2*n_channels)
            - 'ytr': Unique transfer from Y (1, 2*n_channels, 2*n_channels)
            - 'str': Synergistic transfer (1, 2*n_channels, 2*n_channels)
            - ... (12 more atoms)

        If return_format='hyperit':
            Array of shape (2*n_channels, 2*n_channels, 16) where the last
            dimension contains all 16 atoms for each channel pair.

    Notes
    -----
    The Φ-ID framework decomposes mutual information I(X, Y; Z) into 16 atoms
    that capture different types of information relationships:

    - **Transfer atoms** (rtr, xtr, ytr, str): Information about the future
    - **Redundancy atoms** (rr, rtx, rty, rts): Shared information
    - **Unique atoms** (ux, uy, xtx, xty, yty, xts, yts): Exclusive information
    - **Synergy atoms** (ss): Emergent joint information

    Computation follows HyperIT's implementation:
    - Uses phyid.calculate.calc_PhiID with Gaussian estimator
    - Default tau=1 for temporal dynamics
    - Redundancy measure: 'mmi' (minimum mutual information)

    The output is compatible with HyPyP's infrastructure:
    - Each atom can be visualized using viz.plot_links_2d_inter()
    - Each atom can be tested using stats.statscondCluster()
    - Format (1, 2*n_channels, 2*n_channels) matches MI/TE outputs

    Examples
    --------
    >>> import mne
    >>> from hypyp import compute_phiid
    >>>
    >>> # Compute Φ-ID in HyPyP-compatible format
    >>> phiid_dict = compute_phiid([epo1, epo2], return_format='dict')
    >>>
    >>> # Visualize redundant transfer atom
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(phiid_dict['rtr'][0], cmap='viridis')
    >>> plt.title('Redundant Transfer')
    >>> plt.colorbar(label='Information (nats)')
    >>> plt.show()
    >>>
    >>> # Compare with HyperIT format
    >>> phiid_hyperit = compute_phiid([epo1, epo2], return_format='hyperit')
    >>> print(phiid_hyperit.shape)  # (62, 62, 16)
    >>>
    >>> # Access specific atom in HyperIT format
    >>> rtr_matrix = phiid_hyperit[:, :, 0]  # First atom = rtr

    See Also
    --------
    compute_mi_gaussian : Mutual Information (total information sharing)
    compute_te_gaussian : Transfer Entropy (directional information flow)

    References
    ----------
    .. [1] Mediano, P. A., et al. (2021). "Greater than the parts: A review of the
           information decomposition approach to causal emergence."
    .. [2] Luppi, A. I., et al. (2022). "A synergistic core for human brain
           evolution and cognition." Nature Neuroscience.
    """
    # Validate return_format
    if return_format not in ['dict', 'hyperit']:
        raise ValueError("return_format must be 'dict' or 'hyperit'")

    # Compute raw Φ-ID (HyperIT format)
    phiid_raw = _compute_phiid_raw(epochs, epochs_average=epochs_average)

    # Return in requested format
    if return_format == 'hyperit':
        # Return as-is (HyperIT-compatible)
        return phiid_raw

    else:  # return_format == 'dict'
        # Convert to HyPyP-compatible dict format
        phiid_dict = {}

        for idx, atom_name in enumerate(PhiID_atoms_abbr):
            # Extract atom matrix (2*n_channels, 2*n_channels)
            atom_matrix = phiid_raw[:, :, idx]

            # Add artificial frequency dimension for HyPyP compatibility
            # Shape: (1, 2*n_channels, 2*n_channels)
            atom_matrix_hypyp = atom_matrix[np.newaxis, :, :]

            phiid_dict[atom_name] = atom_matrix_hypyp

        return phiid_dict
