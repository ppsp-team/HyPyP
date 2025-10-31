from collections import OrderedDict

import numpy as np
import pandas as pd

from ..dataclasses.freq_band import FreqBands
from .connectivity import Connectivity
from ..plots import plot_coherence_matrix

class Connectivities():
    mode: str
    inter: list[Connectivity]
    intras: list[list[Connectivity]]

    def __init__(
        self,
        mode: str,
        freq_bands: FreqBands,
        matrix: np.ndarray,
        ch_names: list[str] | tuple[list[str], list[str]],
    ):
        self.mode = mode
        self.inter = [] 
        self.intras = [[], []]

        # Determine the number of channels
        n_ch = matrix.shape[1] // 2

        if not isinstance(ch_names, tuple):
            ch_names = (ch_names, ch_names)

        for i, freq_band in enumerate(freq_bands):
            range_axis_1 = slice(0, n_ch)
            range_axis_2 = slice(n_ch, 2*n_ch)
            values = matrix[i, range_axis_1, range_axis_2]
            C = (values - np.mean(values[:])) / np.std(values[:])
            self.inter.append(Connectivity(freq_band, values, C, ch_names))

        for subject_idx in [0, 1]:
            for i, freq_band in enumerate(freq_bands):
                range_axis_1 = slice((subject_idx * n_ch), ((subject_idx + 1) * n_ch)) 
                range_axis_2 = range_axis_1
                values = matrix[i, range_axis_1, range_axis_2]
            
                # Remove self-connections
                values -= np.diag(np.diag(values))
            
                # Compute Z-score normalization for intra connectivity
                C = (values - np.mean(values[:])) / np.std(values[:])

                ch_names_pair = (ch_names[subject_idx], ch_names[subject_idx])
                self.intras[subject_idx].append(Connectivity(freq_band, values, C, ch_names_pair))
    
    @property
    def intra1(self) -> list[Connectivity]:
        return self.intras[0]

    @property
    def intra2(self) -> list[Connectivity]:
        return self.intras[1]
    
    def get_connectivities_based_on_subject_id(self, subject_id: int = None):
        # TODO: should subject_id be zero based or one based for
        if subject_id is None:
            return self.inter

        if subject_id == 1:
            return self.intra1

        if subject_id == 2:
            return self.intra2

        raise ValueError(f"Cannot have connectivity of subject_id '{subject_id}'")
    
    def get_connectivity_for_freq_band(self, freq_band_name, subject_id: int = None):
        for connectivity in self.get_connectivities_based_on_subject_id(subject_id):
            if connectivity.freq_band.name == freq_band_name:
                return connectivity

        raise ValueError(f"Cannot find connectivity for freq_band {freq_band_name}")
    
    def plot_connectivity_for_freq_band(self, freq_band_name):
        conn = self.get_connectivity_for_freq_band(freq_band_name)
        flat = conn.zscore.flatten()
        dfs = []
        df_inter = pd.DataFrame({
            'coherence': flat,
            'channel1': np.repeat(conn.ch_names[0], len(conn.ch_names[1])),
            'channel2': np.array(conn.ch_names[1] * len(conn.ch_names[0])),
            'is_intra': np.full_like(flat, False),
            'is_intra_of': np.full_like(flat, None),
        })
        dfs.append(df_inter)

        for subject_id in [1, 2]:
            conn = self.get_connectivity_for_freq_band(freq_band_name, subject_id)
            flat = conn.zscore.flatten()
            df_intra = pd.DataFrame({
                'coherence': flat,
                'channel1': np.repeat(conn.ch_names[0], len(conn.ch_names[0])),
                'channel2': np.array(conn.ch_names[0] * len(conn.ch_names[0])),
                'is_intra': np.full_like(flat, True),
                'is_intra_of': np.full_like(flat, subject_id),
            })
            dfs.append(df_intra)
        
        df = pd.concat(dfs, ignore_index=True)

        return plot_coherence_matrix(df, 'subject1', 'subject2', 'channel1', 'channel2', [])

    
    
