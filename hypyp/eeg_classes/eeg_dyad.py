from collections import OrderedDict
from dataclasses import dataclass
import warnings

import numpy as np
import mne
import pandas as pd
from mne.preprocessing import ICA
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

from .base_dyad import BaseDyad
from ..utils import (
    create_epochs,
    merge,
    split,
)
from ..prep import (
    ICA_fit,
    ICA_apply,
    AR_local,
    DicAR,
)
from ..analyses import (
    pow,
    compute_freq_bands,
    compute_sync,
)
from ..plots import (
    plot_coherence_matrix,
)

from .complex_signal import ComplexSignal

DEFAULT_EPOCHS_DURATION = 1

@dataclass
class EEGStep():
    epos: list[mne.Epochs] | None

@dataclass
class PSD():
    freqs: np.array
    psd: np.ndarray

@dataclass
class Connectivity():
    freq_band_name: str
    freq_band: tuple[float, float]
    values: np.ndarray
    zscore: np.ndarray
    ch_names: tuple[list[str], list[str]]

    def plot_zscore(self, ax:Axes = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        sns.heatmap(self.zscore, cmap='viridis', cbar=True, ax=ax)
        return fig


class DyadConnectivity():
    mode: str
    inter: list[Connectivity]
    intras: list[list[Connectivity]]

    def __init__(
            self,
        mode: str,
        freq_bands: OrderedDict,
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

        for i, k in enumerate(freq_bands.keys()):
            range_axis_1 = slice(0, n_ch)
            range_axis_2 = slice(n_ch, 2*n_ch)
            values = matrix[i, range_axis_1, range_axis_2]
            C = (values - np.mean(values[:])) / np.std(values[:])
            self.inter.append(Connectivity(k, freq_bands[k], values, C, ch_names))

        for subject_idx in [0, 1]:
            for i, k in enumerate(freq_bands.keys()):
                range_axis_1 = slice((subject_idx * n_ch), ((subject_idx + 1) * n_ch)) 
                range_axis_2 = range_axis_1
                values = matrix[i, range_axis_1, range_axis_2]
            
                # Remove self-connections
                values -= np.diag(np.diag(values))
            
                # Compute Z-score normalization for intra connectivity
                C = (values - np.mean(values[:])) / np.std(values[:])

                ch_names_pair = (ch_names[subject_idx], ch_names[subject_idx])
                self.intras[subject_idx].append(Connectivity(k, freq_bands[k], values, C, ch_names_pair))
    
    @property
    def intra1(self) -> list[Connectivity]:
        return self.intras[0]

    @property
    def intra2(self) -> list[Connectivity]:
        return self.intras[1]
    
    def get_connectivities_based_on_subject_id(self, subject_id: int = None):
        if subject_id is None:
            return self.inter

        if subject_id == 1:
            return self.intra1

        if subject_id == 2:
            return self.intra2

        raise ValueError(f"Cannot have connectivity of subject_id '{subject_id}'")
    
    def get_connectivity_for_freq_band(self, freq_band_name, subject_id: int = None):
        for connectivity in self.get_connectivities_based_on_subject_id(subject_id):
            if connectivity.freq_band_name == freq_band_name:
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

    
    
class EEGDyad(BaseDyad):
    steps: list[EEGStep]
    sfreq: float
    epos: list[mne.Epochs] | None
    raws: list[mne.io.Raw] | None
    icas: list[ICA] | None
    dic_ar: DicAR | None
    steps: list[EEGStep]
    psds: list[PSD]
    complex_signal: ComplexSignal | None

    # analysis results
    connectivities: dict[str, DyadConnectivity]

    def __init__(self):
        super().__init__()
        self.steps = []
        self.sfreq = -1
        self.raws = None
        self.icas = None
        self.dic_ar = None
        self.psds = None
        self.complex_signal = None
        self.connectivities = {}
    
    @staticmethod
    def from_files(file_path1: str, file_path2: str):
        self = EEGDyad()
        if file_path1.endswith('-epo.fif') and file_path2.endswith('-epo.fif'):
            self.epos = [
                mne.read_epochs(file_path1, preload=True),
                mne.read_epochs(file_path2, preload=True),
            ]
            assert self.epo1.info['sfreq'] == self.epo2.info['sfreq']
            self.sfreq = self.epo1.info['sfreq']
            self._equalize_epoch_counts()
        else:
            raise NotImplementedError()

        return self
    
    @staticmethod
    def from_epochs(epo1: mne.Epochs, epo2: mne.Epochs) -> 'EEGDyad':
        assert epo1.info['sfreq'] == epo2.info['sfreq']
        self = EEGDyad()
        self.sfreq = epo1.info['sfreq']
        self.epos = [epo1, epo2]
        self._equalize_epoch_counts()
        return self
    
    @staticmethod
    def from_raws(raw1: mne.io.Raw, raw2: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        assert raw1.info['sfreq'] == raw2.info['sfreq']
        self = EEGDyad()
        self.sfreq = raw1.info['sfreq']
        self.raws = [raw1, raw2]
        self.create_epochs_from_raws(epochs_duration)
        return self
    
    @staticmethod
    def from_raw_merge(raw_merge: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        self = EEGDyad()
        self.sfreq = raw_merge.info['sfreq']
        self.raws = split(raw_merge)
        self.create_epochs_from_raws(epochs_duration)
        return self

    def _assert_has_epochs(self):
        if len(self.steps) ==  0:
            raise RuntimeError('Epochs not created. Use create_epochs(duration)')
    
    def _assert_has_raws(self):
        if self.raws is None:
            raise RuntimeError('No raw data.')
    
    def _assert_has_icas(self):
        if self.icas is None:
            raise RuntimeError('ICAs is None. Make sure to call "ica_fit()" first.')
    
    def _assert_has_psds(self):
        if self.psds is None:
            raise RuntimeError('ICAs is None. Make sure to call "ica_fit()" first.')
    
    def _equalize_epoch_counts(self):
        if len(self.epo1) != len(self.epo2):
            warnings.warn(f"The 2 epochs objects don't have the same length: {len(self.epo1)} != {len(self.epo2)}")
            mne.epochs.equalize_epoch_counts(self.epos)
    
    @property
    def epos(self):
        return self.steps[-1].epos
    
    @epos.setter
    def epos(self, epos):
        self.steps.append(EEGStep([epo.copy() for epo in epos]))

    @property
    def epochs_merged(self) -> mne.Epochs:
        self._assert_has_epochs()
        return merge(self.epo1, self.epo2)
    
    @property
    def epo1(self) -> mne.Epochs:
        self._assert_has_epochs()
        return self.epos[0]
    
    @property
    def epo2(self) -> mne.Epochs:
        self._assert_has_epochs()
        return self.epos[1]
    
    @property
    def raw1(self) -> mne.io.Raw:
        self._assert_has_raws()
        return self.raws[0]
    
    @property
    def raw2(self) -> mne.io.Raw:
        self._assert_has_raws()
        return self.raws[1]
    
    @property
    def ica1(self) -> ICA:
        self._assert_has_icas()
        return self.icas[0]
    
    @property
    def ica2(self) -> ICA:
        self._assert_has_icas()
        return self.icas[1]
    
    @property
    def psd1(self) -> PSD:
        self._assert_has_psds()
        return self.psds[0]
    
    @property
    def psd2(self) -> PSD:
        self._assert_has_psds()
        return self.psds[1]
    
    def create_epochs_from_raws(self, duration:float=DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        list1, list2 = create_epochs([self.raw1], [self.raw2], duration)
        self.epos = [list1[0], list2[0]]
        return self

    def prep_ica_fit(
            self,
            n_components: int,
            method: str = 'infomax',
            fit_params: dict = dict(extended=True),
            random_state: int = 42,
            **kwargs,
    ) -> 'EEGDyad':
        self.icas = ICA_fit(self.epos, n_components, method, fit_params, random_state, **kwargs)
        return self
    
    def prep_ica_apply(
            self,
            subject_idx: int,
            component_idx: int,
            label: str = 'blink',
            ch_type: str = 'eeg',
            threshold: float = 0.9,
            plot: bool = False,
    ) -> 'EEGDyad':
        self._assert_has_icas()
        self.epos = ICA_apply(self.icas, subject_idx, component_idx, self.epos, plot=plot, label=label, ch_type=ch_type, threshold=threshold)
        return self
    
    def prep_autoreject(self, strategy: str = None, threshold: float = None, verbose: bool = None, **kwargs) -> 'EEGDyad':
        # Forward arguments to underlying function
        if strategy is not None:
            kwargs['strategy'] = strategy
        if threshold is not None:
            kwargs['threshold'] = threshold
        if verbose is not None:
            kwargs['verbose'] = verbose
        
        epos, dic_ar = AR_local(self.epos, **kwargs)
        self.epos = epos
        self.dic_ar = dic_ar
        return self

    def analyse_pow(
            self,
            fmin: float,
            fmax: float,
            n_fft: int = 1000,
            n_per_seg: int = 1000,
            epochs_average: bool = True,
            **kwargs,
    ) -> 'EEGDyad':
        self.psds = []
        for epo in self.epos:
            freqs, psd = pow(epo, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg, epochs_average=epochs_average, **kwargs)
            self.psds.append(PSD(freqs, psd))
        return self
    

    def compute_complex_signal_freq_bands(self, freq_bands = None, **kwargs):
        if freq_bands is not None:
            kwargs['freq_bands'] = freq_bands

        self.complex_signal = ComplexSignal(self.epos, self.sfreq, **kwargs)


    def analyse_connectivity(self, mode: str, epochs_average: bool = True):
        assert self.complex_signal is not None
        matrix = compute_sync(self.complex_signal.data, mode = mode, epochs_average = epochs_average)
        self.connectivities[mode] = DyadConnectivity(mode, self.complex_signal.freq_bands, matrix, (self.epo1.ch_names, self.epo2.ch_names))
        return self

    def plot_icas_components(self) -> 'EEGDyad':
        for i, ica in enumerate(self.icas):
            print(f"Subject idx: {i}")
            ica.plot_components()
        return self
    
    