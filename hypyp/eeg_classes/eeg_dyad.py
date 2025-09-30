from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import mne
from mne.preprocessing import ICA

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

DEFAULT_EPOCHS_DURATION = 1

FREQ_BANDS_ALPHAS = {
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
}

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

class DyadConnectivity():
    mode: str
    inter: list[Connectivity]
    intras: list[list[Connectivity]]

    def __init__(self, mode: str, freq_bands: OrderedDict, matrix: np.ndarray):
        self.mode = mode
        self.inter = [] 
        self.intras = [[], []]

        # Determine the number of channels
        n_ch = matrix.shape[1] // 2

        for i, k in enumerate(freq_bands.keys()):
            range_axis_1 = slice(0, n_ch)
            range_axis_2 = slice(n_ch, 2*n_ch)
            values = matrix[i, range_axis_1, range_axis_2]
            C = (values - np.mean(values[:])) / np.std(values[:])
            self.inter.append(Connectivity(k, freq_bands[k], values, C))

        for i in [0, 1]:
            for i, k in enumerate(freq_bands.keys()):
                range_axis_1 = slice((i * n_ch), ((i + 1) * n_ch)) 
                range_axis_2 = range_axis_1
                values = matrix[i, range_axis_1, range_axis_2]
            
                # Remove self-connections
                values -= np.diag(np.diag(values))
            
                # Compute Z-score normalization for intra connectivity
                C = (values - np.mean(values[:])) / np.std(values[:])

                self.intras[i].append(Connectivity(k, freq_bands[k], values, C))
    
    @property
    def intra1(self):
        return self.intras[0]

    @property
    def intra2(self):
        return self.intras[1]

class EEGDyad(BaseDyad):
    steps: list[EEGStep]
    sfreq: float
    epos: list[mne.Epochs] | None
    raws: list[mne.io.Raw] | None
    icas: list[ICA] | None
    dic_ar: DicAR | None
    steps: list[EEGStep]
    psds: list[PSD]

    # analysis results
    connectivity: dict

    def __init__(self):
        super().__init__()
        self.steps = []
        self.sfreq = -1
        self.raws = None
        self.icas = None
        self.dic_ar = None
        self.psds = None
        self.connectivity = {}
    
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
    
    def prep_autoreject(self, **kwargs) -> 'EEGDyad':
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
    
    def analyse_connectivity_ccorr(self, freq_bands: OrderedDict = FREQ_BANDS_ALPHAS, compute_freq_bands_kwargs: dict = {}):
        complex_signal = compute_freq_bands(np.array(self.epos), self.sfreq, freq_bands, **compute_freq_bands_kwargs)

        # Compute connectivity using cross-correlation ('ccorr') and average across epochs
        mode = 'ccorr'
        matrix = compute_sync(complex_signal, mode=mode, epochs_average=True)
        self.connectivity[mode] = DyadConnectivity(mode, freq_bands, matrix)
        return self

    def plot_icas_components(self) -> 'EEGDyad':
        for i, ica in enumerate(self.icas):
            print(f"Subject idx: {i}")
            ica.plot_components("")
        return self
    
    