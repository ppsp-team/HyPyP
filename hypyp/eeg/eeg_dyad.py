import warnings
from dataclasses import dataclass

import mne
from mne.preprocessing import ICA

from ..core.base_step import BaseStep

from ..dataclasses.psd import PSD

from ..connectivity.connectivities import Connectivities
from ..connectivity.connectivity import Connectivity
from ..core.base_dyad import BaseDyad
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
    compute_sync,
)

from ..signal.complex_signal import ComplexSignal

DEFAULT_EPOCHS_DURATION = 1

PREPROCESS_STEP_RAW = 'raw'
PREPROCESS_STEP_ICA_FIT = 'ica_fit'
PREPROCESS_STEP_AR = 'autoreject'

@dataclass
class EEGStep():
    epos: list[mne.Epochs] | None
    key: str

#class MneStep(BaseStep[mne.Epochs]):
#    @property
#    def n_times(self):
#        return self.obj.n_times
#        
#    @property
#    def sfreq(self):
#        return self.obj.info['sfreq']
#        
#    @property
#    def ch_names(self):
#        return self.obj.ch_names
#        
#    def plot(self, **kwargs):
#        return self.obj.plot(**kwargs)

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
    connectivities: dict[str, Connectivity]

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
            self.epos_add_step([
                mne.read_epochs(file_path1, preload=True),
                mne.read_epochs(file_path2, preload=True),
            ], PREPROCESS_STEP_RAW)
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
        self.epos_add_step([epo1, epo2], PREPROCESS_STEP_RAW)
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
    
    def epos_add_step(self, epos, name: str = ''):
        self.steps.append(EEGStep([epo.copy() for epo in epos], key=name))
    
    def create_epochs_from_raws(self, duration:float=DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        list1, list2 = create_epochs([self.raw1], [self.raw2], duration)
        self.epos_add_step([list1[0], list2[0]], PREPROCESS_STEP_RAW)
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
        self.epos_add_step(ICA_apply(self.icas, subject_idx, component_idx, self.epos, plot=plot, label=label, ch_type=ch_type, threshold=threshold), PREPROCESS_STEP_ICA_FIT)
        return self
    
    def prep_autoreject(self, strategy: str = None, threshold: float = None, show: bool = None, **kwargs) -> 'EEGDyad':
        # Forward arguments to underlying function
        if strategy is not None:
            kwargs['strategy'] = strategy
        if threshold is not None:
            kwargs['threshold'] = threshold
        if show is not None:
            # AR_local use the "verbose" argument to control plot display
            kwargs['verbose'] = show
        
        epos, dic_ar = AR_local(self.epos, **kwargs)
        self.epos_add_step(epos, PREPROCESS_STEP_AR)
        self.dic_ar = dic_ar
        return self

    def analyse_pow(
            self,
            fmin: float,
            fmax: float,
            n_fft: int = None,
            n_per_seg: int = None,
            epochs_average: bool = True,
            **kwargs,
    ) -> 'EEGDyad':
        if n_fft is None:
            n_fft = int(self.sfreq)
        if n_per_seg is None:
            n_per_seg = int(self.sfreq)

        self.psds = []
        for epo in self.epos:
            freqs, psd = pow(epo, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg, epochs_average=epochs_average, **kwargs)
            self.psds.append(PSD(freqs, psd, epo.ch_names))
        return self
    

    def compute_complex_signal_freq_bands(self, freq_bands = None, **kwargs):
        if freq_bands is not None:
            kwargs['freq_bands'] = freq_bands

        self.complex_signal = ComplexSignal(self.epos, self.sfreq, **kwargs)


    def analyse_connectivity(self, mode: str, epochs_average: bool = True):
        assert self.complex_signal is not None
        matrix = compute_sync(self.complex_signal.data, mode = mode, epochs_average = epochs_average)
        self.connectivities[mode] = Connectivities(mode, self.complex_signal.freq_bands, matrix, (self.epo1.ch_names, self.epo2.ch_names))
        return self

    def plot_icas_components(self) -> 'EEGDyad':
        for i, ica in enumerate(self.icas):
            print(f"Subject idx: {i}")
            ica.plot_components()
        return self
    
    