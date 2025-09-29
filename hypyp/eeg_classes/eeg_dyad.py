import mne
from dataclasses import dataclass
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

DEFAULT_EPOCHS_DURATION = 1

@dataclass
class EEGStep():
    epos: list[mne.Epochs] | None

class EEGDyad(BaseDyad):
    epos: list[mne.Epochs] | None
    raws: list[mne.io.Raw] | None
    icas: list[ICA] | None
    steps: list[EEGStep]
    dic_ar: DicAR | None

    def __init__(self):
        super().__init__()
        self.raws = None
        self.icas = None
        self.steps = []
        self.dic_ar = None
    
    @staticmethod
    def from_files(file_path1: str, file_path2: str):
        self = EEGDyad()
        if file_path1.endswith('-epo.fif') and file_path2.endswith('-epo.fif'):
            self.epos = [
                mne.read_epochs(file_path1, preload=True),
                mne.read_epochs(file_path2, preload=True),
            ]
            self._equalize_epoch_counts()
        else:
            raise NotImplementedError()

        return self
    
    @staticmethod
    def from_epochs(epo1: mne.Epochs, epo2: mne.Epochs) -> 'EEGDyad':
        self = EEGDyad()
        self.epos = [epo1, epo2]
        self._equalize_epoch_counts()
        return self
    
    @staticmethod
    def from_raws(raw1: mne.io.Raw, raw2: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        self = EEGDyad()
        self.raws = [raw1, raw2]
        self.create_epochs_from_raws(epochs_duration)
        return self
    
    @staticmethod
    def from_raw_merge(raw_merge: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        self = EEGDyad()
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
    
    def create_epochs_from_raws(self, duration:float=DEFAULT_EPOCHS_DURATION) -> 'EEGDyad':
        list1, list2 = create_epochs([self.raw1], [self.raw2], duration)
        self.epos = [list1[0], list2[0]]
        return self
    

    def ica_fit(
            self,
            n_components: int,
            method: str = 'infomax',
            fit_params: dict = dict(extended=True),
            random_state: int = 42,
    ) -> 'EEGDyad':
        self.icas = ICA_fit(self.epos, n_components, method, fit_params, random_state)
        return self
    
    def ica_apply(
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
    
    def plot_icas_components(self):
        for i, ica in enumerate(self.icas):
            print(f"Subject idx: {i}")
            ica.plot_components("")
        return self
    
    def run_autoreject(self, **kwargs):
        epos, dic_ar = AR_local(self.epos, **kwargs)
        self.epos = epos
        self.dic_ar = dic_ar