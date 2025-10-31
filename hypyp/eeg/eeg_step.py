from dataclasses import dataclass

import mne

from ..core.base_step import BaseStep

from ..signal.complex_signal import ComplexSignal
PREPROCESS_STEP_RAW = 'raw'
PREPROCESS_STEP_ICA_FIT = 'ica_fit'
PREPROCESS_STEP_AR = 'autoreject'

class EEGStep(BaseStep[mne.Epochs]):
    @property
    def epos(self):
        return self.obj

    @property
    def n_times(self):
        raise NotImplementedError('Not implemented on epochs')
        
    @property
    def sfreq(self):
        return self.obj.info['sfreq']
        
    @property
    def ch_names(self):
        return self.obj.ch_names
        
    def plot(self, **kwargs):
        return self.obj.plot(**kwargs)

@dataclass
class EEGDyadStep():
    """One DyadStep is a step that has been done on all the subjects of the dyad"""
    steps: list[EEGStep]
    name: str

    @property
    def epos(self) -> list[mne.Epochs]:
        out = []
        for s in self.steps:
            out.append(s.epos)
        return out
    
    @property
    def epo1(self) -> mne.Epochs:
        return self.epos[0]

    @property
    def epo2(self) -> mne.Epochs:
        return self.epos[1]
