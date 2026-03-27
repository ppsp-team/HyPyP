import mne

from ..core import BaseDyad
from ..eeg import EEGDyad
from ..fnirs import FNIRSDyad
from ..dataclasses.synchrony import SynchronyTimeSeries

class MultimodalDyad():
    eeg: EEGDyad | None
    fnirs: FNIRSDyad | None

    def __init__(self, eeg:EEGDyad=None, fnirs:FNIRSDyad=None):
        self.eeg = eeg
        self.fnirs = fnirs

    @property
    def modalities(self) -> list[BaseDyad]:
        ret: list[BaseDyad] = []
        if self.eeg is not None:
            ret.append(self.eeg)
        if self.fnirs is not None:
            ret.append(self.fnirs)
        return ret

    def add_eeg(self, eeg:EEGDyad):
        if self.eeg is not None:
            raise ValueError('MultimodalDyad already has eeg')
        self.eeg = eeg

    def add_fnirs(self, fnirs:EEGDyad):
        if self.fnirs is not None:
            raise ValueError('MultimodalDyad already has fnirs')
        self.fnirs = fnirs

    def get_synchrony_time_series(self) ->  list[SynchronyTimeSeries]:
        return [modality.get_synchrony_time_series() for modality in self.modalities]
