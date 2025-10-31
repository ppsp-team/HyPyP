# Factory class

import mne

from .eeg.eeg_dyad import EEGDyad, DEFAULT_EPOCHS_DURATION

class Dyad():
    def __init__():
        raise NotImplemented('This is a factory class with static method. It is not meant to be instanciated.')

    @staticmethod
    def from_eeg_files(epo1: mne.Epochs, epo2: mne.Epochs) -> EEGDyad:
        return EEGDyad.from_files(epo1, epo2)
    
    @staticmethod
    def from_eeg_epochs(epo1: mne.Epochs, epo2: mne.Epochs) -> EEGDyad:
        return EEGDyad.from_epochs(epo1, epo2)

    @staticmethod
    def from_eeg_raws(raw1: mne.io.Raw, raw2: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> EEGDyad:
        return EEGDyad.from_raws(raw1, raw2, epochs_duration=epochs_duration)
    
    @staticmethod
    def from_eeg_raw_merge(raw_merge: mne.io.Raw, epochs_duration: float = DEFAULT_EPOCHS_DURATION) -> EEGDyad:
        return EEGDyad.from_raw_merge(raw_merge, epochs_duration=epochs_duration)


