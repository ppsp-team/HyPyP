# Factory class

import mne

from .eeg.eeg_dyad import EEGDyad, DEFAULT_EPOCHS_DURATION

class Dyad():
    def __init__():
        raise NotImplemented('This is a factory class with static method. It is not meant to be instanciated.')

    @staticmethod
    def from_eeg_files(epo1: mne.Epochs, epo2: mne.Epochs, label: str = None, verbose: bool = False) -> EEGDyad:
        return EEGDyad.from_files(epo1, epo2, label, verbose=verbose)
    
    @staticmethod
    def from_eeg_epochs(epo1: mne.Epochs, epo2: mne.Epochs, label: str = None, verbose: bool = False) -> EEGDyad:
        return EEGDyad.from_epochs(epo1, epo2, label, verbose=verbose)

    @staticmethod
    def from_eeg_raws(raw1: mne.io.Raw, raw2: mne.io.Raw, label: str = None, epochs_duration: float = DEFAULT_EPOCHS_DURATION, verbose: bool = False) -> EEGDyad:
        return EEGDyad.from_raws(raw1, raw2, label, epochs_duration=epochs_duration, verbose=verbose)
    
    @staticmethod
    def from_eeg_raw_merge(raw_merge: mne.io.Raw, label: str = None, epochs_duration: float = DEFAULT_EPOCHS_DURATION, verbose: bool = False) -> EEGDyad:
        return EEGDyad.from_raw_merge(raw_merge, label, epochs_duration=epochs_duration, verbose=verbose)


