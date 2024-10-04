from typing import List, Tuple

import mne

class Subject:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw
        self.best_ch_names: List[str] | None
        self.events: any # we should know what type this is
        self.epochs: mne.Epochs

    def load_snirf_file(self, filepath):
        self.filepath = filepath        
        self.raw = mne.io.read_raw_fif(filepath, verbose=True, preload=True)
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        if self.raw is None:
            raise RuntimeError('Must load raw data first')

        if self.best_ch_names is None:
            raise RuntimeError('No "best channels" has been set')

        ch_picks = mne.pick_channels(self.raw.ch_names, include = self.best_ch_names)
        best_channels = self.raw.copy().pick(ch_picks)
        self.events, self.event_dict = mne.events_from_annotations(best_channels)
        self.epochs = mne.Epochs(
            best_channels,
            self.events,
            event_id = self.event_dict,
            tmin = tmin,
            tmax = tmax,
            baseline = baseline,
            reject_by_annotation=False)

class DyadFNIRS:
    def __init__(self, s1: Subject, s2: Subject):
        self.s1: Subject = s1
        self.s2: Subject = s2

    
    @property 
    def subjects(self):
        return [self.s1, self.s2]
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        _ = [s.load_epochs(tmin, tmax, baseline) for s in self.subjects]
    