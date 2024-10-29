from typing import List, Tuple
from os import listdir
from os.path import isfile, join

import numpy as np
import mne
import itertools as itertools
import scipy.io

class SubjectFNIRS:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw
        self.best_ch_names: List[str] | None
        self.events: any # we should know what type this is
        self.epochs: mne.Epochs

    def load_fif_file(self, filepath):
        self.filepath = filepath        
        self.raw = mne.io.read_raw_fif(filepath, verbose=True, preload=True)
        return self
    
    def load_snirf_file(self, filepath):
        self.filepath = filepath        
        self.raw = mne.io.read_raw_snirf(filepath, verbose=True, preload=True)
        return self
    
    def set_best_ch_names(self, ch_names):
        self.best_ch_names = ch_names
        return self
    
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
        return self
