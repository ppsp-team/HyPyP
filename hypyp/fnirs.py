from typing import List, Tuple
from os import listdir
from os.path import isfile, join

import numpy as np
import mne
import itertools as itertools
import scipy.io

class DyadSignals:
    def __init__(self, x, y1, y2, info_table1=[], info_table2=[]):
        self.x = x
        self.n = len(x)
        self.dt = x[1] - x[0]
        self.y1 = y1
        self.y2 = y2
        self.info_table1 = info_table1
        self.info_table2 = info_table2

    def sub_hundred(self, range):
        if range[0] == 0 and range[1] == 100:
            return self

        signal_from = self.n * range[0] // 100
        signal_to = self.n * range[1] // 100
        # TODO: use mask instead
        return DyadSignals(
            self.x[signal_from:signal_to],
            self.y1[signal_from:signal_to],
            self.y2[signal_from:signal_to],
            self.info_table1,
            self.info_table2,
        )
    

class DataLoaderFNIRS:
    base_path = join('data')
    fnirs_path = join('data', 'FNIRS')

    @staticmethod
    def list_all_files():
        base = [join(DataLoaderFNIRS.base_path, f) for f in listdir(DataLoaderFNIRS.base_path) if isfile(join(DataLoaderFNIRS.base_path, f))]
        fnirs = [join(DataLoaderFNIRS.fnirs_path, f) for f in listdir(DataLoaderFNIRS.fnirs_path) if isfile(join(DataLoaderFNIRS.fnirs_path, f))]
        return base + fnirs

    @staticmethod
    def list_fif_files():
        return [f for f in DataLoaderFNIRS.list_all_files() if f.endswith('.fif')]
    
    @staticmethod
    def read_two_signals_from_mat(file_path, id1, id2):
        mat = scipy.io.loadmat(file_path)
        x = mat['t'].flatten().astype(np.float64, copy=True)
        y1 = mat['d'][:, id1].flatten().astype(np.complex128, copy=True)
        y2 = mat['d'][:, id2].flatten().astype(np.complex128, copy=True)

        return DyadSignals(x, y1, y2)
    
    def __init__(self):
        pass

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

class DyadFNIRS:
    def __init__(self, s1: SubjectFNIRS, s2: SubjectFNIRS):
        self.s1: SubjectFNIRS = s1
        self.s2: SubjectFNIRS = s2

    
    @property 
    def subjects(self):
        return [self.s1, self.s2]
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        _ = [s.load_epochs(tmin, tmax, baseline) for s in self.subjects]
        return self
