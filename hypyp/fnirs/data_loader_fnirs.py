from os import listdir
from os.path import isfile, join

import numpy as np
import mne
import scipy.io

from .pair_signals import PairSignals

class DataLoaderFNIRS:
    def __init__(self):
        self.paths = [
            join('data'),
            join('data', 'FNIRS'),
        ]

    def add_source(self, path):
        self.paths.append(path)

    def list_all_files(self):
        filepaths = []
        for path in self.paths:
            for filename in listdir(path):
                filepath = join(path, filename)
                if isfile(filepath):
                    filepaths.append(filepath)
        return filepaths

    def list_fif_files(self):
        return [f for f in self.list_all_files() if f.endswith('.fif')]
    
    def read_two_signals_from_mat(self, file_path, id1, id2):
        mat = scipy.io.loadmat(file_path)
        x = mat['t'].flatten().astype(np.float64, copy=True)
        y1 = mat['d'][:, id1].flatten().astype(np.complex128, copy=True)
        y2 = mat['d'][:, id2].flatten().astype(np.complex128, copy=True)

        return PairSignals(x, y1, y2)
    
