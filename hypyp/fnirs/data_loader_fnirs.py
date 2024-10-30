import os
from pathlib import Path
from zipfile import ZipFile

import pooch
import numpy as np
import mne
import scipy.io

from .pair_signals import PairSignals

DOWNLOADS_RELATIVE_PATH = os.path.join('data', 'fNIRS', 'downloads')


class DataLoaderFNIRS:
    def __init__(self):
        self.absolute_root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.paths = [
            self.absolute_path(os.path.join('data')),
            self.absolute_path(os.path.join('data', 'fNIRS')),
        ]
    
    def absolute_path(self, relative_path):
        return os.path.join(self.absolute_root_path, relative_path)

    def add_source(self, path):
        self.paths.append(path)

    def list_all_files(self):
        file_paths = []
        for path in self.paths:
            for p in Path(path).rglob('*'):
                if os.path.isdir(p):
                    file_paths.append(str(p.absolute()))
                elif str(p).endswith('.fif'):
                    file_paths.append(str(p.absolute()))

        # remove duplicates
        unique = list(set(file_paths))
        unique.sort()
        return unique

    def list_fif_files(self):
        return [f for f in self.list_all_files() if f.endswith('.fif')]
    
    def download_demo_dataset(self):
        extract_path = self.absolute_path(DOWNLOADS_RELATIVE_PATH)
        zip_path = pooch.retrieve(
            fname="fathers.zip",
            url="https://researchdata.ntu.edu.sg/api/access/datafile/91950?gbrecs=true",
            known_hash="md5:786e0c13caab4fc744b93070999dff63",
            progressbar=True
        )

        target_path = os.path.join(extract_path, 'fathers')

        if not os.path.exists(target_path):
            with ZipFile(zip_path, 'r') as zip:
                print(f'Extracting to {extract_path}, (target: {target_path})')
                zip.extractall(path=extract_path)

        self.add_source(target_path)
        return target_path
    
    def list_channels_for_file(self, file_path):
        # use the same file for both
        if file_path.endswith('_raw.fif'):
            return mne.io.read_raw_fif(file_path).ch_names
        return []
    
    def get_mne_channel(self, file_path, channel_name):
        s = mne.io.read_raw_fif(file_path, verbose=True)
        return s.copy().pick(mne.pick_channels(s.ch_names, include = [channel_name]))
    
    def read_two_signals_from_mat(self, file_path, id1, id2):
        mat = scipy.io.loadmat(file_path)
        x = mat['t'].flatten().astype(np.float64, copy=True)
        y1 = mat['d'][:, id1].flatten().astype(np.complex128, copy=True)
        y2 = mat['d'][:, id2].flatten().astype(np.complex128, copy=True)

        return PairSignals(x, y1, y2)
    
