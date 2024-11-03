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
            DOWNLOADS_RELATIVE_PATH,
        ]
    
    def absolute_path(self, relative_path):
        return os.path.join(self.absolute_root_path, relative_path)

    def add_source(self, path):
        self.paths.append(path)

    def path_is_nirx(self, path):
        return os.path.isfile(path) and path.endswith('.hdr')

    def path_is_nirs(self, path):
        return os.path.isfile(path) and path.endswith('.nirs')

    def path_is_fif(self, path):
        return os.path.isfile(path) and path.endswith('.fif')

    def path_is_snirf(self, path):
        return os.path.isfile(path) and path.endswith('.snirf')

    def list_all_files(self):
        file_paths = []
        for root_path in self.paths:
            for path in Path(root_path).rglob('*'):
                if self.path_is_fif(str(path)):
                    file_paths.append(str(path.absolute()))

                elif self.path_is_nirs(str(path)):
                    file_paths.append(str(path.absolute()))

                elif self.path_is_nirx(str(path)):
                    file_paths.append(str(path.absolute()))

                elif self.path_is_snirf(str(path)):
                    file_paths.append(str(path.absolute()))

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
    
    # TODO should receive mne object
    def list_channels_for_file(self, path):
        if self.path_is_fif(path):
            return mne.io.read_raw_fif(path).ch_names

        if self.path_is_nirs(path):
            raise Exception('Not implemented yet')

        if self.path_is_nirx(path):
            return mne.io.read_raw_nirx(fname=path).ch_names

        if self.path_is_snirf(path):
            raise Exception('Not implemented yet')

        return []
    
    def get_nirs_ch_names(self, meas_list, lambdas):
        ret = []
        for s, d, _, lmbda in meas_list:
            ret.append(f"S{s}_D{d} {lambdas[lmbda-1][0]}")
        return ret
            
    def get_info_nirs(self, x, mat):
        sfreq = 1 / (x[1] - x[0])
        n_channels = mat['d'].shape[1]
        info = mne.create_info(
            self.get_nirs_ch_names(mat['SD'][0,0][0], mat['SD'][0,0][1]),
            sfreq,
            ch_types=['fnirs_cw_amplitude']*n_channels)
        return info
        
    
    def get_mne_raw(self, path):
        if self.path_is_fif(path):
            return mne.io.read_raw_fif(path, preload=True)

        if self.path_is_nirs(path):
            mat = scipy.io.loadmat(path)
            x = mat['t'].flatten().astype(np.float64, copy=True)
            n_channels = mat['d'].shape[1]
            data = np.zeros((n_channels, len(x)))
            for i in range(n_channels):
                data[i, :] = mat['d'][:, i].flatten().astype(np.complex128, copy=True)

            info = self.get_info_nirs(x, mat)
            return mne.io.RawArray(data, info)

        if self.path_is_nirx(path):
            return mne.io.read_raw_nirx(fname=path, preload=True)

        if self.path_is_snirf(path):
            return mne.io.read_raw_snirf(path, preload=True)

        return None
    
    def get_mne_channel(self, file_path, channel_name):
        s = self.get_mne_raw(file_path)
        return s.copy().pick(mne.pick_channels(s.ch_names, include = [channel_name]))
    
    def read_two_signals_from_mat_obj(self, mat, id1, id2):
        x = mat['t'].flatten().astype(np.float64, copy=True)
        y1 = mat['d'][:, id1].flatten().astype(np.complex128, copy=True)
        y2 = mat['d'][:, id2].flatten().astype(np.complex128, copy=True)

        return PairSignals(x, y1, y2)
    
    def read_two_signals_from_mat(self, file_path, id1, id2):
        return self.read_two_signals_from_mat_obj(scipy.io.loadmat(file_path))
    
