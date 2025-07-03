from typing import List
import os
from pathlib import Path
from zipfile import ZipFile

import pooch
import numpy as np
import scipy.io

from ..wavelet.pair_signals import PairSignals

DOWNLOADS_RELATIVE_PATH = os.path.join('data', 'NIRS', 'downloads')

class DataBrowser:
    absolute_root_path: str
    paths: List[str]

    def __init__(self):
        """
        The DataBrowser class allows to recursively list NIRS files in folders and detect file type

        It can also download demo dataset for convenience.
        """
        self.absolute_root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.paths = [
            self.absolute_path(os.path.join('data')),
            self.absolute_path(os.path.join('data', 'fNIRS')),
            DOWNLOADS_RELATIVE_PATH,
        ]
    
    def absolute_path(self, relative_path:str) -> str:
        """
        Get absolute path from relative path

        Args:
            relative_path (str): a path relative to the project

        Returns:
            str: the absolute path on disk
        """
        return os.path.join(self.absolute_root_path, relative_path)

    def add_source(self, path:str):
        """
        Add a folder to look for NIRS files

        Args:
            path (str): absolute path to add

        Returns:
            self: the DataBrowser object itself. Useful for chaining operations
        """
        self.paths.append(path)
        return self

    @staticmethod
    def is_path_nirx(path:str) -> bool:
        return os.path.isfile(path) and path.endswith('.hdr')

    @staticmethod
    def is_path_fif(path:str) -> bool:
        return os.path.isfile(path) and path.endswith('.fif')

    @staticmethod
    def is_path_snirf(path:str) -> bool:
        return os.path.isfile(path) and path.endswith('.snirf')

    def list_all_files(self) -> List[str]:
        """
        Recursively list all the NIRS files found in the source paths

        Returns:
            List[str]: list of absolute file paths
        """
        file_paths = []
        for root_path in self.paths:
            for path in Path(root_path).rglob('*'):
                if DataBrowser.is_path_fif(str(path)):
                    file_paths.append(str(path.absolute()))

                elif DataBrowser.is_path_nirx(str(path)):
                    file_paths.append(str(path.absolute()))

                elif DataBrowser.is_path_snirf(str(path)):
                    file_paths.append(str(path.absolute()))

        # remove duplicates
        unique = list(set(file_paths))
        unique.sort()
        return unique

    def download_demo_dataset(self) -> str:
        """
        Download a publicly available demo NIRS dataset of dyads recordings

        Source: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/35DNCW

        Returns:
            str: the local path where the dataset has been downloaded
        """
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
    
