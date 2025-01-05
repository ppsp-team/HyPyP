from typing import List
import pickle

import numpy as np
from scipy.stats import ttest_1samp
import pandas as pd

from hypyp.fnirs.base_preprocessor import BasePreprocessor
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame


from .dyad import Dyad

class Cohort():
    def __init__(self, dyads: List[Dyad] = []):
        self.dyads: List[Dyad] = dyads
        self.dyads_shuffle: List[Dyad]|None = None

    @property
    def is_wtc_computed(self):
        for dyad in self.dyads:
            if not dyad.is_wtc_computed:
                return False
        return True

    @property
    def is_wtc_shuffle_computed(self):
        if self.dyads_shuffle is None:
            return False
        for dyad in self.dyads_shuffle:
            if not dyad.is_wtc_computed:
                return False
        return True
    
    @property
    def df(self):
        return self.get_coherence_df()

    def preprocess(self, preprocessor: BasePreprocessor):
        for dyad in self.dyads:
            if not dyad.is_preprocessed:
                dyad.preprocess(preprocessor)
        return self
    
    def compute_wtcs(self, *args, **kwargs):
        for dyad in self.dyads:
            dyad.compute_wtcs(*args, **kwargs)
        
        return self
    
    def clear_dyads_shuffle(self):
        self.dyads_shuffle = None
    
    def get_dyads_shuffle(self) -> List[Dyad]:
        dyads_shuffle = []
        for i, dyad1 in enumerate(self.dyads):
            for j, dyad2 in enumerate(self.dyads):
                if i == j:
                    continue
                dyads_shuffle.append(Dyad(dyad1.s1, dyad2.s2, label=f'shuffle s1:{dyad1.label}-s2:{dyad2.label}', is_shuffle=True))
        return dyads_shuffle

    # TODO add as argument the number of shuffle dyads
    def compute_wtcs_shuffle(self, *args, **kwargs):
        self.dyads_shuffle = self.get_dyads_shuffle()
        for dyad_shuffle in self.dyads_shuffle:
            dyad_shuffle.compute_wtcs(*args, **kwargs)
        return self
    
    def get_coherence_df(self) -> CoherenceDataFrame:
        dfs = []
        if not self.is_wtc_computed:
            raise RuntimeError('wtc not computed')

        for dyad in self.dyads:
            dfs.append(dyad.df)

        if self.is_wtc_shuffle_computed:
            for dyad_shuffle in self.dyads_shuffle:
                dfs.append(dyad_shuffle.df)

        return CoherenceDataFrame.concat(dfs)
    
    #
    # Disk serialisation
    #
    @staticmethod
    def from_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, file_path):
        # Serialize the object to the temporary file
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def save_feather(self, file_path):
        CoherenceDataFrame.save_feather(self.df, file_path)

    def save_csv(self, file_path):
        self.df.to_csv(file_path)
