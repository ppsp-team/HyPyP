from typing import Tuple

import numpy as np

from .pair_signals import PairSignals
from .subject_fnirs import SubjectFNIRS
from .preprocessors.base_preprocessor_fnirs import BasePreprocessorFNIRS

class DyadFNIRS:
    def __init__(self, s1: SubjectFNIRS, s2: SubjectFNIRS):
        self.s1: SubjectFNIRS = s1
        self.s2: SubjectFNIRS = s2
    
    @property 
    def subjects(self):
        return (self.s1, self.s2)
    
    @property
    def is_preprocessed(self):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                return False
        return True

    def preprocess(self, preprocessor: BasePreprocessorFNIRS):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                subject.preprocess(preprocessor)
        return self
    
    def get_pairs(self):
        pairs = []
        s1_data = self.s1.pre.get_data()
        s2_data = self.s2.pre.get_data()

        # TODO raise exception if sfreq is not the same in both

        # TODO crop signals
        # # Crop signals
        # stop = min(len(y1), len(y2))
        # y1 = y1[:stop] 
        # y2 = y2[:stop] 

        n = s1_data.shape[1]
        x = np.linspace(0, n/self.s1.pre.info['sfreq'], n)
        for s1_i, s1_ch_name in enumerate(self.s1.pre.ch_names):
            for s2_i, s2_ch_name in enumerate(self.s2.pre.ch_names):
                # TODO check if we want info_table
                pair = PairSignals(
                    x,
                    s1_data[s1_i,:],
                    s2_data[s2_i,:],
                    ch_name1=s1_ch_name,
                    ch_name2=s2_ch_name,
                    label=f'{s1_ch_name} - {s2_ch_name}',
                )
                pairs.append(pair)
        
        return pairs


    
    