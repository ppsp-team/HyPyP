from typing import Tuple, List
import re

import numpy as np

from ..wavelet.base_wavelet import WTC, BaseWavelet
from .pair_signals import PairSignals
from .subject_fnirs import SubjectFNIRS
from .preprocessors.base_preprocessor_fnirs import BasePreprocessorFNIRS

PairMatch = re.Pattern|str|Tuple[re.Pattern|str,re.Pattern|str]

class DyadFNIRS:
    def __init__(self, s1: SubjectFNIRS, s2: SubjectFNIRS):
        self.s1: SubjectFNIRS = s1
        self.s2: SubjectFNIRS = s2
        self.wtcs: List[WTC] = None
        self.tasks = list(set(s1.tasks) & set(s2.tasks))
    
    @property 
    def subjects(self):
        return (self.s1, self.s2)
    
    @property
    def is_preprocessed(self):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                return False
        return True
    
    @property
    def is_wtc_computed(self):
        return self.wtcs is not None

    def preprocess(self, preprocessor: BasePreprocessorFNIRS):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                subject.preprocess(preprocessor)
        return self
    
    def populate_epochs_from_annotations(self, **kwargs):
        for subject in self.subjects:
            subject.populate_epochs_from_annotations(**kwargs)
        return self

    def get_pairs(self, match:PairMatch=None) -> List[PairSignals]:
        pairs = []

        # TODO raise exception if sfreq is not the same in both

        # TODO crop signals
        # # Crop signals
        # stop = min(len(y1), len(y2))
        # y1 = y1[:stop] 
        # y2 = y2[:stop] 

        # Force match in tuple for leaner code below
        if not isinstance(match, Tuple):
            match = (match, match)

        def check_match(ch_name, m):
            if m is None:
                return True
            if isinstance(m, re.Pattern):
                return m.search(ch_name) is not None
            return m in ch_name


        ch_names1 = [ch_name for ch_name in self.s1.pre.ch_names if check_match(ch_name, match[0])]
        ch_names2 = [ch_name for ch_name in self.s2.pre.ch_names if check_match(ch_name, match[1])]

        for task in self.tasks:
            epochs1 = self.s1.get_epochs_for_task(task[0]).copy().pick(ch_names1)
            epochs2 = self.s2.get_epochs_for_task(task[0]).copy().pick(ch_names2)
            # TODO here we take only the first epoch per task. Should we take more
            s1_data = epochs1.get_data()[0,:,:]
            s2_data = epochs2.get_data()[0,:,:]

            n = s1_data.shape[1]
            x = np.linspace(0, n/self.s1.pre.info['sfreq'], n)

            for s1_i, s1_ch_name in enumerate(ch_names1):
                for s2_i, s2_ch_name in enumerate(ch_names2):
                    # TODO check if we want info_table
                    pairs.append(PairSignals(
                        x,
                        s1_data[s1_i,:],
                        s2_data[s2_i,:],
                        ch_name1=s1_ch_name,
                        ch_name2=s2_ch_name,
                        task=task[0],
                    ))

        return pairs
    
    def get_pair_wtc(self, pair: PairSignals, wavelet: BaseWavelet) -> WTC: 
        return wavelet.wtc(pair.y1, pair.y2, pair.dt, task=pair.task, label=pair.label)
    
    # TODO remove "time_range", this is only for testing
    def compute_wtcs(
        self,
        wavelet: BaseWavelet,
        match:PairMatch=None,
        time_range:Tuple[float,float]=None,
        verbose=False,
    ):
        self.wtcs = []

        for pair in self.get_pairs(match=match):
            if verbose:
                print('Running Wavelet Coherence on pair: ', pair.label)
            if time_range is not None:
                pair = pair.sub(time_range)
            self.wtcs.append(self.get_pair_wtc(pair, wavelet))
        return self
    