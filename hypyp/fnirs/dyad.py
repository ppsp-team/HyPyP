from typing import Tuple, List
from collections import OrderedDict
import re

import numpy as np

from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

from ..wavelet.base_wavelet import WTC, BaseWavelet
from ..wavelet.pair_signals import PairSignals
from .subject import Subject, TASK_NAME_WHOLE_RECORD
from .preprocessors.base_preprocessor import BasePreprocessor

PairMatch = re.Pattern|str|Tuple[re.Pattern|str,re.Pattern|str]

class Dyad:
    def __init__(self, s1: Subject, s2: Subject):
        self.s1: Subject = s1
        self.s2: Subject = s2
        self.wtcs: List[WTC] = None
        self.pairs: List[PairSignals] = None
        self.tasks = list(set(s1.tasks_annotations) & set(s2.tasks_annotations))
    
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

    def preprocess(self, preprocessor: BasePreprocessor):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                subject.preprocess(preprocessor)
        return self
    
    def populate_epochs_from_tasks(self, **kwargs):
        for subject in self.subjects:
            subject.populate_epochs_from_tasks(**kwargs)
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
            if task[0] == TASK_NAME_WHOLE_RECORD:
                s1_data = self.s1.pre.copy().pick(ch_names1).get_data()
                s2_data = self.s2.pre.copy().pick(ch_names2).get_data()
            else:
                epochs1 = self.s1.get_epochs_for_task(task[0]).copy().pick(ch_names1)
                epochs2 = self.s2.get_epochs_for_task(task[0]).copy().pick(ch_names2)
                # TODO here we take only the first epoch per task. Should we take more?
                s1_data = epochs1.get_data(copy=False)[0,:,:]
                s2_data = epochs2.get_data(copy=False)[0,:,:]

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
        return wavelet.wtc(pair)
    
    # TODO remove "time_range", this is only for testing
    def compute_wtcs(
        self,
        wavelet:BaseWavelet=PywaveletsWavelet(),
        match:PairMatch=None,
        time_range:Tuple[float,float]=None,
        verbose=False,
    ):
        self.wtcs = []
        self.pairs = []

        for pair in self.get_pairs(match=match):
            if verbose:
                print('Running Wavelet Coherence on pair: ', pair.label)
            if time_range is not None:
                pair = pair.sub(time_range)
            self.wtcs.append(self.get_pair_wtc(pair, wavelet))
            self.pairs.append(pair)
        return self
    
    def get_wtc_property_matrix(self, property_name: str):
        task_map = OrderedDict()
        row_map = OrderedDict()
        col_map = OrderedDict()

        tasks = sorted(list(set([wtc.task for wtc in self.wtcs])))
        ch_names1 = sorted(list(set([wtc.ch_name1 for wtc in self.wtcs])))
        ch_names2 = sorted(list(set([wtc.ch_name2 for wtc in self.wtcs])))

        for i, task in enumerate(tasks):
            task_map[task] = i

        for j, ch_name1 in enumerate(ch_names1):
            row_map[ch_name1] = j

        for k, ch_name2 in enumerate(ch_names2):
            col_map[ch_name2] = k

        mat = np.zeros((len(tasks), len(ch_names1), len(ch_names2)))

        for wtc in self.wtcs:
            i = task_map[wtc.task]
            j = row_map[wtc.ch_name1]
            k = col_map[wtc.ch_name2]
            mat[i,j,k] = getattr(wtc, property_name)

        return mat, tasks, ch_names1, ch_names2
    
    def get_connection_matrix(self):
        return self.get_wtc_property_matrix('sig_metric')

    def get_p_value_matrix(self):
        return self.get_wtc_property_matrix('sig_p_value')