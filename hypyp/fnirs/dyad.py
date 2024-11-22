from typing import Tuple, List
from collections import OrderedDict
import re

import numpy as np
import matplotlib.pyplot as plt

from ..wavelet.pywavelets_wavelet import PywaveletsWavelet
from ..wavelet.base_wavelet import WTC, BaseWavelet, downsample_in_time
from ..wavelet.pair_signals import PairSignals
from .subject import Subject, TASK_NAME_WHOLE_RECORD
from .preprocessors.base_preprocessor import BasePreprocessor
from ..plots import plot_coherence_matrix, plot_wavelet_coherence

PairMatch = re.Pattern|str|Tuple[re.Pattern|str,re.Pattern|str]

class Dyad:
    def __init__(self, s1: Subject, s2: Subject, label:str=''):
        self.s1: Subject = s1
        self.s2: Subject = s2
        self.wtcs: List[WTC] = None
        self.label = label

        # Intersect the tasks
        self.tasks = []
        s1_tasks = s1.tasks_annotations + s1.tasks_time_range
        s2_tasks = s2.tasks_annotations + s2.tasks_time_range

        for task in s1_tasks:
            if task in s2_tasks:
                self.tasks.append(task)
    
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
                subject.populate_epochs_from_tasks()
        return self
    
    def populate_epochs_from_tasks(self, **kwargs):
        for subject in self.subjects:
            subject.populate_epochs_from_tasks(**kwargs)
        return self

    def get_pairs(self, s1: Subject, s2: Subject, match:PairMatch=None) -> List[PairSignals]:
        pairs = []

        # TODO raise exception if sfreq is not the same in both

        # Force match in tuple for leaner code below
        if not isinstance(match, Tuple):
            match = (match, match)

        def check_match(ch_name, m):
            if m is None:
                return True
            if isinstance(m, re.Pattern):
                return m.search(ch_name) is not None
            return m in ch_name


        ch_names1 = [ch_name for ch_name in s1.ordered_ch_names if check_match(ch_name, match[0])]
        ch_names2 = [ch_name for ch_name in s2.ordered_ch_names if check_match(ch_name, match[1])]

        for task_name, _, _ in self.tasks:
            if task_name == TASK_NAME_WHOLE_RECORD:
                # TODO see if copy() slows down our computation or takes memory
                s1_task_data = s1.pre.copy().pick(ch_names1).get_data()
                s2_task_data = s2.pre.copy().pick(ch_names2).get_data()
            else:
                epochs1 = s1.get_epochs_for_task(task_name).copy().pick(ch_names1)
                epochs2 = s2.get_epochs_for_task(task_name).copy().pick(ch_names2)
                # TODO here we take only the first epoch per task. Should we take more?
                s1_task_data = epochs1.get_data(copy=False)[0,:,:]
                s2_task_data = epochs2.get_data(copy=False)[0,:,:]

            n = s1_task_data.shape[1]
            x = np.linspace(0, n/s1.pre.info['sfreq'], n)

            for s1_i, s1_ch_name in enumerate(ch_names1):
                for s2_i, s2_ch_name in enumerate(ch_names2):
                    y1 = s1_task_data[s1_i,:]
                    y2 = s2_task_data[s2_i,:]
                    # Crop signals in case they are not the same length
                    stop = min(len(y1), len(y2))
                    y1 = y1[:stop] 
                    y2 = y2[:stop] 

                    # TODO check if we want info_table
                    pairs.append(PairSignals(
                        x,
                        y1,
                        y2,
                        ch_name1=s1_ch_name,
                        ch_name2=s2_ch_name,
                        label_s1=s1.label,
                        label_s2=s2.label,
                        task=task_name,
                    ))

        return pairs
    
    def get_pair_wtc(self, pair: PairSignals, wavelet: BaseWavelet, cache_suffix='') -> WTC: 
        return wavelet.wtc(pair, cache_suffix)
    
    # TODO remove "time_range", this is only for testing
    def compute_wtcs(
        self,
        wavelet:BaseWavelet=PywaveletsWavelet(),
        match:PairMatch=None,
        time_range:Tuple[float,float]=None,
        verbose=False,
        intra_subject=False,
        downsample=None
    ):
        self.wtcs = []

        for pair in self.get_pairs(self.s1, self.s2, match=match):
            if verbose:
                print(f'Running Wavelet Coherence for dyad "{self.label}" on pair "{pair.label}"')
            if time_range is not None:
                pair = pair.sub(time_range)
            wtc = self.get_pair_wtc(pair, wavelet, cache_suffix='dyad')
            if downsample is not None:
                wtc.downsample_in_time(downsample)

            self.wtcs.append(wtc)

        if intra_subject:
            self.s1.wtcs = []
            self.s2.wtcs = []
            # TODO see if we are computing more than once
            #if not self.s1.is_wtc_computed:
            for subject in [self.s1, self.s2]:
                for pair in self.get_pairs(subject, subject, match=match):
                    if verbose:
                        print(f'Running Wavelet Coherence intra-subject "{subject.label}" on pair "{pair.label}"')
                    if time_range is not None:
                        pair = pair.sub(time_range)
                    wtc = self.get_pair_wtc(pair, wavelet, cache_suffix='intra')
                    if downsample is not None:
                        wtc.downsample_in_time(downsample)
                    subject.wtcs.append(wtc)

        return self
    
    # TODO Maybe having wtcs as optional (and default to self.wtcs) is confusing
    def get_wtc_property_matrix(self, property_name: str, wtcs:List[WTC]=None, reverse=False):
        if wtcs is None:
            wtcs = self.wtcs

        task_map = OrderedDict()
        row_map = OrderedDict()
        col_map = OrderedDict()

        tasks = sorted(list(set([wtc.task for wtc in wtcs])))
        ch_names1 = sorted(list(set([wtc.ch_name1 for wtc in wtcs])))
        ch_names2 = sorted(list(set([wtc.ch_name2 for wtc in wtcs])))

        if self.s1.channel_roi is not None:
            ch_names1 = self.s1.channel_roi.get_names_in_order(ch_names1)
        if self.s2.channel_roi is not None:
            ch_names2 = self.s2.channel_roi.get_names_in_order(ch_names2)

        for i, task in enumerate(tasks):
            task_map[task] = i

        for j, ch_name1 in enumerate(ch_names1):
            row_map[ch_name1] = j

        for k, ch_name2 in enumerate(ch_names2):
            col_map[ch_name2] = k

        mat = np.zeros((len(tasks), len(ch_names1), len(ch_names2)))

        for wtc in wtcs:
            i = task_map[wtc.task]
            j = row_map[wtc.ch_name1]
            k = col_map[wtc.ch_name2]
            if reverse:
                mat[i,k,j] = getattr(wtc, property_name)
            else:
                mat[i,j,k] = getattr(wtc, property_name)

        return mat, tasks, ch_names1, ch_names2
    
    def get_coherence_matrix(self):
        return self.get_wtc_property_matrix('coherence_metric')

    def get_coherence_matrix_with_intra(self):
        # TODO this is crappy. We should use grouping and facets in display instead
        dyadic = self.get_wtc_property_matrix('coherence_metric')
        dyadic_T = self.get_wtc_property_matrix('coherence_metric', reverse=True)
        s1 = self.get_wtc_property_matrix('coherence_metric', self.s1.wtcs)
        s2 = self.get_wtc_property_matrix('coherence_metric', self.s2.wtcs)

        if dyadic[1] != s1[1] or dyadic[1] != s2[1] :
            raise RuntimeError('Tasks list do not match, cannot concatenate matrices')

        left = np.concatenate((s1[0], dyadic_T[0]), axis=1)
        right = np.concatenate((dyadic[0], s2[0]), axis=1)
        mat = np.concatenate((left, right), axis=2)

        # for intra-subject ch_names1==ch_names2, so it does not matter which one we take
        ch_names1 = s1[2] + dyadic_T[2]
        ch_names2 = dyadic[3] + s2[3]

        return mat, dyadic[1], ch_names1, ch_names2



    def get_p_value_matrix(self):
        return self.get_wtc_property_matrix('coherence_p_value')
    
    #
    # Plots
    # 
    def plot_coherence_matrices(self):
        mat, tasks, ch_names1, ch_names2 = self.get_coherence_matrix()
        fig, axes = plt.subplots(ncols=len(tasks), sharex=True, sharey=True, figsize=(10, 6))
        axes = np.atleast_1d(axes)
        for i, task in enumerate(tasks):
            plot_coherence_matrix(mat[i,:,:], ch_names1, ch_names2, self.s1.label, self.s2.label, title=task, ax=axes[i])
        fig.suptitle(self.label)

    def plot_coherence_matrix_for_task(self, task_name):
        # TODO we should not load the whole matrix if we only want one task
        mat, tasks, ch_names1, ch_names2 = self.get_coherence_matrix_with_intra()
        # TODO deal with id not found
        id = [i for i, task in enumerate(self.tasks) if task[0] == task_name][0]
        plot_coherence_matrix(mat[id,:,:], ch_names1, ch_names2, self.s1.label, self.s2.label, title=self.tasks[id][0])

    def plot_wtc(self, wtc: WTC):
        plot_wavelet_coherence(wtc.wtc, wtc.times, wtc.frequencies, wtc.coif, wtc.sig, downsample=True, title=wtc.label)

    def plot_wtc_by_id(self, id: int):
        wtc = self.wtcs[id]
        plot_wavelet_coherence(wtc.wtc, wtc.times, wtc.frequencies, wtc.coif, wtc.sig, downsample=True, title=wtc.label)

        