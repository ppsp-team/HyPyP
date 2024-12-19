from typing import Tuple, List
from collections import OrderedDict
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..wavelet.pywavelets_wavelet import PywaveletsWavelet
from ..wavelet.base_wavelet import BaseWavelet
from ..wavelet.wtc import WTC, FRAME_COLUMNS
from ..wavelet.pair_signals import PairSignals
from .subject import Subject, TASK_NAME_WHOLE_RECORD
from .preprocessors.base_preprocessor import BasePreprocessor
from ..plots import plot_coherence_matrix, plot_wtc, plot_coherence_matrix_df, plot_coherence_per_task_bars, plot_connectogram
from ..profiling import TimeTracker

PairMatch = re.Pattern|str|Tuple[re.Pattern|str,re.Pattern|str]

class Dyad:
    def __init__(self, s1: Subject, s2: Subject, label:str=''):
        self.s1: Subject = s1
        self.s2: Subject = s2
        self.wtcs: List[WTC] = None
        self.df: pd.DataFrame = None

        if label == '':
            self.label = Dyad.get_label(s1, s2)
        else:
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

    @staticmethod
    def get_label(s1: Subject, s2: Subject):
        return f'{s1.label}-{s2.label}'

    def preprocess(self, preprocessor: BasePreprocessor):
        for subject in self.subjects:
            if not subject.is_preprocessed:
                subject.preprocess(preprocessor)
        return self

    def _append_pairs(self,
                      ch_names1,
                      ch_names2,
                      s1_task_data,
                      s2_task_data,
                      s1,
                      s2,
                      task_name,
                      epoch_id,
                      pairs):
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

                # TODO too much code for this. This is inefficient, we can do better
                # Look for NaN, and split in section
                section_id = 0
                nan_mask1 = np.isnan(y1)
                nan_mask2 = np.isnan(y2)
                nan_mask = nan_mask1 | nan_mask2
                has_nan = np.any(nan_mask)
                # TODO we should have a bigger min_length, and it should not be hardcoded here
                min_length = 10
                if has_nan:
                    nan_idx = np.where(nan_mask)[0]+1
                    # We have to drop the first item of every split but the first split, because it contains the NaN
                    x_sections = [item for item in np.split(x, nan_idx) if len(item)>min_length]
                    y1_sections = [item for item in np.split(y1, nan_idx) if len(item)>min_length]
                    y2_sections = [item for item in np.split(y2, nan_idx) if len(item)>min_length]
                    # remove the lurking NaN in edges of our arrayes
                    for i in range(len(x_sections)):
                        delete = np.isnan(y1_sections[i]) | np.isnan(y2_sections[i])
                        x_sections[i] = np.delete(x_sections[i], delete)
                        y1_sections[i] = np.delete(y1_sections[i], delete)
                        y2_sections[i] = np.delete(y2_sections[i], delete)
                else:
                    x_sections = [x]
                    y1_sections = [y1]
                    y2_sections = [y2]

                for section_id in range(len(x_sections)):
                    # TODO check if we want info_table
                    pairs.append(PairSignals(
                        x_sections[section_id],
                        y1_sections[section_id],
                        y2_sections[section_id],
                        ch_name1=s1_ch_name,
                        ch_name2=s2_ch_name,
                        label_s1=s1.label,
                        label_s2=s2.label,
                        roi1=s1.get_roi_from_channel(s1_ch_name),
                        roi2=s2.get_roi_from_channel(s2_ch_name),
                        label_dyad=Dyad.get_label(s1, s2),
                        task=task_name,
                        epoch=epoch_id,
                        section=section_id,
                    ))
            
    
    def get_pairs(self, s1: Subject, s2: Subject, ch_match:PairMatch=None) -> List[PairSignals]:
        pairs = []

        # TODO raise exception if sfreq is not the same in both

        # Force match in tuple for leaner code below
        if not isinstance(ch_match, Tuple):
            ch_match = (ch_match, ch_match)

        def check_match(ch_name, m):
            if m is None:
                return True
            if isinstance(m, re.Pattern):
                return m.search(ch_name) is not None
            return m in ch_name


        ch_names1 = [ch_name for ch_name in s1.ordered_ch_names if check_match(ch_name, ch_match[0])]
        ch_names2 = [ch_name for ch_name in s2.ordered_ch_names if check_match(ch_name, ch_match[1])]


        seen_tasks = set()
        for task_name, _, _ in self.tasks:
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            if task_name == TASK_NAME_WHOLE_RECORD:
                # TODO see if copy() slows down our computation or takes memory
                s1_task_data = s1.pre.copy().pick(ch_names1).get_data()
                s2_task_data = s2.pre.copy().pick(ch_names2).get_data()
                epoch_id = 0
                self._append_pairs(ch_names1, ch_names2, s1_task_data, s2_task_data, s1, s2, task_name, epoch_id, pairs)
            else:
                epochs1 = s1.get_epochs_for_task(task_name).copy().pick(ch_names1)
                epochs2 = s2.get_epochs_for_task(task_name).copy().pick(ch_names2)
                if len(epochs1) != len(epochs2):
                    warnings.warn("The 2 subjects do not have the same epochs count. Some epochs will be skipped in pairs.")
                
                # add one pair for each epoch
                for i in range(min(len(epochs1), len(epochs2))):
                    s1_task_data = epochs1.get_data(copy=False)[i,:,:]
                    s2_task_data = epochs2.get_data(copy=False)[i,:,:]
                    epoch_id = i
                    self._append_pairs(ch_names1, ch_names2, s1_task_data, s2_task_data, s1, s2, task_name, epoch_id, pairs)

        return pairs
    
    def get_pair_wtc(self, pair: PairSignals, wavelet: BaseWavelet, cache_suffix='') -> WTC: 
        return wavelet.wtc(pair, cache_suffix)
    
    # TODO remove "time_range", this is only for testing
    def compute_wtcs(
        self,
        wavelet:BaseWavelet|None=None,
        ch_match:PairMatch=None,
        time_range:Tuple[float,float]=None,
        verbose=False,
        with_intra=False,
        downsample=None,
        keep_wtcs=True,
    ):
        if wavelet is None:
            wavelet = PywaveletsWavelet()

        self.wtcs = []

        pairs = self.get_pairs(self.s1, self.s2, ch_match=ch_match)

        for pair in pairs:
            if verbose:
                print(f'Running Wavelet Coherence for dyad "{self.label}" on pair "{pair.label}"')
            if time_range is not None:
                pair = pair.sub(time_range)
            wtc = self.get_pair_wtc(pair, wavelet, cache_suffix='dyad')
            if downsample is not None:
                wtc.downsample_in_time(downsample)

            self.wtcs.append(wtc)

        if with_intra:
            self.s1.wtcs = []
            self.s2.wtcs = []
            # TODO see if we are computing more than once
            #if not self.s1.is_wtc_computed:
            for subject in [self.s1, self.s2]:
                for pair in self.get_pairs(subject, subject, ch_match=ch_match):
                    if verbose:
                        print(f'Running Wavelet Coherence intra-subject "{subject.label}" on pair "{pair.label}"')
                    if time_range is not None:
                        pair = pair.sub(time_range)
                    wtc = self.get_pair_wtc(pair, wavelet, cache_suffix='intra')
                    if downsample is not None:
                        wtc.downsample_in_time(downsample)
                    subject.wtcs.append(wtc)

        self.df = self._get_coherence_df(with_intra=with_intra)

        if not keep_wtcs:
            self.wtcs = []
            self.s1.wtcs = []
            self.s1.wtcs = []

        return self
    
    def _get_coherence_df(self, with_intra=False) -> pd.DataFrame:
        if with_intra:
            if not self.s1.is_wtc_computed or not self.s2.is_wtc_computed:
                raise RuntimeError('Intra subject WTCs are not computed. Please check "compute_wtcs" arguments')
            wtcs = self.wtcs + self.s1.wtcs + self.s2.wtcs
        else:
            wtcs = self.wtcs

        df_data = []

        for wtc in wtcs:
            df_data.append(wtc.as_frame_row)

        return pd.DataFrame(df_data, columns=FRAME_COLUMNS)

    
    #
    # Plots
    # 
    def plot_wtc(self, wtc: WTC):
        return plot_wtc(wtc.wtc, wtc.times, wtc.frequencies, wtc.coi, wtc.sfreq, wtc.sig, title=wtc.label)

    def plot_wtc_by_id(self, id: int):
        wtc = self.wtcs[id]
        return plot_wtc(wtc.wtc, wtc.times, wtc.frequencies, wtc.coi, wtc.sfreq, wtc.sig, title=wtc.label)

    def plot_coherence_matrix(self, field1, field2, query=None):
        df = self.df
        if query is not None:
            df = df.query(query)
        return plot_coherence_matrix_df(df,
            self.s1.label,
            self.s2.label,
            field1,
            field2,
            self.s1.ordered_ch_names)
        
    def plot_coherence_matrix_per_channel(self, query=None):
        return self.plot_coherence_matrix('channel1', 'channel2', query)
        
    def plot_coherence_matrix_per_roi(self, query=None):
        return self.plot_coherence_matrix('roi1', 'roi2', query)
    
    def plot_coherence_matrix_per_channel_for_task(self, task):
        return self.plot_coherence_matrix('channel1', 'channel2', query=f'task=="{task}"')
        
    def plot_coherence_matrix_per_roi_for_task(self, task):
        return self.plot_coherence_matrix('roi1', 'roi2', query=f'task=="{task}"')
    
    def plot_coherence_bars_per_task(self, is_intra=False):
        return plot_coherence_per_task_bars(self.df, is_intra=is_intra)
        
    def plot_coherence_connectogram_intra(self, subject, query=None):
        df = self.df
        selector = (df['subject1']==subject.label) & (df['subject2']==subject.label)
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')
        return plot_connectogram(pivot, title=subject.label)

    def plot_coherence_connectogram_s1(self, query=None):
        return self.plot_coherence_connectogram_intra(self.s1, query)

    def plot_coherence_connectogram_s2(self, query=None):
        return self.plot_coherence_connectogram_intra(self.s2, query)