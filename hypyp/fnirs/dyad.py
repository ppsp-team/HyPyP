from typing import Tuple, List, Self
import re
import warnings

import numpy as np
import pandas as pd

from ..wavelet.base_wavelet import BaseWavelet
from ..wavelet.wavelet_implementations.pywavelets_wavelet import PywaveletsWavelet
from ..wavelet.wtc import WTC
from ..wavelet.pair_signals import PairSignals
from ..wavelet.coherence_data_frame import CoherenceDataFrame
from ..utils import TaskList, TASK_NAME_WHOLE_RECORD
from .subject import Subject
from .base_preprocessor import BasePreprocessor
from ..plots import (
    plot_wtc,
    plot_coherence_matrix,
    plot_coherence_bars_per_task,
    plot_coherence_connectogram
)

PairMatch = re.Pattern|str|Tuple[re.Pattern|str,re.Pattern|str]

class Dyad:
    s1: Subject
    s2: Subject
    wtcs: List[WTC]
    df: CoherenceDataFrame|None
    is_shuffle: bool
    label: str
    tasks: TaskList

    def __init__(self, s1: Subject, s2: Subject, label:str='', is_shuffle:bool=False):
        """
        The Dyad object is a pair of subjects of an hyperscanning recording.
        Their recorded channels should be time aligned.

        Args:
            s1 (Subject): subject 1 of the dyad
            s2 (Subject): subject 2 of the dyad
            label (str, optional): Custom label for the dyad. Defaults to `s1.label`-`s2.label`.
            is_shuffle (bool, optional): If the dyad is a permutated pair created for comparison. Defaults to False.
        """
        self.s1 = s1
        self.s2 = s2
        self.wtcs = None
        self.df = None
        self.is_shuffle = is_shuffle

        self.label = label
        if self.label == '':
            self.label = Dyad.get_label_from_subjects(s1, s2)

        # Intersect the tasks
        self.tasks = []
        s1_tasks = s1.tasks_annotations + s1.tasks_time_range
        s2_tasks = s2.tasks_annotations + s2.tasks_time_range

        for task in s1_tasks:
            if task in s2_tasks:
                self.tasks.append(task)
    
    @property 
    def subjects(self) -> Tuple[Subject, Subject]:
        return (self.s1, self.s2)
    
    @property
    def is_preprocessed(self) -> bool:
        for subject in self.subjects:
            if not subject.is_preprocessed:
                return False
        return True
    
    @property
    def is_wtc_computed(self):
        return self.wtcs is not None

    @staticmethod
    def get_label_from_subjects(s1: Subject, s2: Subject) -> str:
        return f'{s1.label}-{s2.label}'

    def preprocess(self, preprocessor: BasePreprocessor) -> Self:
        """
        Run the preprocess pipeline on every subject in the dyad

        Args:
            preprocessor (BasePreprocessor): Which preprocessor to use. If no preprocessing is necessary, use UpstreamPreprocessor()

        Returns:
            Self: the object itself. Useful for chaining operations
        """
        for subject in self.subjects:
            subject.preprocess(preprocessor)
        return self

    def get_pair_wtc(
        self,
        pair: PairSignals,
        wavelet: BaseWavelet,
        bin_seconds: float | None = None,
        period_cuts: List[float] | None = None,
        cache_suffix='',
    ) -> WTC: 
        """
        Compute the Wavelet Transform Coherence for a signal pair

        Args:
            pair (PairSignals): pair of signals from 2 subjects channels
            wavelet (BaseWavelet): the wavelet to use for the wavelet transform
            bin_seconds (float | None, optional): split the resulting WTC in time bins for balancing weights. Defaults to None.
            period_cuts (List[float] | None, optional): split the resulting WTC in period/frequency bins for balancing weights and finer analysis. Defaults to None.
            cache_suffix (str, optional): string to add to the caching key. Useful to split intra and inter subject. Defaults to ''.

        Returns:
            WTC: Wavelet Transform Coherence object resulting from the computation
        """
        return wavelet.wtc(pair, bin_seconds=bin_seconds, period_cuts=period_cuts, cache_suffix=cache_suffix)
    
    def _append_pairs(self,
                      label_dyad,
                      s1_ch_names,
                      s2_ch_names,
                      s1_task_data,
                      s2_task_data,
                      s1,
                      s2,
                      task_name,
                      epoch_id,
                      is_shuffle,
                      pairs):
        n = s1_task_data.shape[1]
        x = np.linspace(0, n/s1.pre.info['sfreq'], n)
        for s1_i, s1_ch_name in enumerate(s1_ch_names):
            for s2_i, s2_ch_name in enumerate(s2_ch_names):
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
                    pairs.append(PairSignals(
                        x_sections[section_id],
                        y1_sections[section_id],
                        y2_sections[section_id],
                        label_ch1=s1_ch_name,
                        label_ch2=s2_ch_name,
                        label_s1=s1.label,
                        label_s2=s2.label,
                        label_roi1=s1.get_roi_from_channel(s1_ch_name),
                        label_roi2=s2.get_roi_from_channel(s2_ch_name),
                        label_dyad=label_dyad,
                        task=task_name,
                        epoch=epoch_id,
                        section=section_id,
                        is_intra=(s1==s2),
                        is_shuffle=is_shuffle,
                    ))
            
    
    def get_pairs(self, s1: Subject, s2: Subject, label_dyad:str=None, ch_match:PairMatch=None, is_shuffle:bool=False) -> List[PairSignals]:
        """_summary_

        Args:
            s1 (Subject): subject 1 of the dyad
            s2 (Subject): subject 2 of the dyad
            label_dyad (str, optional): custom label for the dyad. Defaults to self.label.
            ch_match (PairMatch, optional): string or regex to match channel name. Defaults to None, which means all channels.
            is_shuffle (bool, optional): True if the pair is a permutated pair. Defaults to False.

        Returns:
            List[PairSignals]: a list of all the possible pairs of subject1 channels with subject2 channels
        """
        if label_dyad is None:
            label_dyad = self.label

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


        s1_ch_names = [ch_name for ch_name in s1.ordered_ch_names if check_match(ch_name, ch_match[0])]
        s2_ch_names = [ch_name for ch_name in s2.ordered_ch_names if check_match(ch_name, ch_match[1])]

        seen_tasks = set()
        for task_name, _, _ in self.tasks:
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            if task_name == TASK_NAME_WHOLE_RECORD:
                # TODO see if copy() slows down our computation or takes memory
                s1_task_data = s1.pre.copy().pick(s1_ch_names).get_data()
                s2_task_data = s2.pre.copy().pick(s2_ch_names).get_data()
                epoch_id = 0
                self._append_pairs(
                    label_dyad,
                    s1_ch_names,
                    s2_ch_names,
                    s1_task_data,
                    s2_task_data,
                    s1,
                    s2,
                    task_name,
                    epoch_id,
                    is_shuffle,
                    pairs,
                )
            else:
                epochs1 = s1.get_epochs_for_task(task_name).copy().pick(s1_ch_names)
                epochs2 = s2.get_epochs_for_task(task_name).copy().pick(s2_ch_names)
                if len(epochs1) != len(epochs2):
                    warnings.warn("The 2 subjects do not have the same epochs count. Some epochs will be skipped in pairs.")
                
                # add one pair for each epoch
                for i in range(min(len(epochs1), len(epochs2))):
                    s1_task_data = epochs1.get_data(copy=False)[i,:,:]
                    s2_task_data = epochs2.get_data(copy=False)[i,:,:]
                    epoch_id = i
                    self._append_pairs(
                        label_dyad,
                        s1_ch_names,
                        s2_ch_names,
                        s1_task_data,
                        s2_task_data,
                        s1,
                        s2,
                        task_name,
                        epoch_id,
                        is_shuffle,
                        pairs,
                    )

        return pairs
    
    def compute_wtcs(
        self,
        wavelet:BaseWavelet|None=None,
        ch_match:PairMatch=None,
        only_time_range:Tuple[float,float]=None,
        bin_seconds: float | None = None,
        period_cuts: List[float] | None = None,
        verbose=False,
        with_intra=False,
        downsample=None,
        keep_wtcs=True,
    ) -> Self:
        """
        Compute the Wavelet Transform Coherence for every channel pairs on the dyad

        Args:
            wavelet (BaseWavelet): the wavelet to use for the wavelet transform
            ch_match (PairMatch, optional): string or regex to match channel name. Defaults to None, which means all channels.
            only_time_range (Tuple[float,float], optional): _description_. Defaults to None.
            bin_seconds (float | None, optional): split the resulting WTC in time bins for balancing weights. Defaults to None.
            period_cuts (List[float] | None, optional): split the resulting WTC in period/frequency bins for balancing weights and finer analysis. Defaults to None.
            verbose (bool, optional): verbose flag. Defaults to False.
            with_intra (bool, optional): compute intra-subject as well. Defaults to False.
            downsample (_type_, optional): downsample in time the resulting WTC. Useful to save memory space and faster display. Defaults to None.
            keep_wtcs (bool, optional): if False, all the WTCs will be removed from object after the coherence dataframe has been computed. Useful to save memory space. Defaults to True.

        Returns:
            Self: the object itself. Useful for chaining operations
        """
        if wavelet is None:
            wavelet = PywaveletsWavelet()

        self.wtcs = []

        pairs = self.get_pairs(self.s1, self.s2, ch_match=ch_match, is_shuffle=self.is_shuffle)

        for pair in pairs:
            if verbose:
                print(f'Running Wavelet Coherence for dyad "{self.label}" on pair "{pair.label}"')
            if only_time_range is not None:
                pair = pair.sub(only_time_range)
            wtc = self.get_pair_wtc(pair, wavelet, bin_seconds=bin_seconds, period_cuts=period_cuts, cache_suffix='dyad')
            if downsample is not None:
                wtc.downsample_in_time(downsample)

            self.wtcs.append(wtc)

        # TODO should test this "is_shuffle" condition
        # TODO check if we are already is_intra (same subject) to avoid computing again
        if with_intra and not self.is_shuffle:
            self.s1.intra_wtcs = []
            self.s2.intra_wtcs = []
            # TODO see if we are computing more than once
            #if not self.s1.is_wtc_computed:
            for subject in [self.s1, self.s2]:
                for pair in self.get_pairs(subject, subject, f'{subject.label}(intra)', ch_match=ch_match):
                    if verbose:
                        print(f'Running Wavelet Coherence intra-subject "{subject.label}" on pair "{pair.label}"')
                    if only_time_range is not None:
                        pair = pair.sub(only_time_range)
                    wtc = self.get_pair_wtc(pair, wavelet, bin_seconds=bin_seconds, period_cuts=period_cuts, cache_suffix='intra')
                    if downsample is not None:
                        wtc.downsample_in_time(downsample)
                    subject.intra_wtcs.append(wtc)

        self.df = self._get_coherence_df(with_intra=with_intra)

        if not keep_wtcs:
            self.wtcs = []
            self.s1.intra_wtcs = []
            self.s1.intra_wtcs = []

        return self
    
    def _get_coherence_df(self, with_intra=False) -> pd.DataFrame:
        # TODO test this "is_shuffle" condition
        if with_intra and not self.is_shuffle:
            if not self.s1.is_wtc_computed or not self.s2.is_wtc_computed:
                raise RuntimeError('Intra subject WTCs are not computed. Please check "compute_wtcs" arguments')
            wtcs = self.wtcs + self.s1.intra_wtcs + self.s2.intra_wtcs
        else:
            wtcs = self.wtcs

        frame_rows = []
        for wtc in wtcs:
            frame_rows = frame_rows + wtc.as_frame_rows

        return CoherenceDataFrame.from_wtcs(frame_rows)

    
    #
    # Plots
    # 
    def plot_wtc(self, wtc: WTC, ax=None):
        return plot_wtc(wtc.wtc, wtc.times, wtc.frequencies, wtc.coi, wtc.sfreq, title=wtc.label_pair, ax=ax)

    def plot_wtc_by_id(self, id: int):
        wtc = self.wtcs[id]
        return plot_wtc(wtc.wtc, wtc.times, wtc.frequencies, wtc.coi, wtc.sfreq, title=wtc.label_pair)

    def plot_coherence_matrix(self, field1, field2, query=None):
        df = self.df
        if query is not None:
            df = df.query(query)
            
        return plot_coherence_matrix(df,
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
        return plot_coherence_bars_per_task(self.df, is_intra=is_intra)
        
    def plot_coherence_connectogram_intra(self, subject, query=None):
        df = self.df
        selector = (df['subject1']==subject.label) & (df['subject2']==subject.label)
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')
        return plot_coherence_connectogram(pivot, title=subject.label)

    def plot_coherence_connectogram_s1(self, query=None):
        return self.plot_coherence_connectogram_intra(self.s1, query)

    def plot_coherence_connectogram_s2(self, query=None):
        return self.plot_coherence_connectogram_intra(self.s2, query)