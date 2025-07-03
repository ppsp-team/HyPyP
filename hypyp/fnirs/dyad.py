from typing import Tuple, List
import re
import warnings

import numpy as np
import pandas as pd

from ..wavelet.base_wavelet import BaseWavelet
from ..wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from ..wavelet.wtc import WTC
from ..wavelet.pair_signals import PairSignals
from ..wavelet.coherence_data_frame import CoherenceDataFrame
from ..utils import TaskList, TASK_NAME_WHOLE_RECORD
from .recording import Recording
from .preprocessor.base_preprocessor import BasePreprocessor
from ..plots import (
    plot_wavelet_transform_weights,
    plot_coherence_matrix,
    plot_coherence_bars_per_task,
    plot_coherence_connectogram,
    plot_coherence_connectogram_split,
)

PairChannelMatchSingleType = str | List[str] | re.Pattern
PairChannelMatchType = PairChannelMatchSingleType | Tuple[PairChannelMatchSingleType, PairChannelMatchSingleType]

MIN_SECTION_LENGTH = 10

class Dyad:
    """
    The Dyad object is a pair of recordings (per subject) of an hyperscanning recording.
    Their recorded channels should be time aligned.

    Args:
        s1 (Recording): recording of subject 1 of the dyad
        s2 (Recording): recording of subject 2 of the dyad
        label (str, optional): Custom label for the dyad. Defaults to `s1.label`-`s2.label`.
        is_pseudo (bool, optional): If the dyad is a permutated pair created for comparison. Used to track dyad "type" in results. Defaults to False.
    """
    s1: Recording
    s2: Recording
    label: str
    is_pseudo: bool
    tasks: TaskList # intersection of tasks of subject 1 and subject 2
    wtcs: List[WTC] | None # the computed Wavelet Transform Coherence for each channel pairs in the dyad
    df: CoherenceDataFrame | None # pandas dataframe from computed coherence

    def __init__(self, s1:Recording, s2:Recording, label:str='', is_pseudo:bool=False):
        self.s1 = s1
        self.s2 = s2
        self.wtcs = None
        self.df = None
        self.is_pseudo = is_pseudo

        self.label = label
        if self.label == '':
            self.label = Dyad._get_label_from_recordings(s1, s2)

        # Intersect the tasks
        self.tasks = []
        s1_tasks = s1.tasks
        s2_tasks = s2.tasks

        s2_tasks_names = [t.name for t in s2_tasks]
        found_tasks_names = []
        for task in s1_tasks:
            task_name = task.name
            if task_name in s2_tasks_names:
                self.tasks.append(task)
                found_tasks_names.append(task_name)
    
    @property 
    def recordings(self) -> Tuple[Recording, Recording]:
        return (self.s1, self.s2)
    
    @property
    def is_preprocessed(self) -> bool:
        for recording in self.recordings:
            if not recording.is_preprocessed:
                return False
        return True
    
    @property
    def is_wtc_computed(self):
        return self.wtcs is not None

    @staticmethod
    def _get_label_from_recordings(s1:Recording, s2:Recording) -> str:
        return f'{s1.subject_label}-{s2.subject_label}'

    def preprocess(self, preprocessor: BasePreprocessor):
        """
        Run the preprocess pipeline on every subject recordings in the dyad

        Args:
            preprocessor (BasePreprocessor): Which preprocessor class to use. If no preprocessing is necessary, use MnePreprocessorUpstream()

        Returns:
            self: the Dyad object itself. Useful for chaining operations
        """
        for recording in self.recordings:
            recording.preprocess(preprocessor)
        return self

    def _append_pairs(self,
                      label_dyad:str,
                      s1_ch_names:List[str],
                      s2_ch_names:List[str],
                      s1_task_data:np.ndarray,
                      s2_task_data:np.ndarray,
                      s1:Recording,
                      s2:Recording,
                      task_name:str,
                      epoch_id:int,
                      is_intra_of:int|None,
                      is_pseudo:bool,
                      pairs:List[PairSignals]):
        n = s1_task_data.shape[1]
        x = np.linspace(0, n/s1.preprocessed.info['sfreq'], n)
        for s1_i, s1_ch_name in enumerate(s1_ch_names):
            for s2_i, s2_ch_name in enumerate(s2_ch_names):
                y1 = s1_task_data[s1_i,:]
                y2 = s2_task_data[s2_i,:]
                # Crop signals in case they are not the same length
                stop = min(len(y1), len(y2))
                y1 = y1[:stop] 
                y2 = y2[:stop] 

                # Look for NaN, and split in section
                section_id = 0
                nan_mask = np.isnan(y1) | np.isnan(y2)
                has_nan = np.any(nan_mask)

                if has_nan:
                    nan_idx = np.where(nan_mask)[0]+1
                    # We have to drop the first item of every split but the first split, because it contains the NaN
                    x_sections = [section for section in np.split(x, nan_idx) if len(section)>MIN_SECTION_LENGTH]
                    y1_sections = [section for section in np.split(y1, nan_idx) if len(section)>MIN_SECTION_LENGTH]
                    y2_sections = [section for section in np.split(y2, nan_idx) if len(section)>MIN_SECTION_LENGTH]
                    
                    # remove the lurking NaN in edges of our arrays
                    for i in range(len(x_sections)):
                        delete_flags = np.isnan(y1_sections[i]) | np.isnan(y2_sections[i])
                        x_sections[i] = np.delete(x_sections[i], delete_flags)
                        y1_sections[i] = np.delete(y1_sections[i], delete_flags)
                        y2_sections[i] = np.delete(y2_sections[i], delete_flags)
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
                        label_roi1=s1.get_roi_from_channel(s1_ch_name),
                        label_roi2=s2.get_roi_from_channel(s2_ch_name),
                        label_s1=s1.subject_label,
                        label_s2=s2.subject_label,
                        label_dyad=label_dyad,
                        label_task=task_name,
                        epoch_idx=epoch_id,
                        section_idx=section_id,
                        is_intra=(is_intra_of is not None),
                        is_intra_of=is_intra_of,
                        is_pseudo=is_pseudo,
                    ))
            
    
    def get_pairs(self, s1:Recording, s2:Recording, label_dyad:str|None=None, ch_match:PairChannelMatchType|None=None, is_intra_of:int|None=None, is_pseudo:bool=False) -> List[PairSignals]:
        """
        Generate all the signal pairs between the 2 subjects and returns them in a format suitable for signal processing

        Args:
            s1 (Recording): recording of subject 1 of the dyad
            s2 (Recording): recording of subject 2 of the dyad
            label_dyad (str | None, optional): custom label for the dyad. Defaults to self.label.
            ch_match (PairMatchType, optional): string, list or regex to match channel name.
                                                Can be a tuple of 2 items if subject1 and subject2 have different matches.
                                                Defaults to None, which means all channels.
            is_pseudo (bool, optional): True if the pair is a permutated pair. Defaults to False.

        Returns:
            List[PairSignals]: a list of all the possible pairs of s1 channels with s2 channels
        """
        if label_dyad is None:
            label_dyad = self.label

        pairs: List[PairSignals] = []

        if s1.preprocessed.info['sfreq'] != s2.preprocessed.info['sfreq']:
            raise RuntimeError('Recordings must have the same sampling frequency')

        # Force match in tuple for leaner code below
        if not isinstance(ch_match, tuple):
            ch_match = (ch_match, ch_match)

        def check_match(ch_name, m):
            if m is None:
                return True
            if isinstance(m, re.Pattern):
                return m.search(ch_name) is not None
            if isinstance(m, List):
                return ch_name in m
            return m in ch_name


        s1_ch_names = [ch_name for ch_name in s1.ordered_ch_names if check_match(ch_name, ch_match[0])]
        s2_ch_names = [ch_name for ch_name in s2.ordered_ch_names if check_match(ch_name, ch_match[1])]

        seen_tasks = set()
        for task in self.tasks:
            if task.name in seen_tasks:
                continue
            seen_tasks.add(task.name)
            if task.name == TASK_NAME_WHOLE_RECORD:
                s1_task_data = s1.preprocessed.copy().pick(s1_ch_names).get_data()
                s2_task_data = s2.preprocessed.copy().pick(s2_ch_names).get_data()
                epoch_id = 0
                self._append_pairs(
                    label_dyad,
                    s1_ch_names,
                    s2_ch_names,
                    s1_task_data,
                    s2_task_data,
                    s1,
                    s2,
                    task.name,
                    epoch_id,
                    is_intra_of,
                    is_pseudo,
                    pairs,
                )
            else:
                epochs1 = s1.get_epochs_for_task(task.name).copy().pick(s1_ch_names)
                epochs2 = s2.get_epochs_for_task(task.name).copy().pick(s2_ch_names)
                if len(epochs1) != len(epochs2):
                    warnings.warn("The 2 recordings do not have the same epochs count. Some epochs will be skipped in pairs.")
                
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
                        task.name,
                        epoch_id,
                        is_intra_of,
                        is_pseudo,
                        pairs,
                    )

        return pairs
    
    def compute_wtcs(
        self,
        wavelet: BaseWavelet | None = None,
        ch_match: PairChannelMatchType | None = None,
        only_time_range: Tuple[float,float] | None = None,
        bin_seconds: float | None = None,
        period_cuts: List[float] | None = None,
        verbose: bool = False,
        with_intra: bool = True,
        downsample: int | None = None,
        keep_wtcs: bool = True,
    ):
        """
        Compute the Wavelet Transform Coherence for every channel pairs on the dyad

        Args:
            wavelet (BaseWavelet | None): the wavelet to use for the wavelet transform. Defaults to ComplexMorletWavelet() without arguments.
            ch_match (PairMatchType | None, optional): string, list or regex to match channel name.
                                                Can be a tuple of 2 items if subject1 and subject2 have different matches.
                                                Defaults to None, which means all channels.
            only_time_range (Tuple[float,float] | None, optional): compute only a portion of the signal, defined as time range tuple (start, stop). Defaults to None.
            bin_seconds (float | None, optional): split the resulting WTC in time bins for balancing weights. Defaults to None.
            period_cuts (List[float] | None, optional): split the resulting WTC in period/frequency bins for balancing weights and finer analysis. Defaults to None.
            verbose (bool, optional): verbose flag. Defaults to False.
            with_intra (bool, optional): compute intra-subject as well. Defaults to False.
            downsample (int | None, optional): downsample in time the resulting WTC. Useful to save memory space and faster display. Defaults to None.
            keep_wtcs (bool, optional): if False, all the WTCs will be removed from object after the coherence dataframe has been computed. Useful to save memory space. Defaults to True.

        Returns:
            self: the Dyad object itself. Useful for chaining operations
        """
        if wavelet is None:
            wavelet = ComplexMorletWavelet()

        self.wtcs = []

        pairs = self.get_pairs(self.s1, self.s2, ch_match=ch_match, is_pseudo=self.is_pseudo)

        for pair in pairs:
            if verbose:
                print(f'Running Wavelet Coherence for dyad "{self.label}" on pair "{pair.label}"')
            if only_time_range is not None:
                pair = pair.sub(only_time_range)
            wtc = wavelet.wtc(pair, bin_seconds=bin_seconds, period_cuts=period_cuts)
            if downsample is not None:
                wtc.downsample_in_time(downsample)

            self.wtcs.append(wtc)

        if with_intra:
            for i, recording in enumerate([self.s1, self.s2]):
                recording.intra_wtcs = []
                # if we have different channels for each recording, the intra wtc should use only the ones for this subject
                if isinstance(ch_match, tuple):
                    ch_match_intra = ch_match[i]
                else:
                    ch_match_intra = ch_match
                    
                # We have to keep track of the subject identifier when we are intra, since they would be displayed differently (parent VS child for example)
                is_intra_of = i+1
                for pair in self.get_pairs(recording, recording, f'{recording.subject_label}(intra)', ch_match=ch_match_intra, is_intra_of=is_intra_of):
                    if verbose:
                        print(f'Running Wavelet Coherence intra-subject "{recording.subject_label}" on pair "{pair.label}"')
                    if only_time_range is not None:
                        pair = pair.sub(only_time_range)
                    wtc = wavelet.wtc(pair, bin_seconds=bin_seconds, period_cuts=period_cuts)
                    if downsample is not None:
                        wtc.downsample_in_time(downsample)
                    recording.intra_wtcs.append(wtc)

        self.df = self._get_coherence_df(with_intra=with_intra)

        if not keep_wtcs:
            self.wtcs = []
            self.s1.intra_wtcs = []
            self.s1.intra_wtcs = []

        return self
    
    def _get_coherence_df(self, with_intra=False) -> pd.DataFrame:
        if with_intra:
            if not self.s1.is_wtc_computed or not self.s2.is_wtc_computed:
                raise RuntimeError('Intra subject WTCs are not computed. Please check "compute_wtcs" arguments')
            wtcs = self.wtcs + self.s1.intra_wtcs + self.s2.intra_wtcs
        else:
            wtcs = self.wtcs

        frame_rows = []
        for wtc in wtcs:
            frame_rows = frame_rows + wtc.as_frame_rows

        return CoherenceDataFrame.from_wtc_frame_rows(frame_rows)

    def reset(self):
        self.s1.intra_wtcs = None
        self.s2.intra_wtcs = None
        self.wtcs = None
        self.df = None
        return self

    #
    # Plots
    # 
    def plot_wtc(self, wtc: WTC, **kwargs):
        """
        Plot the Wavelet Transform Coherence

        Args:
            wtc (WTC): WTC object
        """
        return plot_wavelet_transform_weights(
            wtc.W,
            wtc.times,
            wtc.frequencies,
            wtc.coif,
            wtc.sfreq,
            title=wtc.label_pair,
            **kwargs)

    def plot_coherence_matrix(self, field1:str='channel1', field2:str='channel2', query:str | None=None, **kwargs):
        """
        Plot the computed coherence metric for pair of fields (channel or roi) in a matrix format

        Args:
            field1 (str): name of the field in dataframe for x axis. Defaults to 'channel1'.
            field2 (str): name of the field in dataframe for y axis. Defaults to 'channel2'.
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        df = self.df
        if query is not None:
            df = df.query(query)
            
        ch_names1 = self.s1.ordered_ch_names
        ch_names2 = self.s2.ordered_ch_names
        ordered_names = ch_names1 + [name for name in ch_names2 if name not in ch_names1] 

        return plot_coherence_matrix(
            df,
            self.s1.subject_label,
            self.s2.subject_label,
            field1,
            field2,
            ordered_names,
            **kwargs)
        
    def plot_coherence_matrix_per_channel(self, query:str|None=None, **kwargs):
        """
        Wraps plot_coherence_matrix to plot per channel

        Args:
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        return self.plot_coherence_matrix(
            'channel1',
            'channel2',
            query,
            **kwargs)
        
    def plot_coherence_matrix_per_roi(self, query:str|None=None, **kwargs):
        """
        Wraps plot_coherence_matrix to plot per region of interest

        Args:
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        return self.plot_coherence_matrix(
            'roi1',
            'roi2',
            query,
            **kwargs)
    
    def plot_coherence_matrix_per_channel_for_task(self, task:str, **kwargs):
        """
        Wraps plot_coherence_matrix_per_channel to plot for a specific task

        Args:
            task (str): task name
        """
        return self.plot_coherence_matrix(
            'channel1',
            'channel2',
            query=f'task=="{task}"',
            **kwargs)
        
    def plot_coherence_matrix_per_roi_for_task(self, task:str, **kwargs):
        """
        Wraps plot_coherence_matrix_per_roi to plot for a specific task

        Args:
            task (str): task name
        """
        return self.plot_coherence_matrix(
            'roi1',
            'roi2',
            query=f'task=="{task}"',
            **kwargs)
    
    def plot_coherence_bars_per_task(self, **kwargs):
        """
        Plot coherence metric per task for comparison
        """
        return plot_coherence_bars_per_task(
            self.df,
            **kwargs)
        
    #
    # Plot connectogram (Proof of Concept)
    # 
    def plot_coherence_connectogram_intra(self, recording:Recording, query:str|None=None, **kwargs):
        df = self.df
        selector = df['is_intra']==True
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')
        if 'title' not in kwargs:
            kwargs['title'] = recording.subject_label

        return plot_coherence_connectogram(
            pivot,
            **kwargs)

    def plot_coherence_connectogram_s1(self, query:str|None=None, **kwargs):
        return self.plot_coherence_connectogram_intra(
            self.s1,
            query,
            **kwargs)

    def plot_coherence_connectogram_s2(self, query:str|None=None, **kwargs):
        return self.plot_coherence_connectogram_intra(
            self.s2,
            query,
            **kwargs)

    def plot_coherence_connectogram(self, query:str|None=None, title:str|None=None, **kwargs):
        df = self.df.copy()
        selector = df['is_intra']==False
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        # rename to have them separated in the plot
        df_filtered.loc[:, 'roi1'] = 's1_' + df_filtered['roi1'].astype(str)
        df_filtered.loc[:, 'roi2'] = 's2_' + df_filtered['roi2'].astype(str)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')

        if title is None:
            title=f'{self.s1.subject_label} / {self.s2.subject_label}'

        return plot_coherence_connectogram_split(
            pivot,
            title=title,
            **kwargs)

