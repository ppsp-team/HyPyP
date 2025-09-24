from typing import List
from pathlib import Path
import re

import numpy as np
import mne
import itertools as itertools
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from .preprocessor.base_preprocessor import BasePreprocessor, BaseStep
from .preprocessor.implementations.mne_preprocessor_as_is import MnePreprocessorAsIs
from .channel_roi import ChannelROI
from ..utils import (
    epochs_from_tasks,
    TASK_BEGINNING,
    TASK_END,
    TASK_NAME_WHOLE_RECORD,
    Task,
    TaskList,
    generate_random_label,
    key_dict_as_data_frame,
)
from ..wavelet.cwt import CWT
from ..wavelet.wtc import WTC

class Recording:
    """
    The Recording object encapsulates the logic around the recording for one participant.
    The preprocessing is run on the channels of the Recording

    Args:
        label (str, optional): unique label for the subject. Defaults to a random string.
        tasks (TaskList, optional): list of tasks during the recording of the participant, that will be extracted from events in the raw files to build epochs. Defaults to [].
        channel_roi (ChannelROI | None, optional): region of interest object to group channels. Defaults to None.
    """
    filepath: str | None
    subject_label: str
    channel_roi: ChannelROI | None
    mne_raw: mne.io.Raw | None
    intra_wtcs: List[WTC] | None # intra-subject wtc
    epochs_per_task: List[mne.Epochs] | None
    tasks: TaskList
    preprocess_steps: List[BaseStep] | None

    def __init__(
        self,
        subject_label:str='',
        tasks:TaskList=[],
        channel_roi:ChannelROI|None=None
    ):
        self.filepath = None
        self.subject_label = subject_label
        self.channel_roi = channel_roi
        self.mne_raw = None
        self.intra_wtcs = None
        self.epochs_per_task = None
        self.preprocess_steps = None
        self.tasks = tasks

        if len(tasks) == 0:
            # Use tasks with special values instead of time_range,
            # since we don't know yet the duration of the record
            self.tasks = [Task(TASK_NAME_WHOLE_RECORD, onset_event_id=TASK_BEGINNING, offset_event_id=TASK_END)]

    def _assert_is_preprocessed(self):
        if not self.is_preprocessed:
            raise RuntimeError('Recording is not preprocessed. Did you run preprocess() ?')

    def _assert_is_epochs_loaded(self):
        if not self.is_epochs_loaded:
            raise RuntimeError('Recording does not have epochs loaded. Did you run populate_epochs_from_tasks() ?')

    def _assert_is_wtcs_computed(self):
        if not self.intra_wtcs:
            raise RuntimeError('Recording has no intra_wtcs. Did you run compute_wtcs(with_intra=True) ?')

    @staticmethod
    def get_default_preprocessor():
        return MnePreprocessorAsIs()
    
    @property
    def task_keys(self):
        return [task.name for task in self.tasks]

    @property
    def is_preprocessed(self) -> bool:
        return self.preprocess_steps is not None
    
    @property
    def is_epochs_loaded(self) -> bool:
        return self.epochs_per_task is not None
    
    @property
    def is_wtc_computed(self):
        return self.intra_wtcs is not None

    @property
    def raw(self):
        return self.mne_raw

    @property
    def preprocessed(self) -> mne.io.Raw:
        self._assert_is_preprocessed()
        # We want the last step of all the preprocessing
        return self.preprocess_steps[-1].obj
    
    @property
    def cwts(self) -> List[CWT]:
        self._assert_is_wtcs_computed()
        seen = []
        for wtc in self.intra_wtcs:
            if wtc.cwt1 not in seen:
                seen.append(wtc.cwt1)
        return seen


    @property
    def pre(self) -> mne.io.Raw:
        return self.preprocessed
    
    @property
    def mne_preprocessed(self) -> mne.io.Raw:
        return self.preprocessed

    @property
    def mne_pre(self) -> mne.io.Raw:
        return self.pre

    @property
    def ordered_ch_names(self) -> List[str]:
        if self.channel_roi is None:
            return self.preprocessed.ch_names
        return self.channel_roi.get_ch_names_in_order(self.preprocessed.ch_names)

    @property
    def ordered_roi_names(self) -> List[str]:
        if self.channel_roi is None:
            return []
        return list(self.channel_roi.rois.keys())

    @property
    def preprocess_step_keys(self):
        self._assert_is_preprocessed()
        # get in the reverse order so that the last step is first in list
        keys = []
        for i in range(len(self.preprocess_steps)-1, -1, -1):
            keys.append(self.preprocess_steps[i].key)
        return keys

    @property
    def preprocess_step_choices(self):
        self._assert_is_preprocessed()
        steps_dict = dict()
        for i in range(len(self.preprocess_steps)-1, -1, -1):
            step = self.preprocess_steps[i]
            steps_dict[step.key] = step.desc
        return steps_dict
    
    def _fill_subject_label(self):
        if self.subject_label == '':
            self.subject_label = self.mne_raw.info['subject_info']['his_id']
        
        if self.subject_label == '':
            self.subject_label = generate_random_label(10)
    
    def load_file(
        self,
        filepath:str,
        preprocessor:BasePreprocessor|None=None,
        preprocess=True,
        verbose=False
    ):
        """
        Load a raw NIRS file as the recording of the Recording

        Args:
            filepath (str): disk path of the file to load
            preprocessor (BasePreprocess, optional): which preprocessor to use. Defaults to MnePreprocessUpstream which only load the data as-is.
            preprocess (bool, optional): if the preprocessing should be done right away, or if the file should only be loaded and the preprocessing would be run later. Defaults to True.
            verbose (bool, optional): verbosity flag. Defaults to False.

        Raises:
            RuntimeError: When file is not found

        Returns:
            self: the Recording object itself. Useful for chaining operations
        """
        if preprocessor is None:
            preprocessor = Recording.get_default_preprocessor()

        if not Path(filepath).is_file():
            raise RuntimeError(f'Cannot find file {filepath}')

        self.filepath = filepath        
        self.mne_raw = preprocessor.read_file(filepath, verbose=verbose)

        self._fill_subject_label()

        if preprocess:
            self.preprocess(preprocessor, verbose=verbose)

        return self
    
    def load_raw(
        self,
        raw:mne.io.Raw,
        preprocessor:BasePreprocessor|None=None,
        preprocess=True,
        verbose=False
    ):
        """
        Use an existing Raw object to load the data into a Recording

        Args:
            raw (mne.io.Raw): preloaded mne.io.Raw object
            preprocessor (BasePreprocess, optional): which preprocessor to use. Defaults to MnePreprocessUpstream which only load the data as-is.
            preprocess (bool, optional): if the preprocessing should be done right away, or if the file should only be loaded and the preprocessing would be run later. Defaults to True.
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            self: the Recording object itself. Useful for chaining operations
        """
        if preprocessor is None:
            preprocessor = Recording.get_default_preprocessor()

        self.mne_raw = raw
        self._fill_subject_label()

        if preprocess:
            self.preprocess(preprocessor, verbose=verbose)

        return self
    
    def preprocess(self, preprocessor:BasePreprocessor, verbose:bool=False):
        """
        Run the preprocessing for the raw recording of the Recording

        Args:
            preprocessor (BasePreprocessor): which preprocessor object to use. See existing implementations or extend BasePreprocessor abstract class to develop your own
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            self: the Recording object itself. Useful for chaining operations
        """
        self.preprocess_steps = preprocessor.run(self.mne_raw, verbose=verbose)
        self.populate_epochs_from_tasks(verbose=verbose)
        return self

    def get_preprocess_step(self, key:str) -> BaseStep:
        """
        Get a specific step of the preprocessing pipeline

        Args:
            key (str): name of the step

        Returns:
            BaseStep: an implementation of BaseStep class, depending on the preprocessor implementation
        """
        self._assert_is_preprocessed()
        for step in self.preprocess_steps:
            if step.key == key:
                return step

        raise RuntimeError(f'No preprocess step named "{key}"')
    
    def populate_epochs_from_tasks(self, verbose:bool=False):
        """
        Given the list of tasks (annotations and timed) of the Recording,
        find the given data in the preprocessed channels and load as epochs that will be compared between subjects
        and store them in the object

        Args:
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            self: the Recording object itself. Useful for chaining operations
        """
        self.epochs_per_task = epochs_from_tasks(self.preprocessed, self.tasks, verbose=verbose)
        return self
    
    def get_epochs_for_task(self, task_name:str) -> List[mne.Epochs]:
        """
        Given a task name, return all the Epochs object

        Args:
            task_name (str): name of the task

        Returns:
            List[mne.Epochs]: list of the Epochs. Epochs must have been populated beforehand
        """
        self._assert_is_epochs_loaded()
        id = None
        try:
            id = [i for i in range(len(self.task_keys)) if self.task_keys[i] == task_name][0]
        except:
            pass

        if id is None:
            raise RuntimeError(f'Cannot find epochs for task "{task_name}"')

        return self.epochs_per_task[id]
    
    def get_roi_from_channel(self, ch_name:str) -> str:
        """
        Given a channel name, return the region of interest it belongs to, if any

        Args:
            ch_name (str): name of the channel

        Returns:
            str: name of the region of interest
        """
        if self.channel_roi is None:
            return ''
        return self.channel_roi.get_roi_from_channel(ch_name)
    
    def get_channel_to_standard_montage_map(self, as_data_frame:bool=True):
        my_montage = self.mne_raw.get_montage()
        my_positions_dict = my_montage.get_positions()['ch_pos']

        # See all montages with mne.channels.get_builtin_montages()
        # print(mne.channels.get_builtin_montages())
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        standard_positions_dict = standard_montage.get_positions()['ch_pos']
        standard_keys = list(standard_positions_dict.keys())
        standard_positions = list(standard_positions_dict.values())

        my_positions_lookup = {}

        # copy source detector dict
        for k in my_positions_dict.keys():
            my_positions_lookup[k] = my_positions_dict[k]

        # add the channels which are the midpoint of source and detector
        for ch_name in self.mne_raw.ch_names:
            res = re.search(r"^(S\d+)_(D\d+).*$", ch_name)
            source = res[1]
            detector = res[2]

            ch_key = f"{source}_{detector}"
            ch_pos = (my_positions_dict[source] + my_positions_dict[detector]) / 2
            my_positions_lookup[ch_key] = ch_pos

        mapped_dict = {}

        # make sure we keep the same ordering

        # Find all distances, and nearest in standard
        for k in my_positions_lookup.keys():
            ch_pos_2d = my_positions_lookup[k].reshape(1, -1)  # reshape to 2D
            
            # Get nearest standard electrode
            dists = cdist(ch_pos_2d, standard_positions)
            
            nearest_idx = np.argmin(dists)
            mapped_dict[k] = {
                'name': standard_keys[nearest_idx],
                'dist': dists[0, nearest_idx],
            }

        if as_data_frame:
            return key_dict_as_data_frame(mapped_dict, col_names=['Source/Detector', 'Standard 1020', 'Distance'])

        return mapped_dict

    def plot_steps_for_channel(self, ch_name: str, show_cwt=True):
        # get only the "S1_D1" part of "S1_D1 760"
        ch_base_name = re.sub(' .*$', '', ch_name) 
        rows = len(self.preprocess_steps)
        if show_cwt:
            rows = rows + 1

        height = rows * 3
        fig, axes = plt.subplots(rows, 1, figsize=(12, height), sharex=True, sharey=False)
        axes = np.atleast_1d(axes)

        for j, step in enumerate(self.preprocess_steps):
            raw = step.obj
            raw_data = raw.get_data()

            for k, local_ch_name in enumerate(raw.ch_names):
                if local_ch_name == f"{ch_base_name} 760":
                    axes[j].plot(raw.times, raw_data[k,:], color='g', label=local_ch_name)
                if local_ch_name == f"{ch_base_name} 850":
                    axes[j].plot(raw.times, raw_data[k,:], color='C1', label=local_ch_name)
                if local_ch_name == f"{ch_base_name} hbo":
                    axes[j].plot(raw.times, raw_data[k,:], color='r', label=local_ch_name)
                if local_ch_name == f"{ch_base_name} hbr":
                    axes[j].plot(raw.times, raw_data[k,:], color='b', label=local_ch_name)
                axes[j].legend()
                axes[j].set_title(step.desc)

        found_cwt = None
        if show_cwt:
            for cwt in self.cwts:
                cwt_base_name = re.sub(' .*$', '', cwt.label) 
                if cwt_base_name == ch_base_name:
                    found_cwt = cwt
                    break
            if found_cwt is None:
                raise ValueError(f"Cannot find CWT for ch_name '{ch_name}'")

            cwt.plot(ax=axes[rows-1], show_colorbar=False)

        fig.suptitle(f"Steps for {ch_name}")
        fig.tight_layout()

        return fig


