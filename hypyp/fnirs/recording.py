from typing import List
from pathlib import Path

import mne
import itertools as itertools

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
)
from ..wavelet.base_wavelet import WTC

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
        self.subject_label = subject_label if subject_label != '' else generate_random_label(10)
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
    def preprocessed(self) -> mne.io.Raw:
        self._assert_is_preprocessed()
        # We want the last step of all the preprocessing
        return self.preprocess_steps[-1].obj

    @property
    def pre(self) -> mne.io.Raw:
        return self.preprocessed
    
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
            preprocessor = MnePreprocessorAsIs()

        if not Path(filepath).is_file():
            raise RuntimeError(f'Cannot find file {filepath}')

        self.filepath = filepath        
        self.mne_raw = preprocessor.read_file(filepath, verbose=verbose)
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