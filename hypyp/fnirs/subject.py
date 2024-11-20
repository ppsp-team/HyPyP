from typing import List
from pathlib import Path

import mne
import itertools as itertools

from .preprocessors.base_preprocessor import BasePreprocessor, BasePreprocessStep
from ..utils import epochs_from_tasks_annotations, TASK_BEGINNING, TASK_END, Task, TaskList, epochs_from_tasks_time_range

TASK_NAME_WHOLE_RECORD = 'whole_record'

class Subject:
    def __init__(self, label:str='', tasks_annotations:TaskList=[], tasks_time_range:TaskList=[]):
        self.filepath: str = None
        self.label = label
        self.raw: mne.io.Raw = None
        self.epochs_per_task: List[mne.Epochs] = None
        self.preprocess_steps: List[BasePreprocessStep] = None
        self.tasks_annotations: TaskList = []
        self.tasks_time_range: TaskList = []

        if len(tasks_annotations) > 0:
            for task in tasks_annotations:
                assert isinstance(task[0], str)
                assert len(task) == 3
            self.tasks_annotations: TaskList = tasks_annotations

        if len(tasks_time_range) > 0:
            for task in tasks_time_range:
                assert isinstance(task[0], str)
                assert len(task) == 3
            self.tasks_time_range: TaskList = tasks_time_range

        if len(tasks_annotations) == 0 and len(tasks_time_range) == 0:
            # TODO this should use tasks_time_range
            self.tasks_annotations = [(TASK_NAME_WHOLE_RECORD, TASK_BEGINNING, TASK_END)]

    def _assert_is_preprocessed(self):
        if not self.is_preprocessed:
            raise RuntimeError('Subject is not preprocessed. Did you run preprocess() ?')

    def _assert_is_epochs_loaded(self):
        if not self.is_epochs_loaded:
            raise RuntimeError('Subject does not have epochs loaded. Did you run populate_epochs_from_tasks() ?')

    @property
    def task_keys(self):
        return [task[0] for task in self.tasks_annotations + self.tasks_time_range]

    @property
    def is_preprocessed(self) -> bool:
        return self.preprocess_steps is not None
    
    @property
    def is_epochs_loaded(self) -> bool:
        return self.epochs_per_task is not None
    
    @property
    def pre(self) -> mne.io.Raw:
        self._assert_is_preprocessed()
        # We want the last step of all the preprocessing
        return self.preprocess_steps[-1].obj

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
    
    def load_file(self, preprocessor: BasePreprocessor, filepath: str, preprocess=True):
        if not Path(filepath).is_file():
            raise RuntimeError(f'Cannot find file {filepath}')

        self.filepath = filepath        
        self.raw = preprocessor.read_file(filepath)
        if preprocess:
            self.preprocess(preprocessor)
        return self
    
    def preprocess(self, preprocessor: BasePreprocessor):
        self.preprocess_steps = preprocessor.run(self.raw)
        return self

    def get_preprocess_step(self, key):
        self._assert_is_preprocessed()
        for step in self.preprocess_steps:
            if step.key == key:
                return step

        raise RuntimeError(f'No preprocess step named "{key}"')
    
    def populate_epochs_from_tasks(self):
        self.epochs_per_task = []
        if len(self.tasks_annotations) > 0:
            self.epochs_per_task = self.epochs_per_task + epochs_from_tasks_annotations(self.pre, self.tasks_annotations)
        if len(self.tasks_time_range) > 0:
            self.epochs_per_task = self.epochs_per_task + epochs_from_tasks_time_range(self.pre, self.tasks_time_range)
        return self
    
    def get_epochs_for_task(self, task_name: str):
        self._assert_is_epochs_loaded()
        id = None
        try:
            id = [i for i in range(len(self.task_keys)) if self.task_keys[i] == task_name][0]
        except:
            pass

        if id is None:
            raise RuntimeError(f'Cannot find epochs for task "{task_name}"')

        return self.epochs_per_task[id]
        
#    def set_event_ids(self, foo):
#        pass