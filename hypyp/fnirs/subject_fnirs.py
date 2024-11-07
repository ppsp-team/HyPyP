from typing import List, Tuple
from enum import Enum

import numpy as np
import mne
import itertools as itertools

from .data_loader_fnirs import DataLoaderFNIRS
from .preprocessors.base_preprocessor_fnirs import BasePreprocessorFNIRS, BasePreprocessStep

class SubjectFNIRS:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw = None
        self.preprocess_steps: List[BasePreprocessStep] = None

    def _assert_is_preprocessed(self):
        if not self.is_preprocessed:
            raise RuntimeError('Subject is not preprocessed')

    @property
    def is_preprocessed(self) -> bool:
        return self.preprocess_steps is not None
    
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
    
    def load_file(self, loader: DataLoaderFNIRS, filepath: str):
        self.filepath = filepath        
        self.raw = loader.read_file(filepath)
        return self
    
    def preprocess(self, preprocessor: BasePreprocessorFNIRS):
        self.preprocess_steps = preprocessor.run(self.raw)
        return self

    def get_preprocess_step(self, key):
        self._assert_is_preprocessed()
        for step in self.preprocess_steps:
            if step.key == key:
                return step

        raise RuntimeError(f'No preprocess step named "{key}"')