from typing import List, Tuple
from enum import Enum

import numpy as np
import mne
import itertools as itertools

from .data_loader_fnirs import DataLoaderFNIRS
from .preprocessors.base_preprocessor_fnirs import BasePreprocessorFNIRS, PreprocessStep

class SubjectFNIRS:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw = None
        self.preprocess_steps: List[PreprocessStep] = None

        self.best_ch_names: List[str] | None
        self.events: any # we should know what type this is
        self.epochs: mne.Epochs

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
        return self.preprocess_steps[-1].raw

    def load_file(self, loader: DataLoaderFNIRS, filepath: str):
        self.filepath = filepath        
        self.raw = loader.get_mne_raw(filepath)
        return self
    
    def preprocess(self, preprocessor: BasePreprocessorFNIRS):
        self.preprocess_steps = preprocessor.run(self.raw)
        return self

    def get_analysis_properties(self):
        self._assert_is_preprocessed()
        ret = dict()
        for i in range(len(self.preprocess_steps)-1, -1, -1):
            step = self.preprocess_steps[i]
            ret[step.key] = step.desc
        return ret
    
    def get_preprocess_step(self, key):
        self._assert_is_preprocessed()
        for step in self.preprocess_steps:
            if step.key == key:
                return step

        raise RuntimeError(f'No preprocess step named "{key}"')
    
    # TODO: remove this
    def set_best_ch_names(self, ch_names):
        self.best_ch_names = ch_names
        return self
    
    # TODO: check if we still want this
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        if self.raw is None:
            raise RuntimeError('Must load raw data first')

        if self.best_ch_names is None:
            raise RuntimeError('No "best channels" has been set')

        ch_picks = mne.pick_channels(self.raw.ch_names, include = self.best_ch_names)
        best_channels = self.raw.copy().pick(ch_picks)
        self.events, self.event_dict = mne.events_from_annotations(best_channels)
        self.epochs = mne.Epochs(
            best_channels,
            self.events,
            event_id = self.event_dict,
            tmin = tmin,
            tmax = tmax,
            baseline = baseline,
            reject_by_annotation=False)
        return self
