from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar

import mne

PREPROCESS_STEP_BASE_KEY = 'base'
PREPROCESS_STEP_BASE_DESC = 'Loaded data'

PREPROCESS_STEP_OD_KEY = 'od'
PREPROCESS_STEP_OD_DESC = 'Optical density'

PREPROCESS_STEP_OD_CLEAN_KEY = 'od_clean'
PREPROCESS_STEP_OD_CLEAN_DESC = 'Optical density cleaned'

PREPROCESS_STEP_HAEMO_KEY = 'haemo'
PREPROCESS_STEP_HAEMO_DESC = 'Hemoglobin'

PREPROCESS_STEP_HAEMO_FILTERED_KEY = 'haemo_filtered'
PREPROCESS_STEP_HAEMO_FILTERED_DESC = 'Hemoglobin Band-pass Filtered'

# Generic type for underlying fnirs implementation (mne raw / cedalion recording)
T = TypeVar('T')

class BasePreprocessStep(ABC, Generic[T]):
    def __init__(self, obj: T, key: str, desc: str = '', tracer: dict = None):
        self.obj: T = obj
        self.key: str = key
        self.desc: str
        if desc:
            self.desc = desc
        else:
            self.desc = key
        self.tracer: dict = tracer 

    @property
    @abstractmethod
    def n_times(self) -> int:
        pass

    @property
    @abstractmethod
    def sfreq(self) -> float:
        pass

    @property
    @abstractmethod
    def ch_names(self) -> List[str]:
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass

class BasePreprocessorFNIRS(ABC, Generic[T]):
    @abstractmethod
    def read_file(self, path: str) -> T:
        pass

    @abstractmethod
    def run(self, raw: T) -> List[BasePreprocessStep[T]]:
        pass

