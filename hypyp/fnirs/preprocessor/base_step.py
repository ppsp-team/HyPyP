from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar

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

class BaseStep(ABC, Generic[T]):
    """
    Stores the results of a step in a preprocess pipeline

    Args:
        obj (T): the implementation dependent object representing the step processed data
        key (str): identifier for the step
        desc (str | None, optional): description of the setup. Defaults to "key" value.
    """
    obj: T
    key: str
    desc: str

    def __init__(self, obj:T, key:str, desc:str|None=None):
        self.obj = obj
        self.key = key
        if desc is None:
            self.desc = key
        else:
            self.desc = desc

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

    @property
    def duration(self) -> float:
        return self.n_times / self.sfreq

    @abstractmethod
    def plot(self, **kwargs):
        pass
