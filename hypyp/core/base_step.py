from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar

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
    name: str
    desc: str

    def __init__(self, obj:T, name:str, desc:str|None=None):
        self.obj = obj
        self.name = name
        if desc is None:
            self.desc = name
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
