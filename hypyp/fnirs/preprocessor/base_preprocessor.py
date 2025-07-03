from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar

from .base_step import BaseStep

# Generic type for underlying fnirs implementation (mne raw / cedalion recording)
T = TypeVar('T')

class BasePreprocessor(ABC, Generic[T]):
    @abstractmethod
    def read_file(self, path:str, verbose:bool=False) -> T:
        """
        Load the file to be preprocessed. Does not run the preprocess yet

        Args:
            path (str): NIRS file location
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            T: returns an implementation dependant object representation of the loaded data
        """
        pass

    @abstractmethod
    def run(self, raw:T, verbose:bool=False):
        """
        run all the preprocessing steps on a raw object

        Args:
            raw (T): implementation dependant object representation of the loaded data. Use with "read_file()"
            verbose (bool, optional): verbosity flag. Defaults to False.

        Returns:
            List[BasePreprocessStep[T]]: list of all the intermediary steps of the preprocessing
        """
        pass

