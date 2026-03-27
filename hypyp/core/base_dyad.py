from abc import ABC, abstractmethod

from ..dataclasses.synchrony import SynchronyTimeSeries

class BaseDyad(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_synchrony_time_series() -> SynchronyTimeSeries:
        pass
