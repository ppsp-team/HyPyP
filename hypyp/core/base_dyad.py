from abc import ABC, abstractmethod

class BaseDyad(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_synchrony_time_series():
        pass
