from typing import Tuple

from .subject_fnirs import SubjectFNIRS

class DyadFNIRS:
    def __init__(self, s1: SubjectFNIRS, s2: SubjectFNIRS):
        self.s1: SubjectFNIRS = s1
        self.s2: SubjectFNIRS = s2

    
    @property 
    def subjects(self):
        return [self.s1, self.s2]
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        _ = [s.load_epochs(tmin, tmax, baseline) for s in self.subjects]
        return self
