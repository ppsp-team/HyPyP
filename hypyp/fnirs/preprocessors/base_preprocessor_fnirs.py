from abc import ABC, abstractmethod
from typing import List

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

class PreprocessStep:
    def __init__(self, raw: mne.io.Raw, key: str, desc: str = '', tracer: dict = None):
        self.raw = raw
        self.key = key
        if desc:
            self.desc = desc
        else:
            self.desc = key
        self.tracer = tracer 

class BasePreprocessorFNIRS(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, raw: mne.io.Raw) -> List[PreprocessStep]:
        pass
