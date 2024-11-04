import mne

from .base_preprocessor_fnirs import *

class DummyPreprocessorFNIRS(BasePreprocessorFNIRS):
    def __init__(self):
        super().__init__()
    
    def run(self, raw: mne.io.Raw):
        return [
            PreprocessStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC)
        ]
        
