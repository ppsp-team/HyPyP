from ..base_preprocessor import *
from .mne_preprocessor import MnePreprocessor, MnePreprocessStep

# This is the same as MnePreprocessor, but without the default pipeline
class UpstreamPreprocessor(MnePreprocessor):
    def __init__(self):
        super().__init__()
    
    def run(self, raw: mne.io.Raw, verbose: bool = False):
        # TODO honor verbose
        step = MnePreprocessStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC)
        steps = [step]
        return steps
        
