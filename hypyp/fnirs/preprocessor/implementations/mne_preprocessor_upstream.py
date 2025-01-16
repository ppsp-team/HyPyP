import mne

from ..base_step import *
from .mne_preprocessor_basic import MnePreprocessorBasic
from .mne_step import MneStep

# This is the same as MnePreprocessorBasic, but without the default preprocessor
class MnePreprocessorUpstream(MnePreprocessorBasic):
    def __init__(self):
        super().__init__()
    
    def run(self, raw: mne.io.Raw, verbose: bool = False):
        # TODO honor verbose
        step = MneStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC)
        steps = [step]
        return steps
        
