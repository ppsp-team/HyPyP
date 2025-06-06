import mne

from ...data_browser import DataBrowser
from ..base_step import *
from ..base_preprocessor import BasePreprocessor
from .mne_step import MneStep

# This is the same as MnePreprocessorBasic, but without the default preprocessor
class MnePreprocessorAsIs(BasePreprocessor[mne.io.Raw]):
    """
    The MnePreprocessorAsIs class uses the loaded data as-is, which means that
    one should provide haemoglobin concentration already prepared in a previous
    preprocessing pipeline.    

    Use this preprocessor if you already cleaned your data and have haemoglobin concentration.

    This is the default preprocessor.
    """
    def __init__(self):
        super().__init__()
    
    def read_file(self, path:str, verbose:bool=False):
        if DataBrowser.is_path_fif(path):
            return mne.io.read_raw_fif(path, preload=True, verbose=verbose)

        if DataBrowser.is_path_nirx(path):
            return mne.io.read_raw_nirx(fname=path, preload=True, verbose=verbose)

        if DataBrowser.is_path_snirf(path):
            return mne.io.read_raw_snirf(path, preload=True, verbose=verbose)

        return None
    
    def run(self, raw: mne.io.Raw, verbose: bool = False) -> list[MneStep]:
        if verbose:
            print('Using MnePreprocessorAsIs, using raw data as already preprocessed')

        step = MneStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC)
        steps = [step]
        return steps
        
