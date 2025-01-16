from itertools import compress

import mne

from ...data_browser import DataBrowser
from ..base_preprocessor import BasePreprocessor
from ..base_step import *
from .mne_step import MneStep

class MnePreprocessorBasic(BasePreprocessor[mne.io.Raw]):
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
    
    def get_mne_channel(self, file_path, channel_name):
        s = self.read_file(file_path)
        return s.copy().pick(mne.pick_channels(s.ch_names, include = [channel_name]))
    
    def run(self, raw: mne.io.Raw, verbose: bool = False):
        steps = []
        steps.append(MneStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC))

        ## TODO: it seems that .snirf files have a different measurement unit
        #if not self.ignore_distances:
        #    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
        #    dists = mne.preprocessing.nirs.source_detector_distances(self.raw.info, picks=picks)
        #    self.raw.pick(picks[dists > 0.01])

        haemo_picks = mne.pick_types(raw.info, fnirs=['hbo', 'hbr'])

        # If we have haemo_picks, it means it is already preprocessed
        # TODO: this code flow if confusing
        if len(haemo_picks) > 0:
            raw_haemo = raw.copy().pick(haemo_picks)
            steps.append(MneStep(raw_haemo, PREPROCESS_STEP_HAEMO_FILTERED_KEY, PREPROCESS_STEP_HAEMO_FILTERED_DESC))
            return steps

        raw_od = mne.preprocessing.nirs.optical_density(raw)
        quality_sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
        raw_od.info['bads'] = list(compress(raw_od.ch_names, quality_sci < 0.1))
        steps.append(MneStep(raw_od, PREPROCESS_STEP_OD_KEY, PREPROCESS_STEP_OD_DESC))

        picks = mne.pick_types(raw_od.info, fnirs=True, exclude='bads')
        raw_od_clean = raw_od.copy().pick(picks)
        steps.append(MneStep(raw_od_clean, PREPROCESS_STEP_OD_CLEAN_KEY, PREPROCESS_STEP_OD_CLEAN_DESC))

        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od_clean, ppf=0.1)
        steps.append(MneStep(raw_haemo, PREPROCESS_STEP_HAEMO_KEY, PREPROCESS_STEP_HAEMO_DESC))
        raw_haemo_filtered = raw_haemo.copy().filter(
                                                0.05,
                                                0.7,
                                                h_trans_bandwidth=0.2,
                                                l_trans_bandwidth=0.02,
                                                verbose=verbose)
        steps.append(MneStep(raw_haemo_filtered, PREPROCESS_STEP_HAEMO_FILTERED_KEY, PREPROCESS_STEP_HAEMO_FILTERED_DESC))
        return steps
        