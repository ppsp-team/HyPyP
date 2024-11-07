from itertools import compress

import mne

from .base_preprocessor_fnirs import *

class MnePreprocessStep(BasePreprocessStep[mne.io.Raw]):
    @property
    def n_times(self):
        return self.obj.n_times
        
    @property
    def sfreq(self):
        return self.obj.info['sfreq']
        
    @property
    def ch_names(self):
        return self.obj.ch_names
        
    def plot(self, **kwargs):
        return self.obj.plot(**kwargs)

class MnePreprocessorFNIRS(BasePreprocessorFNIRS[mne.io.Raw]):
    def __init__(self):
        super().__init__()
    
    def run(self, raw: mne.io.Raw):
        steps = []
        steps.append(MnePreprocessStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC))

        ## TODO: it seems that .snirf files have a different measurement unit
        #if not self.ignore_distances:
        #    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
        #    dists = mne.preprocessing.nirs.source_detector_distances(self.raw.info, picks=picks)
        #    self.raw.pick(picks[dists > 0.01])

        haemo_picks = mne.pick_types(raw.info, fnirs=['hbo', 'hbr'])

        # If we have haemo_picks, it means it is already preprocessed
        # TODO: this code flow if confusing
        if len(haemo_picks) > 0:
            raw_haemo = self.raw.copy().pick(haemo_picks)
            steps.append(MnePreprocessStep(raw_haemo, PREPROCESS_STEP_HAEMO_FILTERED_KEY, PREPROCESS_STEP_HAEMO_DESC))
            return steps

        raw_od = mne.preprocessing.nirs.optical_density(raw)
        quality_sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
        raw_od.info['bads'] = list(compress(raw_od.ch_names, quality_sci < 0.1))

        steps.append(MnePreprocessStep(
                        raw_od, 
                        PREPROCESS_STEP_OD_KEY,
                        PREPROCESS_STEP_OD_DESC,
                        tracer=dict(quality_sci=quality_sci)))

        picks = mne.pick_types(raw_od.info, fnirs=True, exclude='bads')

        raw_od_clean = raw_od.copy().pick(picks)
        steps.append(MnePreprocessStep(
                        raw_od_clean, 
                        PREPROCESS_STEP_OD_CLEAN_KEY,
                        PREPROCESS_STEP_OD_CLEAN_DESC,
                        tracer=dict(quality_sci=quality_sci)))

        # TODO: see if we want to expose parameters here
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od_clean, ppf=0.1)
        steps.append(MnePreprocessStep(raw_haemo, PREPROCESS_STEP_HAEMO_KEY, PREPROCESS_STEP_HAEMO_DESC))

        # TODO: have these parameters exposed?
        raw_haemo_filtered = raw_haemo.copy().filter(
                                                0.05,
                                                0.7,
                                                h_trans_bandwidth=0.2,
                                                l_trans_bandwidth=0.02)
        steps.append(MnePreprocessStep(
            raw_haemo_filtered,
            PREPROCESS_STEP_HAEMO_FILTERED_KEY,
            PREPROCESS_STEP_HAEMO_FILTERED_DESC))

        return steps
        

# This is the same as MnePreprocessor, but without the default pipeline
class DummyPreprocessorFNIRS(MnePreprocessorFNIRS):
    def __init__(self):
        super().__init__()
    
    def run(self, raw: mne.io.Raw):
        return [
            MnePreprocessStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC)
        ]
        