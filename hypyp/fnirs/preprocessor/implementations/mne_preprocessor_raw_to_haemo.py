from itertools import compress

import mne

from ..base_step import *
from .mne_step import MneStep
from .mne_preprocessor_as_is import MnePreprocessorAsIs

class MnePreprocessorRawToHaemo(MnePreprocessorAsIs):
    """
    The MnePreprocessorRawToHaemo class uses mne fNIRS features to run some basic preprocessing steps.
    
    It does these:

    1. Store the raw data as a step
    2. Convert raw data to optical density
    3. Remove bad channels scalp_coupling_index
    4. Convert optical density to haemoglobin concentration.
    5. Filters the haemoglobin concentration based on standard high pass and low pass filters

    The steps can then be inspected on a recording, for validation.

    Use this preprocessor to explore raw data.

    NOTE: If the data loaded with MNE contains channels with `hbo` or `hbr`, it will be considered
    already preprocessed and will be returned as-is.

    """
    def __init__(self):
        super().__init__()
    
    def run(self, raw:mne.io.Raw, verbose:bool=False) -> list[MneStep]:
        if verbose:
            print('Using MnePreprocessorRawToHaemo, converting fNIRS raw data to haemoglobin concentrations')

        steps = []
        steps.append(MneStep(raw, PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_BASE_DESC))

        haemo_picks = mne.pick_types(raw.info, fnirs=['hbo', 'hbr'])

        # If we have haemo_picks, it means it is already preprocessed
        # TODO: this code flow if confusing
        if len(haemo_picks) > 0:
            if verbose:
                print('Data loaded using MNE seems to already be converted to haemoglobin concentrations')
                
            steps.append(MneStep(
                raw.copy().pick(haemo_picks),
                PREPROCESS_STEP_HAEMO_FILTERED_KEY,
                PREPROCESS_STEP_HAEMO_FILTERED_DESC
            ))
            return steps

        raw_od = mne.preprocessing.nirs.optical_density(raw)
        steps.append(MneStep(raw_od, PREPROCESS_STEP_OD_KEY, PREPROCESS_STEP_OD_DESC))

        quality_sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
        raw_od.info['bads'] = list(compress(raw_od.ch_names, quality_sci < 0.1))
        picks = mne.pick_types(raw_od.info, fnirs=True, exclude='bads')
        raw_od_clean = raw_od.copy().pick(picks)
        steps.append(MneStep(raw_od_clean, PREPROCESS_STEP_OD_CLEAN_KEY, PREPROCESS_STEP_OD_CLEAN_DESC))

        # TODO: should set Partial Pathlength Factor (PPF) depending on age.
        # See https://doi.org/10.1117/1.JBO.18.10.105004, Table 1
        raw_haemo: mne.io.Raw = mne.preprocessing.nirs.beer_lambert_law(raw_od_clean)
        steps.append(MneStep(raw_haemo, PREPROCESS_STEP_HAEMO_KEY, PREPROCESS_STEP_HAEMO_DESC))
        raw_haemo_filtered = raw_haemo.copy().filter(0.02, 0.7, verbose=verbose)

        steps.append(MneStep(raw_haemo_filtered, PREPROCESS_STEP_HAEMO_FILTERED_KEY, PREPROCESS_STEP_HAEMO_FILTERED_DESC))

        return steps
        