import pytest

import xarray as xr
import cedalion

from hypyp.fnirs.preprocessors.base_preprocessor_fnirs import PREPROCESS_STEP_BASE_KEY
from hypyp.fnirs.preprocessors.cedalion_preprocessor_fnirs import CedalionDataLoaderFNIRS, CedalionPreprocessorFNIRS

snirf_file = './data/fNIRS/DCARE_02_sub1.snirf'

def test_cedalion_preprocessor():
    loader = CedalionDataLoaderFNIRS()
    preprocessor = CedalionPreprocessorFNIRS()

    steps = preprocessor.run(loader.read_file(snirf_file))
    step = steps[0]
    assert step.key == PREPROCESS_STEP_BASE_KEY
    assert isinstance(step.obj, xr.DataArray)
    assert step.n_times == len(step.obj['time'])
    assert step.sfreq == 7.8125
    assert len(step.ch_names) > 0
    assert step.ch_names[0] == 'S1D1'
