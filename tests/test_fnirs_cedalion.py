import pytest

import xarray as xr
import cedalion

from hypyp.fnirs.preprocessors.base_preprocessor import PREPROCESS_STEP_BASE_KEY
from hypyp.fnirs.preprocessors.cedalion_preprocessor import CedalionPreprocessor

snirf_file = './data/fNIRS/DCARE_02_sub1.snirf'

def test_cedalion_preprocessor():
    if CedalionPreprocessor is None:
        pytest.skip("Optional dependency Cedalion is not installed")
        
    preprocessor = CedalionPreprocessor()

    steps = preprocessor.run(preprocessor.read_file(snirf_file))
    step = steps[0]
    assert step.key == PREPROCESS_STEP_BASE_KEY
    assert isinstance(step.obj, xr.DataArray)
    assert step.n_times == len(step.obj['time'])
    assert step.sfreq == 7.8125
    assert len(step.ch_names) > 0
    assert step.ch_names[0] == 'S1D1'
