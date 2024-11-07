import pytest
import warnings
import logging

import numpy as np
import mne

from hypyp.fnirs.subject_fnirs import SubjectFNIRS
from hypyp.fnirs.dyad_fnirs import DyadFNIRS
from hypyp.fnirs.data_loader_fnirs import DataLoaderFNIRS
from hypyp.fnirs.preprocessors.base_preprocessor_fnirs import PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_HAEMO_FILTERED_KEY
from hypyp.fnirs.preprocessors.mne_preprocessor_fnirs import MnePreprocessStep, MnePreprocessorFNIRS, DummyPreprocessorFNIRS

#set events
tmin = 0 
tmax = 300
baseline = (0, 0)

ch_list_s1 = ["S4_D4 hbo"] 
ch_list_s2 = ["S7_D6 hbo"]

fif_file = './data/sub-110_session-1_pre.fif'
snirf_file1 = './data/fNIRS/DCARE_02_sub1.snirf'
snirf_file2 = './data/fNIRS/DCARE_02_sub2.snirf'
fnirs_files = [fif_file, snirf_file1, snirf_file2]

# avoid all the output from mne
logging.disable()

# Try to load every file types we have
@pytest.mark.parametrize("file_path", fnirs_files)
def test_data_loader_all_types(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = DataLoaderFNIRS().read_file(file_path)
    assert raw.info['sfreq'] > 0
    assert len(raw.ch_names) > 0
    
def test_nirs_ch_names():
    # This is the typical structure of a .nirs file (Homer2 format)
    meas_list = np.array([[1,1,1,1],
                          [1,2,1,1],
                          [2,1,1,1],
                          [1,1,1,2],
                          [1,2,1,2],
                          [2,1,1,2]])

    lambdas = np.array([760, 850]).reshape((-1,1))
    ch_names = DataLoaderFNIRS().get_nirs_ch_names(meas_list, lambdas)
    assert len(ch_names) == meas_list.shape[0]
    assert ch_names[0] == 'S1_D1 760'
    assert ch_names[-1] == 'S2_D1 850'
 
def test_preprocess_step():
    key = 'foo_key'
    desc = 'foo_description'
    raw = mne.io.RawArray(np.array([[1., 2.]]), mne.create_info(['foo'], 1))
    pre_step = MnePreprocessStep(raw, key, desc)
    assert pre_step.obj.get_data().shape[0] == 1
    assert pre_step.obj.get_data().shape[1] == 2
    assert pre_step.key == key
    assert pre_step.desc == desc
    assert pre_step.tracer is None

    # With tracer
    pre_step_with_tracer = MnePreprocessStep(raw, key, desc, tracer=dict(foo=np.zeros((2,2))))
    assert pre_step_with_tracer.tracer is not None
    assert pre_step_with_tracer.tracer['foo'][0,0] == 0

def test_subject():
    # filename does not end with _raw.fif
    # ignore the warning
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    filepath = snirf_file1
    subject = SubjectFNIRS()
    subject.load_file(DataLoaderFNIRS(), filepath)
    assert subject.filepath == filepath
    assert subject.raw is not None
    assert subject.preprocess_steps is None
    assert subject.is_preprocessed == False

def test_dummy_preprocessor():
    subject = SubjectFNIRS().load_file(DataLoaderFNIRS(), snirf_file1)
    subject.preprocess(DummyPreprocessorFNIRS())
    assert len(subject.preprocess_steps) == 1
    assert subject.is_preprocessed == True
    assert subject.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY

def test_mne_preprocessor():
    subject = SubjectFNIRS().load_file(DataLoaderFNIRS(), snirf_file1)
    subject.preprocess(MnePreprocessorFNIRS())
    assert len(subject.preprocess_steps) > 1
    assert subject.is_preprocessed == True
    assert subject.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert subject.preprocess_steps[-1].key == PREPROCESS_STEP_HAEMO_FILTERED_KEY
    assert subject.pre is not None
    assert subject.preprocess_step_keys[0] == subject.preprocess_steps[-1].key

    # can get step by key
    assert subject.get_preprocess_step(PREPROCESS_STEP_BASE_KEY).key == PREPROCESS_STEP_BASE_KEY
    assert subject.get_preprocess_step(PREPROCESS_STEP_HAEMO_FILTERED_KEY).key == PREPROCESS_STEP_HAEMO_FILTERED_KEY

def test_subject_dyad():
    # Use the same file for the 2 subjects
    subject = SubjectFNIRS().load_file(DataLoaderFNIRS(), snirf_file1)
    dyad = DyadFNIRS(subject, subject)
    assert dyad.is_preprocessed == False

    dyad.preprocess(DummyPreprocessorFNIRS())
    assert dyad.is_preprocessed == True

    pairs = dyad.get_pairs()
    n_channels = len(subject.pre.ch_names)
    assert len(pairs) == n_channels * n_channels
    assert pairs[0].label is not None
    assert pairs[0].ch_name1 == subject.pre.ch_names[0]
    assert pairs[0].ch_name2 == subject.pre.ch_names[0]
    
def test_list_paths():
    loader = DataLoaderFNIRS()
    paths = loader.paths
    assert len(paths) > 0

def test_list_paths():
    loader = DataLoaderFNIRS()
    previous_count = len(loader.paths)
    loader.add_source('/foo')
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

def test_list_files():
    loader = DataLoaderFNIRS()
    assert len(loader.paths) > 0
    assert len(loader.list_fif_files()) > 0
    # path should be absolute
    assert loader.list_fif_files()[0].startswith('/')

    
# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    loader = DataLoaderFNIRS()
    previous_count = len(loader.paths)
    loader.download_demo_dataset()
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():



    

