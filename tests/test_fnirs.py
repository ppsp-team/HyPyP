import pytest
import warnings
import logging
import re

import numpy as np
from numpy.testing import assert_array_almost_equal
import mne

from hypyp.fnirs.cohort_fnirs import CohortFNIRS
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
from hypyp.fnirs.subject_fnirs import SubjectFNIRS
from hypyp.fnirs.dyad_fnirs import DyadFNIRS
from hypyp.fnirs.data_loader_fnirs import DataBrowserFNIRS
from hypyp.fnirs.preprocessors.base_preprocessor_fnirs import PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_HAEMO_FILTERED_KEY
from hypyp.fnirs.preprocessors.mne_preprocessor_fnirs import MnePreprocessStep, MnePreprocessorFNIRS, DummyPreprocessorFNIRS
from hypyp.utils import TASK_NEXT_EVENT

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

#
# Data Browser
#
def test_list_paths():
    loader = DataBrowserFNIRS()
    paths = loader.paths
    assert len(paths) > 0

def test_list_paths():
    loader = DataBrowserFNIRS()
    previous_count = len(loader.paths)
    loader.add_source('/foo')
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

def test_list_files():
    loader = DataBrowserFNIRS()
    assert len(loader.paths) > 0
    assert len(loader.list_all_files()) > 0
    # path should be absolute
    assert loader.list_all_files()[0].startswith('/')


#
# Preprocessor
#

# Try to load every file types we have
@pytest.mark.parametrize("file_path", fnirs_files)
def test_data_loader_all_types(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = MnePreprocessorFNIRS().read_file(file_path)
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
    ch_names = MnePreprocessorFNIRS().get_nirs_ch_names(meas_list, lambdas)
    assert len(ch_names) == meas_list.shape[0]
    assert ch_names[0] == 'S1_D1 760'
    assert ch_names[-1] == 'S2_D1 850'
 
def test_preprocess_step():
    key = 'foo_key'
    desc = 'foo_description'
    raw = mne.io.RawArray(np.array([[1., 2.]]), mne.create_info(['foo'], 1))
    step = MnePreprocessStep(raw, key, desc)
    assert step.obj.get_data().shape[0] == 1
    assert step.obj.get_data().shape[1] == 2
    assert step.key == key
    assert step.desc == desc
    assert step.tracer is None
    assert step.duration == 2

    # With tracer
    pre_step_with_tracer = MnePreprocessStep(raw, key, desc, tracer=dict(foo=np.zeros((2,2))))
    assert pre_step_with_tracer.tracer is not None
    assert pre_step_with_tracer.tracer['foo'][0,0] == 0

#
# Subject
#

def test_subject():
    filepath = snirf_file1
    subject = SubjectFNIRS()
    subject.load_file(MnePreprocessorFNIRS(), filepath)
    assert subject.filepath == filepath
    assert subject.raw is not None
    assert len(subject.tasks) == 1 # default task, which is the complete record
    assert subject.preprocess_steps is None
    assert subject.is_preprocessed == False
    assert subject.epochs_per_task is None # need preprocessing to extract epochs

def test_subject_tasks():
    subject = SubjectFNIRS(tasks=[('my_task', 1, 2)])
    assert len(subject.tasks) == 1
    subject.task_keys[0] == 'my_task'

def test_subject_epochs():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT),
        ('task2', 3, TASK_NEXT_EVENT),
    ]
    subject = SubjectFNIRS(tasks=tasks)
    preprocessor = MnePreprocessorFNIRS()
    subject.load_file(preprocessor, snirf_file1, preprocess=True)
    subject.populate_epochs_from_annotations()
    assert len(subject.get_epochs_for_task('task1')) == 2
    n_events = subject.get_epochs_for_task('task1').events.shape[0]
    assert n_events == 2
    assert len(subject.get_epochs_for_task('task2')) == 3

def test_dummy_preprocessor():
    subject = SubjectFNIRS(tasks=[('task1', 1, TASK_NEXT_EVENT)]).load_file(MnePreprocessorFNIRS(), snirf_file1)
    subject.preprocess(DummyPreprocessorFNIRS())
    assert len(subject.preprocess_steps) == 1
    assert subject.is_preprocessed == True
    assert subject.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert subject.epochs_per_task is None
    subject.populate_epochs_from_annotations()
    assert len(subject.epochs_per_task) > 0

def test_mne_preprocessor():
    subject = SubjectFNIRS().load_file(MnePreprocessorFNIRS(), snirf_file1)
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

#
# Dyad
#

def test_subject_dyad():
    # Use the same file for the 2 subjects
    preprocessor = MnePreprocessorFNIRS()
    subject = SubjectFNIRS().load_file(preprocessor, snirf_file1)
    dyad = DyadFNIRS(subject, subject)
    assert dyad.is_preprocessed == False

    dyad.preprocess(preprocessor)
    dyad.populate_epochs_from_annotations()
    assert dyad.is_preprocessed == True

    pairs = dyad.get_pairs()
    n_channels = len(subject.pre.ch_names)
    #assert len(pairs) == n_channels * n_channels * n_epochs # TODO this is the test we with, with epochs
    assert len(pairs) == n_channels * n_channels
    assert pairs[0].label is not None
    assert pairs[0].ch_name1 == subject.pre.ch_names[0]
    assert pairs[0].ch_name2 == subject.pre.ch_names[0]

def test_dyad_tasks_intersection():
    tasks = [
        ('my_task1', 1, 2),
        ('my_task2', 3, 4),
        ('my_task3', 5, 6),
    ]
    s1 = SubjectFNIRS(tasks=[tasks[0], tasks[1]])
    s2 = SubjectFNIRS(tasks=[tasks[1], tasks[2]])
    assert len(DyadFNIRS(s1, s1).tasks) == 2
    assert len(DyadFNIRS(s2, s2).tasks) == 2
    assert len(DyadFNIRS(s1, s2).tasks) == 1
    
    
def test_dyad_compute_pair_wtc():
    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    pair = dyad.get_pairs()[0].sub((0, 10)) # Take 10% of the file
    wtc = dyad.get_pair_wtc(pair, PywaveletsWavelet())
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(wtc.wtc) == pytest.approx(1)

def test_dyad_compute_all_wtc():
    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    assert dyad.is_wtc_computed == False
    dyad.compute_wtcs(PywaveletsWavelet(), time_range=(0,5)) # TODO have a more simple wavelet for fast computing
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == len(subject.pre.ch_names)**2
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(dyad.wtcs[0].wtc) == pytest.approx(1)
    
def test_dyad_compute_str_match_wtc():
    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    dyad.compute_wtcs(PywaveletsWavelet(), match='760', time_range=(0,5))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == (len(subject.pre.ch_names)/2)**2

def test_dyad_compute_regex_match_wtc():
    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    regex = re.compile(r'^S1.*760')
    dyad.compute_wtcs(PywaveletsWavelet(), match=regex, time_range=(0,5))
    assert len(dyad.wtcs) == 4
    assert dyad.wtcs[0].label == dyad.get_pairs()[0].label

def test_dyad_compute_tuple_match_wtc():
    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    regex1 = re.compile(r'^S1_D1.*760')
    regex2 = re.compile(r'.*760')
    dyad.compute_wtcs(PywaveletsWavelet(), match=(regex1, regex2), time_range=(0,5))
    assert len(dyad.wtcs) == 16
    #[print(wtc.label) for wtc in dyad.wtcs]

def test_dyad_wtc_per_task():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT), # these 2 events have different duration
        ('task3', 3, TASK_NEXT_EVENT),
    ]
    subject = SubjectFNIRS(tasks=tasks).load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    subject.populate_epochs_from_annotations()
    dyad = DyadFNIRS(subject, subject)
    ch_name = 'S1_D1 760'
    pairs = dyad.get_pairs(match=ch_name)
    assert len(pairs) == 2
    dyad.compute_wtcs(PywaveletsWavelet(), match=ch_name, time_range=(0,5))
    assert len(dyad.wtcs) == 2
    assert dyad.wtcs[0].wtc.shape[1] != dyad.wtcs[1].wtc.shape[1] # not the same duration
    assert 'task1' in [wtc.task for wtc in dyad.wtcs] # order may have changed because of task intersection

def test_cohort_wtc():
    subject1 = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True).populate_epochs_from_annotations()
    dyad1 = DyadFNIRS(subject1, subject1)

    subject2 = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file2, preprocess=True).populate_epochs_from_annotations()
    dyad2 = DyadFNIRS(subject2, subject2)

    dyad3 = DyadFNIRS(subject1, subject2)

    # Add a bunch of "dyad3" to our list, so we have a number of "others" for our first dyad
    dyads = [dyad1, dyad2, dyad3, dyad3, dyad3, dyad3]
    cohort = CohortFNIRS(dyads)
    assert len(cohort.dyads) == len(dyads)
    assert len(cohort.dyads_shuffle) == len(dyads)*(len(dyads)-1)
    assert cohort.is_wtc_computed == False

    wtcs_kwargs = dict(match='S1_D1 760', time_range=(0,5))

    cohort.compute_wtcs(PywaveletsWavelet(), **wtcs_kwargs)
    assert cohort.is_wtc_computed == True
    assert len(dyad1.wtcs) == 1
    
    # dyads shuffle are computed only when we want significance
    assert cohort.is_wtc_shuffle_computed == False
    assert cohort.dyads_shuffle[0].wtcs is None
    assert cohort.dyads[0].wtcs[0].sig is None

    cohort.compute_wtcs_shuffle(PywaveletsWavelet(), **wtcs_kwargs)

    assert cohort.is_wtc_shuffle_computed == True
    assert len(cohort.dyads_shuffle[0].wtcs) == 1

    cohort.compute_wtcs_significance()
    assert cohort.dyads[0].wtcs[0].sig_p_value > 0

@pytest.mark.skip(reason="TODO: have significance comparison")
def test_significance():
    pass

#def test_dyad_connection_matrix():
#    subject = SubjectFNIRS().load_file(DummyPreprocessorFNIRS(), snirf_file1, preprocess=True)
#    dyad = DyadFNIRS(subject, subject)
#    dyad.compute_wtcs(PywaveletsWavelet(), match='760', time_range=(0,5))
    
    
# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    loader = DataBrowserFNIRS()
    previous_count = len(loader.paths)
    loader.download_demo_dataset()
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():



    

