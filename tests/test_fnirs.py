import pytest
from unittest.mock import patch
import warnings
import logging
import re

import numpy as np
from numpy.testing import assert_array_almost_equal
import mne

from hypyp.fnirs.cohort import Cohort
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet
from hypyp.fnirs.subject import Subject
from hypyp.fnirs.dyad import Dyad
from hypyp.fnirs.data_browser import DataBrowser
from hypyp.fnirs.preprocessors.base_preprocessor import PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_HAEMO_FILTERED_KEY
from hypyp.fnirs.preprocessors.mne_preprocessor import MnePreprocessStep, MnePreprocessor, DummyPreprocessor
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
    loader = DataBrowser()
    paths = loader.paths
    assert len(paths) > 0

def test_list_paths():
    loader = DataBrowser()
    previous_count = len(loader.paths)
    loader.add_source('/foo')
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

def test_list_files():
    loader = DataBrowser()
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
    raw = MnePreprocessor().read_file(file_path)
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
    ch_names = MnePreprocessor().get_nirs_ch_names(meas_list, lambdas)
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
    subject = Subject(label='my_subject')
    subject.load_file(MnePreprocessor(), filepath)
    assert subject.label == 'my_subject'
    assert subject.filepath == filepath
    assert subject.raw is not None
    assert len(subject.tasks_annotations) == 1 # default task, which is the complete record
    assert subject.preprocess_steps is None
    assert subject.is_preprocessed == False
    assert subject.epochs_per_task is None # need preprocessing to extract epochs

def test_subject_tasks():
    subject = Subject(tasks_annotations=[('my_task', 1, 2)])
    assert len(subject.tasks_annotations) == 1
    subject.task_keys[0] == 'my_task'

def test_subject_epochs():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT),
        ('task2', 3, TASK_NEXT_EVENT),
    ]
    subject = Subject(tasks_annotations=tasks)
    preprocessor = MnePreprocessor()
    subject.load_file(preprocessor, snirf_file1, preprocess=True)
    subject.populate_epochs_from_tasks()
    assert len(subject.get_epochs_for_task('task1')) == 2
    n_events = subject.get_epochs_for_task('task1').events.shape[0]
    assert n_events == 2
    assert len(subject.get_epochs_for_task('task2')) == 3

def test_subject_force_task():
    tasks = [
        ('task1', 1, 2),
        ('task2', 4, 5),
    ]
    subject = Subject(tasks_time_range=tasks)
    preprocessor = MnePreprocessor()
    subject.load_file(preprocessor, snirf_file1, preprocess=True)
    subject.populate_epochs_from_tasks()
    assert len(subject.get_epochs_for_task('task1')) == 1
    #n_events = subject.get_epochs_for_task('task1').events.shape[0]
    #assert n_events == 2
    #assert len(subject.get_epochs_for_task('task2')) == 3
    

def test_dummy_preprocessor():
    subject = Subject(tasks_annotations=[('task1', 1, TASK_NEXT_EVENT)]).load_file(MnePreprocessor(), snirf_file1)
    subject.preprocess(DummyPreprocessor())
    assert len(subject.preprocess_steps) == 1
    assert subject.is_preprocessed == True
    assert subject.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert subject.epochs_per_task is None
    subject.populate_epochs_from_tasks()
    assert len(subject.epochs_per_task) > 0

def test_mne_preprocessor():
    subject = Subject().load_file(MnePreprocessor(), snirf_file1)
    subject.preprocess(MnePreprocessor())
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
    preprocessor = MnePreprocessor()
    subject = Subject().load_file(preprocessor, snirf_file1)
    dyad = Dyad(subject, subject)
    assert dyad.is_preprocessed == False

    dyad.preprocess(preprocessor)
    dyad.populate_epochs_from_tasks()
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
    s1 = Subject(tasks_annotations=[tasks[0], tasks[1]])
    s2 = Subject(tasks_annotations=[tasks[1], tasks[2]])
    assert len(Dyad(s1, s1).tasks) == 2
    assert len(Dyad(s2, s2).tasks) == 2
    assert len(Dyad(s1, s2).tasks) == 1
    
    
def test_dyad_compute_pair_wtc():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    pair = dyad.get_pairs()[0].sub((0, 10)) # Take 10% of the file
    wtc = dyad.get_pair_wtc(pair, PywaveletsWavelet())
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(wtc.wtc) == pytest.approx(1)

def test_dyad_cwt_cache_during_wtc():
    subject1 = Subject(label='subject1').load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    subject2 = Subject(label='subject2').load_file(DummyPreprocessor(), snirf_file2, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject1, subject2)
    pair = dyad.get_pairs()[0].sub((0, 10)) # Take 10% of the file
    wavelet = PywaveletsWavelet(cache_dict=dict())
    with patch.object(wavelet, 'cwt', wraps=wavelet.cwt) as spy_method:
        wtc_no_cache = dyad.get_pair_wtc(pair, wavelet)
        assert spy_method.call_count == 2
        wtc_with_cache = dyad.get_pair_wtc(pair, wavelet) # this should not call cwt, but use the cache
        assert spy_method.call_count == 2
        assert np.all(wtc_no_cache.wtc == wtc_with_cache.wtc)
        # make sure we can clear the cache
        wavelet.clear_cache()
        dyad.get_pair_wtc(pair, wavelet)
        assert spy_method.call_count == 4

def test_dyad_compute_all_wtc():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    assert dyad.is_wtc_computed == False
    dyad.compute_wtcs(time_range=(0,5)) # TODO have a more simple wavelet for fast computing
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == len(subject.pre.ch_names)**2
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(dyad.wtcs[0].wtc) == pytest.approx(1)
    
def test_dyad_compute_str_match_wtc():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    dyad.compute_wtcs(match='760', time_range=(0,5))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == (len(subject.pre.ch_names)/2)**2

def test_dyad_compute_regex_match_wtc():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    regex = re.compile(r'^S1.*760')
    dyad.compute_wtcs(match=regex, time_range=(0,5))
    assert len(dyad.wtcs) == 4
    assert dyad.wtcs[0].label == dyad.get_pairs()[0].label

def test_dyad_compute_tuple_match_wtc():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    regex1 = re.compile(r'^S1_D1.*760')
    regex2 = re.compile(r'.*760')
    dyad.compute_wtcs(match=(regex1, regex2), time_range=(0,5))
    assert len(dyad.wtcs) == 16
    #[print(wtc.label) for wtc in dyad.wtcs]

def test_dyad_wtc_per_task():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT), # these 2 events have different duration
        ('task3', 3, TASK_NEXT_EVENT),
    ]
    subject = Subject(tasks_annotations=tasks).load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    subject.populate_epochs_from_tasks()
    dyad = Dyad(subject, subject)
    ch_name = 'S1_D1 760'
    pairs = dyad.get_pairs(match=ch_name)
    assert len(pairs) == 2
    dyad.compute_wtcs(match=ch_name, time_range=(0,5))
    assert len(dyad.wtcs) == 2
    assert dyad.wtcs[0].wtc.shape[1] != dyad.wtcs[1].wtc.shape[1] # not the same duration
    assert 'task1' in [wtc.task for wtc in dyad.wtcs] # order may have changed because of task intersection

def test_cohort_wtc():
    subject1 = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad1 = Dyad(subject1, subject1)

    subject2 = Subject().load_file(DummyPreprocessor(), snirf_file2, preprocess=True).populate_epochs_from_tasks()
    dyad2 = Dyad(subject2, subject2)

    dyad3 = Dyad(subject1, subject2)

    # Add a bunch of "dyad3" to our list, so we have a number of "others" for our first dyad
    dyads = [dyad1, dyad2, dyad3, dyad3, dyad3, dyad3]
    cohort = Cohort(dyads)
    assert len(cohort.dyads) == len(dyads)
    assert len(cohort.dyads_shuffle) == len(dyads)*(len(dyads)-1)
    assert cohort.is_wtc_computed == False

    wtcs_kwargs = dict(match='S1_D1 760', time_range=(0,5))

    cohort.compute_wtcs(**wtcs_kwargs)
    assert cohort.is_wtc_computed == True
    assert len(dyad1.wtcs) == 1
    
    # dyads shuffle are computed only when we want significance
    assert cohort.is_wtc_shuffle_computed == False
    assert cohort.dyads_shuffle[0].wtcs is None
    assert cohort.dyads[0].wtcs[0].sig is None

    cohort.compute_wtcs_shuffle(**wtcs_kwargs)

    assert cohort.is_wtc_shuffle_computed == True
    assert len(cohort.dyads_shuffle[0].wtcs) == 1

    cohort.compute_wtcs_significance()
    assert cohort.dyads[0].wtcs[0].sig_p_value > 0

def test_dyad_computes_whole_record_by_default():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True)
    dyad = Dyad(subject, subject)
    dyad.compute_wtcs(match='S1_D1 760', time_range=(0,5))
    assert len(dyad.wtcs) == 1

def test_dyad_does_not_compute_tasks_when_epochs_not_loaded():
    subject = Subject(tasks_annotations=[('task1', 1, TASK_NEXT_EVENT)]).load_file(DummyPreprocessor(), snirf_file1, preprocess=True)
    dyad = Dyad(subject, subject)
    with pytest.raises(Exception):
        # This should raise an exception, since the epochs have not been loaded from annotations
        dyad.compute_wtcs(match='S1_D1 760', time_range=(0,5))

def test_dyad_connection_matrix():
    subject = Subject().load_file(DummyPreprocessor(), snirf_file1, preprocess=True)
    dyad = Dyad(subject, subject)
    match = re.compile(r'^S1_.*760')
    dyad.compute_wtcs(match=match, time_range=(0,5))
    print(dyad.get_pairs(match=match))
    # channels detectors expected: D1-D1, D1-D2, D2-D1, D2-D2
    assert len(dyad.wtcs) == 4

def test_dyad_connection_matrix():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT),
        ('task2', 2, TASK_NEXT_EVENT),
        ('task3', 3, TASK_NEXT_EVENT),
    ]
    subject = Subject(tasks_annotations=tasks).load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    dyad = Dyad(subject, subject).compute_wtcs(match=re.compile(r'^S1_.*760'), time_range=(0,5))
    # channels detectors expected on 3 tasks: D1-D1, D1-D2, D2-D1, D2-D2
    assert len(dyad.wtcs) == 3*4
    conn_matrix, task_names, row_names, col_names = dyad.get_connection_matrix()

    assert len(conn_matrix.shape) == 3
    assert conn_matrix.shape[0] == 3 # 3 tasks
    assert conn_matrix.shape[1] == 2
    assert conn_matrix.shape[2] == 2

    assert task_names[0] == 'task1'
    assert row_names[0] == 'S1_D1 760'
    assert row_names[-1] == 'S1_D2 760'
    assert col_names[0] == 'S1_D1 760'
    assert col_names[-1] == 'S1_D2 760'

    # Since the dyad is twice the same subject, the diagonal should be 1
    assert conn_matrix[0,0,0] == pytest.approx(1)
    assert conn_matrix[0,1,1] == pytest.approx(1)
    assert conn_matrix[0,0,1] < 1
    # Same subject so the matrix should be symetric on every task
    assert np.all(conn_matrix[:,0,1] == conn_matrix[:,1,0])

    # Make sure results for different tasks are not the same
    assert conn_matrix[0,0,1] != conn_matrix[1,0,1]

def test_dyad_p_value_matrix():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT),
        ('task2', 2, TASK_NEXT_EVENT),
        ('task3', 3, TASK_NEXT_EVENT),
    ]
    match = re.compile(r'^S1_.*760')
    subject1 = Subject(tasks_annotations=tasks).load_file(DummyPreprocessor(), snirf_file1, preprocess=True).populate_epochs_from_tasks()
    subject2 = Subject(tasks_annotations=tasks).load_file(DummyPreprocessor(), snirf_file2, preprocess=True).populate_epochs_from_tasks()
    dyad1 = Dyad(subject1, subject1)
    dyad2 = Dyad(subject2, subject2)

    # TODO we have too many methods to call, this should be much simpler
    # Add a bunch of "dyad2" in the cohort to have a p-value
    # TODO we should use synthetic data instead
    kwargs = dict(match=match, time_range=(0,5))
    Cohort([dyad1, dyad2, dyad2, dyad2, dyad2]).compute_wtcs(**kwargs, significance=True)

    # We need a cohort to have a p-value
    p_value_matrix = dyad1.get_p_value_matrix()[0]
    assert len(p_value_matrix.shape) == 3

    assert p_value_matrix[0,0,0] > 0
    assert p_value_matrix[0,0,0] < 1

@pytest.mark.skip(reason="TODO: have significance comparison")
def test_significance():
    pass

@pytest.mark.skip(reason="TODO: optimisation")
def test_pair_indexing_in_matrix():
    pass

# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    loader = DataBrowser()
    previous_count = len(loader.paths)
    loader.download_demo_dataset()
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():



    

