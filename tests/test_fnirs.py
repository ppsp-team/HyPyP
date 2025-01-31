import pytest
from unittest.mock import patch
import warnings
import logging
import re
import tempfile

import numpy as np
from numpy.testing import assert_array_almost_equal
import mne

from hypyp.fnirs.cohort import Cohort
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.fnirs.subject import Subject
from hypyp.fnirs.channel_roi import ChannelROI
from hypyp.fnirs.dyad import Dyad
from hypyp.fnirs.data_browser import DataBrowser
from hypyp.fnirs.preprocessor.base_step import PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_HAEMO_FILTERED_KEY
from hypyp.fnirs.preprocessor.implementations.mne_step import MneStep
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_basic import MnePreprocessorBasic
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_upstream import MnePreprocessorUpstream
from hypyp.utils import TASK_NEXT_EVENT

fif_file = './data/sub-110_session-1_pre.fif'
snirf_file1 = './data/fNIRS/DCARE_02_sub1.snirf'
snirf_file2 = './data/fNIRS/DCARE_02_sub2.snirf'
fnirs_files = [fif_file, snirf_file1, snirf_file2]

# avoid all the output from mne
logging.disable()

# Test helpers
def get_test_subject():
    tasks = [('task1', 0, 20)]
    return Subject(tasks_time_range=tasks).load_file(snirf_file1)

def get_test_subjects():
    tasks = [('task1', 0, 20)]
    subject1 = Subject(tasks_time_range=tasks).load_file(snirf_file1)
    subject2 = Subject(tasks_time_range=tasks).load_file(snirf_file2)
    return subject1, subject2

def get_test_ch_match_one():
    return 'S1_D1 760'

def get_test_ch_match_few():
    return re.compile(r'^S1_.*760')

#
# Data Browser
#
def test_list_paths():
    browser = DataBrowser()
    paths = browser.paths
    assert len(paths) > 0

def test_add_source():
    browser = DataBrowser()
    previous_count = len(browser.paths)
    browser.add_source('/foo')
    new_paths = browser.paths
    assert len(new_paths) == previous_count + 1

def test_list_files():
    browser = DataBrowser()
    assert len(browser.paths) > 0
    assert len(browser.list_all_files()) > 0
    # path should be absolute
    assert browser.list_all_files()[0].startswith('/')


#
# Preprocessing pipeline
#

# Try to load every file types we have
@pytest.mark.parametrize("file_path", fnirs_files)
def test_data_loader_all_types(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = MnePreprocessorBasic().read_file(file_path)
    assert raw.info['sfreq'] > 0
    assert len(raw.ch_names) > 0
    
def test_preprocess_step():
    key = 'foo_key'
    desc = 'foo_description'
    raw = mne.io.RawArray(np.array([[1., 2.]]), mne.create_info(['foo'], 1))
    step = MneStep(raw, key, desc)
    assert step.obj.get_data().shape[0] == 1
    assert step.obj.get_data().shape[1] == 2
    assert step.key == key
    assert step.desc == desc
    assert step.duration == 2

#
# Subject
#

def test_subject():
    filepath = snirf_file1
    subject = Subject(label='my_subject')
    subject.load_file(filepath, MnePreprocessorBasic(), preprocess=False)
    assert subject.label == 'my_subject'
    assert subject.filepath == filepath
    assert subject.raw is not None
    assert len(subject.tasks_annotations) == 1 # default task, which is the complete record
    assert subject.preprocess_steps is None
    assert subject.is_preprocessed == False
    assert subject.epochs_per_task is None # need preprocessing to extract epochs

def test_subject_tasks_annotations():
    subject = Subject(tasks_annotations=[('my_task', 1, 2)])
    assert len(subject.tasks_annotations) == 1
    assert subject.task_keys[0] == 'my_task'

def test_subject_tasks_time_range():
    subject = Subject(tasks_time_range=[('my_task_in_time', 1, 2)])
    assert len(subject.tasks_time_range) == 1
    assert subject.task_keys[0] == 'my_task_in_time'

def test_subject_tasks_combined():
    subject = Subject(
        tasks_annotations=[('my_task', 1, 2)],
        tasks_time_range=[('my_task_in_time', 1, 2)]
    )
    assert len(subject.task_keys) == 2

def test_subject_epochs():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT),
        ('task2', 3, TASK_NEXT_EVENT),
    ]
    subject = Subject(tasks_annotations=tasks)
    subject.load_file(snirf_file1, MnePreprocessorBasic())
    assert len(subject.get_epochs_for_task('task1')) == 2
    n_events = subject.get_epochs_for_task('task1').events.shape[0]
    assert n_events == 2
    assert len(subject.get_epochs_for_task('task2')) == 3

def test_subject_time_range_task():
    tasks = [
        ('task1', 1, 2),
        ('task2', 4, 5),
    ]
    subject = Subject(tasks_time_range=tasks)
    subject.load_file(snirf_file1, MnePreprocessorBasic())
    epochs_task1 = subject.get_epochs_for_task('task1')
    epochs_task2 = subject.get_epochs_for_task('task2')
    assert len(epochs_task1) == 1
    n_events = epochs_task1.events.shape[0]
    assert n_events == 1
    assert len(epochs_task2) == 1
    
def test_subject_time_range_task_recurring_event():
    tasks = [
        ('task1', 1, 2),
        ('task1', 4, 5),
        ('task1', 8, 10),
    ]
    subject = Subject(tasks_time_range=tasks)
    subject.load_file(snirf_file1, MnePreprocessorBasic())
    epochs = subject.get_epochs_for_task('task1')
    assert len(epochs) == 3
    n_events = epochs.events.shape[0]
    assert n_events == 3
    

def test_upstream_preprocessor():
    subject = Subject(tasks_annotations=[('task1', 1, TASK_NEXT_EVENT)]).load_file(snirf_file1, MnePreprocessorUpstream())
    assert len(subject.preprocess_steps) == 1
    assert subject.is_preprocessed == True
    assert subject.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert subject.epochs_per_task is not None
    assert len(subject.epochs_per_task) > 0

def test_mne_preprocessor():
    preprocessor = MnePreprocessorBasic()
    subject = Subject().load_file(snirf_file1, preprocessor, preprocess=False)
    subject.preprocess(preprocessor)
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
    subject1 = Subject().load_file(snirf_file1, preprocess=False)
    subject2 = Subject().load_file(snirf_file2, preprocess=False)
    dyad = Dyad(subject1, subject2)
    assert dyad.is_preprocessed == False

    dyad.preprocess(MnePreprocessorBasic())
    assert dyad.is_preprocessed == True

    pairs = dyad.get_pairs(dyad.s1, dyad.s2)
    #assert len(pairs) == n_channels * n_channels * n_epochs # TODO this is the test we with, with epochs
    assert len(pairs) == len(subject1.pre.ch_names) * len(subject2.pre.ch_names)
    assert pairs[0].label is not None
    assert pairs[0].label_ch1 == subject1.pre.ch_names[0]
    assert pairs[0].label_ch2 == subject2.pre.ch_names[0]
    assert subject1.label in dyad.label

def test_dyad_pairs_recurring_event():
    tasks = [
        ('task1', 1, 2),
        ('task1', 3, 4),
        ('task1', 5, 6),
    ]
    # Use the same file for the 2 subjects
    subject1 = Subject(tasks_time_range=tasks).load_file(snirf_file1)
    subject2 = Subject(tasks_time_range=tasks).load_file(snirf_file2)
    dyad = Dyad(subject1, subject2)

    pairs = dyad.get_pairs(dyad.s1, dyad.s2, ch_match=get_test_ch_match_one())
    assert len(pairs) == 3
    # make sure we don't have the same signal
    assert np.sum(pairs[0].y1 - pairs[1].y1) != 0
    assert pairs[0].epoch_id == 0
    assert pairs[1].epoch_id == 1
    assert pairs[2].epoch_id == 2
    
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
    
def test_dyad_tasks_with_same_name_different_definition():
    # tasks with the same name should be considered to be the same task, even if they have a different "definition" (e.g. start at different times)
    tasks1 = [
        ('my_task1', 1, 2),
        ('my_task2', 3, 4),
        ('my_task3', 5, 6),
        ('different_name', 99, 100),
    ]
    tasks2 = [
        ('my_task1', 11, 12),
        ('my_task2', 13, 14),
        ('my_task3', 15, 16),
        ('unknown_task', 99, 100),
    ]
    s1 = Subject(tasks_annotations=tasks1)
    s2 = Subject(tasks_annotations=tasks2)
    merged_task_names = [t[0] for t in Dyad(s1, s2).tasks]
    assert 'my_task1' in merged_task_names
    assert 'my_task2' in merged_task_names
    assert 'my_task3' in merged_task_names
    assert 'different_name' not in merged_task_names
    assert 'unknown_task' not in merged_task_names

def test_dyad_check_sfreq_same():
    subject1 = Subject().load_file(snirf_file1)
    subject1.pre.resample(sfreq=5)
    subject2 = Subject().load_file(snirf_file2)
    dyad = Dyad(subject1, subject2)
    with pytest.raises(Exception):
        dyad.get_pairs(dyad.s1, dyad.s2)
    
def test_dyad_compute_pair_wtc():
    # test with the same subject, so we can check we have a high coherence
    subject = get_test_subject()
    dyad = Dyad(subject, subject)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wtc = ComplexMorletWavelet().wtc(pair)
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(wtc.W) == pytest.approx(1)
    assert wtc.label_dyad == dyad.label

def test_dyad_cwt_cache_during_wtc():
    subject1, subject2 = get_test_subjects()
    dyad = Dyad(subject1, subject2)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wavelet = ComplexMorletWavelet(cache=dict())
    with patch.object(wavelet, 'cwt', wraps=wavelet.cwt) as spy_method:
        wtc_no_cache = wavelet.wtc(pair)
        assert spy_method.call_count == 2
        wtc_with_cache = wavelet.wtc(pair) # this should not call cwt, but use the cache
        assert spy_method.call_count == 2
        assert np.all(wtc_no_cache.W == wtc_with_cache.W)
        # make sure we can clear the cache
        wavelet._clear_cache()
        wavelet.wtc(pair)
        assert spy_method.call_count == 4

def test_dyad_coi_cache_during_wtc():
    subject1, subject2 = get_test_subjects()
    dyad = Dyad(subject1, subject2)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wavelet = ComplexMorletWavelet(cache=dict())
    with patch.object(wavelet, '_get_cone_of_influence', wraps=wavelet._get_cone_of_influence) as spy_method:
        wtc_no_cache = wavelet.wtc(pair)
        assert spy_method.call_count == 1
        wtc_with_cache = wavelet.wtc(pair)
        assert spy_method.call_count == 1
        assert np.all(wtc_no_cache.coi == wtc_with_cache.coi)
        assert np.all(wtc_no_cache.coif == wtc_with_cache.coif)
        # make sure we can clear the cache
        wavelet._clear_cache()
        _ = wavelet.wtc(pair)
        assert spy_method.call_count == 2

def test_dyad_cwt_cache_with_different_times():
    # When computing intra-subject, we cannot re-use the cache since the cwt might have been computed cropped to match lenght of both signal
    # We would have this error if the cache is not invalidated and we have different length
    #   "ValueError: operands could not be broadcast together with shapes (40,79) (40,157)"
    # This can happen for annotation based tasks. Here we force a different task by changing the 2nd subject
    subject1 = Subject(label='subject1', tasks_time_range=[('my_task', 0, 10)]).load_file(snirf_file1)
    subject2 = Subject(label='subject2', tasks_time_range=[('my_task', 0, 20)]).load_file(snirf_file2)
    # Force a different task length for subject2 to have a different lenght
    dyad = Dyad(subject1, subject1)
    dyad.s2 = subject2 # hack for the test
    dyad.compute_wtcs(ch_match=get_test_ch_match_one(), with_intra=True)

def test_dyad_compute_all_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    assert dyad.is_wtc_computed == False
    dyad.compute_wtcs(only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == len(subject.pre.ch_names)**2
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(dyad.wtcs[0].W) == pytest.approx(1)

    assert len(dyad.df['channel1'].unique()) == 32
    
def test_dyad_computes_intra_subject():
    subject1 = Subject().load_file(snirf_file1)
    subject2 = Subject().load_file(snirf_file2)
    dyad = Dyad(subject1, subject2)
    dyad.compute_wtcs(only_time_range=(0,10), with_intra=True)
    assert subject1.is_wtc_computed == True
    assert subject2.is_wtc_computed == True
    assert len(dyad.wtcs) == len(subject1.intra_wtcs)
    assert len(dyad.wtcs) == len(subject2.intra_wtcs)
    
def test_dyad_computes_intra_subject_channel_match():
    subject1 = Subject().load_file(snirf_file1)
    subject2 = Subject().load_file(snirf_file2)
    dyad = Dyad(subject1, subject2)
    # have a different channel for each subject
    ch_match = ('S1_D1 760', 'S1_D2 760')
    dyad.compute_wtcs(ch_match=ch_match, only_time_range=(0,10), with_intra=True)
    df_intra = dyad.df[dyad.df['is_intra'] == True]
    df_inter = dyad.df[dyad.df['is_intra'] == False]
    assert len(df_intra) > 0
    assert np.all(df_intra['channel1'] == df_intra['channel2'])
    assert np.all(df_inter['channel1'] != df_inter['channel2'])

def test_dyad_compute_str_match_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    dyad.compute_wtcs(ch_match='760', only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == (len(subject.pre.pick('all').ch_names)/2)**2

def test_dyad_compute_regex_match_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few(), only_time_range=(0,10))
    assert len(dyad.wtcs) == 4
    assert dyad.wtcs[0].label_pair == dyad.get_pairs(dyad.s1, dyad.s2)[0].label

def test_dyad_compute_tuple_match_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    regex1 = re.compile(r'^S1_D1.*760')
    regex2 = re.compile(r'.*760')
    dyad.compute_wtcs(ch_match=(regex1, regex2), only_time_range=(0,10))
    assert len(dyad.wtcs) == 16
    #[print(wtc.label) for wtc in dyad.wtcs]

def test_dyad_compute_list_match_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    ch_list = ['S1_D1 760', 'S1_D2 760', 'S2_D1 760']
    dyad.compute_wtcs(ch_match=ch_list, only_time_range=(0,10))
    assert len(dyad.wtcs) == len(ch_list) * len(ch_list)
    
def test_dyad_compute_list_per_subject_match_wtc():
    subject = Subject().load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    ch_list1 = ['S1_D1 760', 'S1_D2 760']
    ch_list2 = ['S2_D1 760', 'S2_D2 760']
    dyad.compute_wtcs(ch_match=(ch_list1, ch_list2), only_time_range=(0,10))
    assert len(dyad.wtcs) == len(ch_list1) * len(ch_list2)
    assert len(dyad.df['channel1'].unique()) == 2

    assert ch_list1[0] in dyad.df['channel1'].unique()
    assert ch_list2[0] not in dyad.df['channel1'].unique()

    assert ch_list1[0] not in dyad.df['channel2'].unique()
    assert ch_list2[0] in dyad.df['channel2'].unique()
    

def test_dyad_wtc_per_task():
    tasks = [
        ('task1', 1, TASK_NEXT_EVENT), # these 2 events have different duration
        ('task3', 3, TASK_NEXT_EVENT),
    ]
    subject = Subject(tasks_annotations=tasks).load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    ch_name = get_test_ch_match_one()
    pairs = dyad.get_pairs(dyad.s1, dyad.s2, ch_match=ch_name)
    # we will have multiple pairs because we have one pair per epoch
    assert len(pairs) == 5
    dyad.compute_wtcs(ch_match=ch_name)
    assert len(dyad.wtcs) == len(pairs)
    # must compare the first and last wtcs to make sure we are on different tasks (otherwise we might compare 2 epochs of the same task)
    assert dyad.wtcs[0].W.shape[1] != dyad.wtcs[-1].W.shape[1] # not the same duration
    assert 'task1' in [wtc.task for wtc in dyad.wtcs] # order may have changed because of task intersection

def test_dyad_task_annotations_and_time_range_combined():
    tasks_annotations = [
        ('task_annotation1', 1, TASK_NEXT_EVENT),
    ]
    tasks_time_range = [
        ('task_time_range', 10, 20),
    ]
    subject = Subject(tasks_annotations=tasks_annotations, tasks_time_range=tasks_time_range)
    subject.load_file(snirf_file1)
    dyad = Dyad(subject, subject)
    ch_name = get_test_ch_match_one()
    pairs = dyad.get_pairs(dyad.s1, dyad.s2, ch_match=ch_name)
    # We have the count from annotations + the count from time_range
    assert len(pairs) == 3
    dyad.compute_wtcs(ch_match=ch_name)
    assert len(dyad.wtcs) == len(pairs)
    found_tasks = [wtc.task for wtc in dyad.wtcs]
    assert 'task_annotation1' in found_tasks
    assert 'task_time_range' in found_tasks

def test_dyad_wtc_nan_channel_section():
    tasks = [
        ('task1', 0, 500),
    ]
    # Use the same file for the 2 subjects
    subject1 = Subject(tasks_time_range=tasks).load_file(snirf_file1)
    subject2 = Subject(tasks_time_range=tasks).load_file(snirf_file2)
    dyad = Dyad(subject1, subject2)

    epochs = subject1.get_epochs_for_task('task1')
    data = epochs.get_data()
    # Set some values to NaN to split in 3 sections
    data[0, 0, 20:40] = np.nan
    t = int(100*epochs.info['sfreq'])
    data[0, 0, t:t+1] = np.nan
    epochs._data = data
    dyad.compute_wtcs(ch_match=epochs.ch_names[0], downsample=None)
    df = dyad.df
    assert len(dyad.wtcs) == 3
    # the first section is too small, coherence should be NaN
    assert np.all(np.isnan(df[df['section']==0]['coherence'].head(1)))
    # the next 2 sections have enough data for wtc
    assert np.all(np.isfinite(df[df['section']==1]['coherence'].head(1)))
    assert np.all(np.isfinite(df[df['section']==2]['coherence'].head(1)))

def test_cohort_wtc():
    subject1, subject2 = get_test_subjects()
    dyad1 = Dyad(subject1, subject1)
    dyad2 = Dyad(subject2, subject2)
    dyad3 = Dyad(subject1, subject2)

    # Add a bunch of "dyad3" to our list, so we have a number of "others" for our first dyad
    dyads = [dyad1, dyad2, dyad3, dyad3, dyad3, dyad3]
    cohort = Cohort(dyads)
    assert len(cohort.dyads) == len(dyads)
    assert cohort.is_wtc_computed == False

    wtcs_kwargs = dict(ch_match=get_test_ch_match_one())

    cohort.compute_wtcs(**wtcs_kwargs)
    df = cohort.df
    assert cohort.is_wtc_computed == True
    assert len(dyad1.wtcs) == 1
    
    # dyads shuffle are computed only when we want significance
    assert cohort.is_wtc_shuffle_computed == False
    assert cohort.dyads_shuffle is None
    assert np.all(df['is_shuffle'] == False)

    cohort.compute_wtcs_shuffle(**wtcs_kwargs)
    df_with_shuffle = cohort.df
    assert len(cohort.dyads_shuffle) == len(dyads)*(len(dyads)-1)

    assert cohort.is_wtc_shuffle_computed == True
    assert len(cohort.dyads_shuffle[0].wtcs) == 1
    assert len(df_with_shuffle['is_shuffle'].unique()) == 2


def test_dyad_computes_whole_record_by_default():
    subject = get_test_subject()
    dyad = Dyad(subject, subject)
    dyad.compute_wtcs(ch_match=get_test_ch_match_one())
    assert len(dyad.wtcs) == 1

def test_dyad_does_not_compute_tasks_when_epochs_not_loaded():
    subject = Subject(tasks_annotations=[('task1', 1, TASK_NEXT_EVENT)])
    subject.load_file(snirf_file1, preprocess=False)
    dyad = Dyad(subject, subject)
    with pytest.raises(Exception):
        # This should raise an exception, since the epochs have not been loaded from annotations
        dyad.compute_wtcs(ch_match=get_test_ch_match_one(), only_time_range=(0,10))

def test_dyad_coherence_pandas():
    subject1, subject2 = get_test_subjects()
    dyad = Dyad(subject1, subject2)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few())
    df = dyad._get_coherence_df()
    assert len(df['task'].unique()) == len(dyad.s1.task_keys)
    assert len(df['channel1'].unique()) == 2
    assert len(df['channel2'].unique()) == 2
    assert len(df['dyad'].unique()) == 1
    assert df['subject1'].unique()[0] == subject1.label
    assert df['subject2'].unique()[0] == subject2.label

def test_dyad_coherence_pandas_with_intra():
    subject1, subject2 = get_test_subjects()
    dyad = Dyad(subject1, subject2)

    with pytest.raises(Exception):
        dyad.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=False)
        # Since intra subject is not computed, it should raise
        dyad._get_coherence_df(with_intra=True)

    dyad.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=True)
    df = dyad._get_coherence_df(with_intra=True)
    assert len(df['task'].unique()) == len(dyad.s1.task_keys)
    assert len(df['dyad'].unique()) == 3
    assert len(df['is_intra'].unique()) == 2

def test_cohort_coherence_pandas():
    subject1, subject2 = get_test_subjects()
    subject3, _ = get_test_subjects()
    dyad1 = Dyad(subject1, subject2, label='dyad1')
    dyad2 = Dyad(subject1, subject3, label='dyad2')
    dyad3 = Dyad(subject2, subject3, label='dyad3')
    cohort = Cohort([dyad1, dyad2, dyad3])
    cohort.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=False)
    df = cohort.df
    assert len(df['task'].unique()) == len(dyad1.s1.task_keys)
    assert len(df['channel1'].unique()) == 2
    assert len(df['channel2'].unique()) == 2
    assert len(df['dyad'].unique()) == 3
    assert np.all(df['is_intra'] == False)

def test_cohort_coherence_pandas_with_intra():
    cohort = Cohort([Dyad(*get_test_subjects(), label='dyad1')])
    cohort.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=True)
    df = cohort.df
    assert len(df['is_intra'].unique()) == 2


def test_dyad_coherence_pandas_on_roi():
    subject1, subject2 = get_test_subjects()
    dyad = Dyad(subject1, subject2)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few())
    df = dyad._get_coherence_df()
    assert len(df['roi1']) == 4
    assert len(df['roi2']) == 4

def test_wtc_downsampling():
    subject = get_test_subject()
    dyad = Dyad(subject, subject)
    n = 100
    Cohort([dyad]).compute_wtcs(ch_match=get_test_ch_match_one(), downsample=n)
    assert len(dyad.wtcs[0].times) <= n
    

def test_save_cohort_to_disk():
    subject = get_test_subject()
    dyad = Dyad(subject, subject)
    cohort = Cohort([dyad])

    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name
        cohort.save_pickle(file_path)
        cohort_reloaded = Cohort.from_pickle(file_path)
    
    assert len(cohort_reloaded.dyads) == len(cohort.dyads)

def test_save_cohort_df_to_disk():
    subject = get_test_subject()
    dyad = Dyad(subject, subject)
    cohort = Cohort([dyad])
    cohort.compute_wtcs()

    with tempfile.NamedTemporaryFile(suffix='.feather') as temp_file:
        file_path = temp_file.name
        cohort.save_feather(file_path)
        df = CoherenceDataFrame.from_feather(file_path)
    
    assert np.all(df['subject1'] == subject.label)
    
def test_cohort_run_estimation(capsys):
    subject = get_test_subject()
    dyads = []
    for _ in range(10):
        dyads.append(Dyad(subject, subject))
    cohort = Cohort(dyads)
    cohort.estimate_wtcs_run_time()
    out = capsys.readouterr()
    assert 'time' in str(out)

def test_lionirs_channel_grouping():
    roi_file_path = 'data/lionirs/channel_grouping_7ROI.mat'
    croi = ChannelROI.from_lionirs_file(roi_file_path)
    assert len(croi.rois.keys()) == 14

    key1 = list(croi.rois.keys())[0]
    roi1 = croi.rois[key1]
    ch1 = roi1[0]
    assert key1 == 'PreFr_L'
    assert len(roi1) == 5
    assert ch1 == 'S1_D9'

    assert len(croi.ordered_ch_names) == 50
    # Actual ordering:
    # ['S1_D9', 'S2_D9', 'S2_D2', 'S1_D2', 'S2_D1', 'S3_D1', 'S3_D2', ...]

    unordered_names = ['S1_D2 hbo', 'S1_D2 hbr', 'S1_D9 hbo', 'S1_D9 hbr', 'whatever']
    ordered_names = croi.get_ch_names_in_order(unordered_names)
    assert ordered_names[0] == 'S1_D9 hbo'
    assert ordered_names[1] == 'S1_D9 hbr'
    assert ordered_names[2] == 'S1_D2 hbo'
    assert ordered_names[3] == 'S1_D2 hbr'
    assert ordered_names[4] == 'whatever'

    assert croi.group_boundaries_sizes[:2] == [0, 5]
    assert croi.group_boundaries_sizes[-1] == len(croi.ordered_ch_names)

    assert croi.get_roi_from_channel(ordered_names[0]) == 'PreFr_L'

def test_ordered_subject_ch_names():
    roi_file_path = 'data/lionirs/channel_grouping_7ROI.mat'
    croi = ChannelROI.from_lionirs_file(roi_file_path)
    subject = Subject(channel_roi=croi).load_file(snirf_file1)
    ch_names = subject.ordered_ch_names
    assert ch_names[0] == 'S2_D2 760'


# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    browser = DataBrowser()
    previous_count = len(browser.paths)
    browser.download_demo_dataset()
    new_paths = browser.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():

