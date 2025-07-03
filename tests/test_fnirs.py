import pytest
from unittest.mock import patch
import warnings
import logging
import re
import tempfile

import numpy as np
from numpy.testing import assert_array_almost_equal
import mne

from hypyp.fnirs.study import Study
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.pair_signals import PairSignals
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.fnirs.recording import Recording
from hypyp.fnirs.channel_roi import ChannelROI
from hypyp.fnirs.dyad import Dyad
from hypyp.fnirs.data_browser import DataBrowser
from hypyp.fnirs.preprocessor.base_step import PREPROCESS_STEP_BASE_KEY, PREPROCESS_STEP_HAEMO_FILTERED_KEY
from hypyp.fnirs.preprocessor.implementations.mne_step import MneStep
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_raw_to_haemo import MnePreprocessorRawToHaemo
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_as_is import MnePreprocessorAsIs
from hypyp.utils import TASK_NEXT_EVENT, Task

fif_file = './data/sub-110_session-1_pre.fif'
snirf_file1 = './data/NIRS/DCARE_02_sub1.snirf'
snirf_file2 = './data/NIRS/DCARE_02_sub2.snirf'
fnirs_files = [fif_file, snirf_file1, snirf_file2]

# avoid all the output from mne
logging.disable()

# Test helpers
def get_test_recording():
    tasks = [Task('task1', onset_time=0, duration=20)]
    return Recording(tasks=tasks).load_file(snirf_file1)

def get_test_recordings(count:int=2):
    tasks = [Task('task1', onset_time=0, duration=20)]
    recordings = [Recording(tasks=tasks, subject_label=f's{i+1}').load_file(snirf_file1) for i in range(count)]
    return recordings

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
    raw = MnePreprocessorRawToHaemo().read_file(file_path)
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
# Subject Recordings
#

def test_recording():
    filepath = snirf_file1
    recording = Recording(subject_label='my_subject')
    recording.load_file(filepath, MnePreprocessorRawToHaemo(), preprocess=False)
    assert recording.subject_label == 'my_subject'
    assert recording.filepath == filepath
    assert recording.mne_raw is not None
    assert len(recording.tasks) == 1 # default task, which is the complete record
    assert recording.preprocess_steps is None
    assert recording.is_preprocessed == False
    assert recording.epochs_per_task is None # need preprocessing to extract epochs

def test_recording_tasks():
    recording = Recording(tasks=[Task('my_task_in_time', onset_time=1, duration=2)])
    assert len(recording.tasks) == 1
    assert recording.task_keys[0] == 'my_task_in_time'

def test_recording_epochs():
    tasks = [
        Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT),
        Task('task2', onset_event_id=3, offset_event_id=TASK_NEXT_EVENT),
    ]
    recording = Recording(tasks=tasks)
    recording.load_file(snirf_file1, MnePreprocessorRawToHaemo())
    assert len(recording.get_epochs_for_task('task1')) == 2
    n_events = recording.get_epochs_for_task('task1').events.shape[0]
    assert n_events == 2
    assert len(recording.get_epochs_for_task('task2')) == 3

def test_recording_time_range_task():
    tasks = [
        Task('task1', onset_time=1, duration=1),
        Task('task2', onset_time=4, duration=1),
    ]
    recording = Recording(tasks=tasks)
    recording.load_file(snirf_file1, MnePreprocessorRawToHaemo())
    epochs_task1 = recording.get_epochs_for_task('task1')
    epochs_task2 = recording.get_epochs_for_task('task2')
    assert len(epochs_task1) == 1
    n_events = epochs_task1.events.shape[0]
    assert n_events == 1
    assert len(epochs_task2) == 1
    
def test_recording_time_range_task_recurring_event():
    tasks = [
        Task('task1', onset_time=1, duration=1),
        Task('task1', onset_time=4, duration=1),
        Task('task1', onset_time=8, duration=1),
    ]
    recording = Recording(tasks=tasks)
    recording.load_file(snirf_file1, MnePreprocessorRawToHaemo())
    epochs = recording.get_epochs_for_task('task1')
    assert len(epochs) == 3
    n_events = epochs.events.shape[0]
    assert n_events == 3
    

def test_upstream_preprocessor():
    recording = Recording(tasks=[Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT)]).load_file(snirf_file1, MnePreprocessorAsIs())
    assert len(recording.preprocess_steps) == 1
    assert recording.is_preprocessed == True
    assert recording.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert recording.epochs_per_task is not None
    assert len(recording.epochs_per_task) > 0

def test_mne_preprocessor():
    preprocessor = MnePreprocessorRawToHaemo()
    recording = Recording().load_file(snirf_file1, preprocessor, preprocess=False)
    recording.preprocess(preprocessor)
    assert len(recording.preprocess_steps) > 1
    assert recording.is_preprocessed == True
    assert recording.preprocess_steps[0].key == PREPROCESS_STEP_BASE_KEY
    assert recording.preprocess_steps[-1].key == PREPROCESS_STEP_HAEMO_FILTERED_KEY
    assert recording.preprocessed is not None
    assert recording.preprocess_step_keys[0] == recording.preprocess_steps[-1].key

    # can get step by key
    assert recording.get_preprocess_step(PREPROCESS_STEP_BASE_KEY).key == PREPROCESS_STEP_BASE_KEY
    assert recording.get_preprocess_step(PREPROCESS_STEP_HAEMO_FILTERED_KEY).key == PREPROCESS_STEP_HAEMO_FILTERED_KEY

#
# Dyad
#

def test_recording_dyad():
    s1 = Recording().load_file(snirf_file1, preprocess=False)
    s2 = Recording().load_file(snirf_file2, preprocess=False)
    dyad = Dyad(s1, s2)
    assert dyad.is_preprocessed == False

    dyad.preprocess(MnePreprocessorRawToHaemo())
    assert dyad.is_preprocessed == True

    pairs = dyad.get_pairs(dyad.s1, dyad.s2)
    #assert len(pairs) == n_channels * n_channels * n_epochs # TODO this is the test we with, with epochs
    assert len(pairs) == len(s1.preprocessed.ch_names) * len(s2.preprocessed.ch_names)
    assert pairs[0].label is not None
    assert pairs[0].label_ch1 == s1.preprocessed.ch_names[0]
    assert pairs[0].label_ch2 == s2.preprocessed.ch_names[0]
    assert s1.subject_label in dyad.label

def test_dyad_pairs_recurring_event():
    tasks = [
        Task('task1', onset_time=1, duration=1),
        Task('task1', onset_time=3, duration=1),
        Task('task1', onset_time=5, duration=1),
    ]
    # Use the same file for the 2 subjects
    s1 = Recording(tasks=tasks).load_file(snirf_file1)
    s2 = Recording(tasks=tasks).load_file(snirf_file2)
    dyad = Dyad(s1, s2)

    pairs = dyad.get_pairs(dyad.s1, dyad.s2, ch_match=get_test_ch_match_one())
    assert len(pairs) == 3
    # make sure we don't have the same signal
    assert np.sum(pairs[0].y1 - pairs[1].y1) != 0
    assert pairs[0].epoch_idx == 0
    assert pairs[1].epoch_idx == 1
    assert pairs[2].epoch_idx == 2
    
def test_dyad_tasks_intersection():
    tasks = [
        Task('my_task1', onset_time=1, duration=1),
        Task('my_task2', onset_time=3, duration=1),
        Task('my_task3', onset_time=5, duration=1),
    ]
    s1 = Recording(tasks=[tasks[0], tasks[1]])
    s2 = Recording(tasks=[tasks[1], tasks[2]])
    assert len(Dyad(s1, s1).tasks) == 2
    assert len(Dyad(s2, s2).tasks) == 2
    assert len(Dyad(s1, s2).tasks) == 1
    
def test_dyad_tasks_with_same_name_different_definition():
    # tasks with the same name should be considered to be the same task, even if they have a different "definition" (e.g. start at different times)
    tasks1 = [
        Task('my_task1', onset_event_id=1, duration=1),
        Task('my_task2', onset_event_id=3, duration=1),
        Task('my_task3', onset_event_id=5, duration=1),
        Task('different_name', onset_event_id=99, duration=1),
    ]
    tasks2 = [
        Task('my_task1', onset_event_id=11, duration=1),
        Task('my_task2', onset_event_id=13, duration=1),
        Task('my_task3', onset_event_id=15, duration=1),
        Task('unknown_task', onset_event_id=99, duration=1),
    ]
    s1 = Recording(tasks=tasks1)
    s2 = Recording(tasks=tasks2)
    merged_task_names = [task.name for task in Dyad(s1, s2).tasks]
    assert 'my_task1' in merged_task_names
    assert 'my_task2' in merged_task_names
    assert 'my_task3' in merged_task_names
    assert 'different_name' not in merged_task_names
    assert 'unknown_task' not in merged_task_names

def test_dyad_check_sfreq_same():
    s1 = Recording().load_file(snirf_file1)
    s1.preprocessed.resample(sfreq=5)
    s2 = Recording().load_file(snirf_file2)
    dyad = Dyad(s1, s2)
    with pytest.raises(Exception):
        dyad.get_pairs(dyad.s1, dyad.s2)
    
def test_dyad_compute_pair_wtc():
    # test with the same subject, so we can check we have a high coherence
    recording = get_test_recording()
    dyad = Dyad(recording, recording)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wtc = ComplexMorletWavelet().wtc(pair)
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(wtc.W) == pytest.approx(1)
    assert wtc.label_dyad == dyad.label

def test_dyad_cwt_cache_during_wtc():
    s1, s2 = get_test_recordings()
    dyad = Dyad(s1, s2)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wavelet = ComplexMorletWavelet(cache=dict())
    with patch.object(wavelet, 'cwt', wraps=wavelet.cwt) as spy_method:
        wtc_no_cache = wavelet.wtc(pair)
        assert spy_method.call_count == 2
        wtc_with_cache = wavelet.wtc(pair) # this should not call cwt, but use the cache
        assert spy_method.call_count == 2
        assert np.all(wtc_no_cache.W == wtc_with_cache.W)
        # make sure we can clear the cache
        wavelet.clear_cache()
        wavelet.wtc(pair)
        assert spy_method.call_count == 4

def test_dyad_coi_cache_during_wtc():
    s1, s2 = get_test_recordings()
    dyad = Dyad(s1, s2)
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
        wavelet.clear_cache()
        _ = wavelet.wtc(pair)
        assert spy_method.call_count == 2

def test_dyad_cwt_cache_with_different_times():
    # When computing intra-subject, we cannot re-use the cache since the cwt might have been computed cropped to match lenght of both signal
    # We would have this error if the cache is not invalidated and we have different length
    #   "ValueError: operands could not be broadcast together with shapes (40,79) (40,157)"
    # This can happen for annotation based tasks. Here we force a different task by changing the 2nd subject
    s1 = Recording(subject_label='subject1', tasks=[Task('my_task', onset_time=0, duration=10)]).load_file(snirf_file1)
    s2 = Recording(subject_label='subject2', tasks=[Task('my_task', onset_time=0, duration=20)]).load_file(snirf_file2)
    # Force a different task length for subject2 to have a different lenght
    dyad = Dyad(s1, s1)
    dyad.s2 = s2 # hack for the test
    dyad.compute_wtcs(ch_match=get_test_ch_match_one(), with_intra=True)

def test_dyad_compute_all_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    assert dyad.is_wtc_computed == False
    dyad.compute_wtcs(only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == len(recording.preprocessed.ch_names)**2
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(dyad.wtcs[0].W) == pytest.approx(1)

    assert len(dyad.df['channel1'].unique()) == 32
    
def test_dyad_computes_intra_subject():
    s1 = Recording().load_file(snirf_file1)
    s2 = Recording().load_file(snirf_file2)
    dyad = Dyad(s1, s2)
    dyad.compute_wtcs(only_time_range=(0,10), with_intra=True)
    assert s1.is_wtc_computed == True
    assert s2.is_wtc_computed == True
    assert len(dyad.wtcs) == len(s1.intra_wtcs)
    assert len(dyad.wtcs) == len(s2.intra_wtcs)
    
def test_dyad_computes_intra_subject_channel_match():
    s1 = Recording().load_file(snirf_file1)
    s2 = Recording().load_file(snirf_file2)
    dyad = Dyad(s1, s2)
    # have a different channel for each subject
    ch_match = ('S1_D1 760', 'S1_D2 760')
    dyad.compute_wtcs(ch_match=ch_match, only_time_range=(0,10), with_intra=True)
    df_intra = dyad.df[dyad.df['is_intra'] == True]
    df_inter = dyad.df[dyad.df['is_intra'] == False]
    assert len(df_intra) > 0
    assert np.all(df_intra['channel1'] == df_intra['channel2'])
    assert np.all(df_inter['channel1'] != df_inter['channel2'])
    # we keep track of the subject id for intra-subject coherence
    assert np.sum(df_intra['is_intra_of'] == 1) == 1
    assert np.sum(df_intra['is_intra_of'] == 2) == 1

def test_dyad_compute_str_match_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    dyad.compute_wtcs(ch_match='760', only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == (len(recording.preprocessed.pick('all').ch_names)/2)**2

def test_dyad_compute_regex_match_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few(), only_time_range=(0,10))
    assert len(dyad.wtcs) == 4
    assert dyad.wtcs[0].label_pair == dyad.get_pairs(dyad.s1, dyad.s2)[0].label

def test_dyad_compute_tuple_match_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    regex1 = re.compile(r'^S1_D1.*760')
    regex2 = re.compile(r'.*760')
    dyad.compute_wtcs(ch_match=(regex1, regex2), only_time_range=(0,10))
    assert len(dyad.wtcs) == 16
    #[print(wtc.label) for wtc in dyad.wtcs]

def test_dyad_compute_list_match_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    ch_list = ['S1_D1 760', 'S1_D2 760', 'S2_D1 760']
    dyad.compute_wtcs(ch_match=ch_list, only_time_range=(0,10))
    assert len(dyad.wtcs) == len(ch_list) * len(ch_list)
    
def test_dyad_compute_list_per_subject_match_wtc():
    recording = Recording().load_file(snirf_file1)
    dyad = Dyad(recording, recording)
    ch_list1 = ['S1_D1 760', 'S1_D2 760']
    ch_list2 = ['S2_D1 760', 'S2_D2 760']
    dyad.compute_wtcs(ch_match=(ch_list1, ch_list2), only_time_range=(0,10), with_intra=False)
    assert len(dyad.wtcs) == len(ch_list1) * len(ch_list2)
    assert len(dyad.df['channel1'].unique()) == 2

    assert ch_list1[0] in dyad.df['channel1'].unique()
    assert ch_list2[0] not in dyad.df['channel1'].unique()

    assert ch_list1[0] not in dyad.df['channel2'].unique()
    assert ch_list2[0] in dyad.df['channel2'].unique()
    

def test_dyad_wtc_per_task():
    tasks = [
        Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT), # these 2 events have different duration
        Task('task3', onset_event_id=3, offset_event_id=TASK_NEXT_EVENT),
    ]
    recording = Recording(tasks=tasks).load_file(snirf_file1)
    dyad = Dyad(recording, recording)
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
    tasks = [
        Task('task_annotation1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT),
        Task('task_time_range', onset_time=10, duration=20),
    ]
    recording = Recording(tasks=tasks)
    recording.load_file(snirf_file1)
    dyad = Dyad(recording, recording)
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
        Task('task1', onset_time=0, duration=500),
    ]
    # Use the same file for the 2 subjects
    s1 = Recording(tasks=tasks).load_file(snirf_file1)
    s2 = Recording(tasks=tasks).load_file(snirf_file2)
    dyad = Dyad(s1, s2)

    epochs = s1.get_epochs_for_task('task1')
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

def test_study_wtc():
    s1, s2 = get_test_recordings()
    dyad1 = Dyad(s1, s1)
    dyad2 = Dyad(s2, s2)
    dyad3 = Dyad(s1, s2)

    # Add a bunch of "dyad3" to our list, so we have a number of "others" for our first dyad
    dyads = [dyad1, dyad2, dyad3, dyad3, dyad3, dyad3]
    study = Study(dyads)
    assert len(study.dyads) == len(dyads)
    assert study.is_wtc_computed == False

    wtcs_kwargs = dict(ch_match=get_test_ch_match_one())
    study.compute_wtcs(**wtcs_kwargs, show_time_estimation=False)
    df = study.df
    assert study.is_wtc_computed == True
    assert len(dyad1.wtcs) == 1
    
    # dyads shuffle are computed only when we want significance
    assert study.is_wtc_shuffle_computed == False
    assert study.dyads_shuffled is None
    assert np.all(df['is_pseudo'] == False)

    study.compute_wtcs_shuffle(**wtcs_kwargs)
    df_with_shuffle = study.df
    assert len(study.dyads_shuffled) == len(dyads)*(len(dyads)-1)

    assert study.is_wtc_shuffle_computed == True
    assert len(study.dyads_shuffled[0].wtcs) == 1
    assert len(df_with_shuffle['is_pseudo'].unique()) == 2

def test_study_is_pseudo_no_duplicate():
    recordings = get_test_recordings(4)

    study = Study([Dyad(recordings[0], recordings[1]), Dyad(recordings[2], recordings[3])])
    wtcs_kwargs = dict(ch_match=get_test_ch_match_one())
    study.compute_wtcs(**wtcs_kwargs, show_time_estimation=False, with_intra=False)
    assert len(study.df) == 2
    study.compute_wtcs_shuffle(**wtcs_kwargs)
    assert len(study.df) == 4

    study.reset()

    study.compute_wtcs(**wtcs_kwargs, show_time_estimation=False, with_intra=True)
    assert len(study.df) == 6
    assert len(study.df[study.df['is_intra']==True]) == 4
    study.compute_wtcs_shuffle(**wtcs_kwargs)
    # should not have more "is_intra"
    assert len(study.df) == 8
    assert len(study.df[study.df['is_intra']==True]) == 4


def test_dyad_computes_whole_record_by_default():
    recording = get_test_recording()
    dyad = Dyad(recording, recording)
    dyad.compute_wtcs(ch_match=get_test_ch_match_one())
    assert len(dyad.wtcs) == 1

def test_dyad_does_not_compute_tasks_when_epochs_not_loaded():
    recording = Recording(tasks=[Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT)])
    recording.load_file(snirf_file1, preprocess=False)
    dyad = Dyad(recording, recording)
    with pytest.raises(Exception):
        # This should raise an exception, since the epochs have not been loaded from annotations
        dyad.compute_wtcs(ch_match=get_test_ch_match_one(), only_time_range=(0,10))

def test_dyad_coherence_pandas():
    s1, s2 = get_test_recordings()
    dyad = Dyad(s1, s2)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few())
    df = dyad._get_coherence_df()
    assert len(df['task'].unique()) == len(dyad.s1.task_keys)
    assert len(df['channel1'].unique()) == 2
    assert len(df['channel2'].unique()) == 2
    assert len(df['dyad'].unique()) == 1
    assert df['subject1'].unique()[0] == s1.subject_label
    assert df['subject2'].unique()[0] == s2.subject_label

def test_dyad_coherence_pandas_with_intra():
    s1, s2 = get_test_recordings()
    dyad = Dyad(s1, s2)

    with pytest.raises(Exception):
        dyad.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=False)
        # Since intra recording is not computed, it should raise
        dyad._get_coherence_df(with_intra=True)

    dyad.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=True)
    df = dyad._get_coherence_df(with_intra=True)
    assert len(df['task'].unique()) == len(dyad.s1.task_keys)
    assert len(df['dyad'].unique()) == 3
    assert len(df['is_intra'].unique()) == 2

def test_study_coherence_pandas():
    s1, s2 = get_test_recordings()
    s3, _ = get_test_recordings()
    dyad1 = Dyad(s1, s2, label='dyad1')
    dyad2 = Dyad(s1, s3, label='dyad2')
    dyad3 = Dyad(s2, s3, label='dyad3')
    study = Study([dyad1, dyad2, dyad3])
    study.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=False, show_time_estimation=False)
    df = study.df
    assert len(df['task'].unique()) == len(dyad1.s1.task_keys)
    assert len(df['channel1'].unique()) == 2
    assert len(df['channel2'].unique()) == 2
    assert len(df['dyad'].unique()) == 3
    assert np.all(df['is_intra'] == False)

def test_study_coherence_pandas_with_intra():
    study = Study([Dyad(*get_test_recordings(), label='dyad1')])
    study.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=True, show_time_estimation=False)
    df = study.df
    assert len(df['is_intra'].unique()) == 2


def test_dyad_coherence_pandas_on_roi():
    s1, s2 = get_test_recordings()
    dyad = Dyad(s1, s2)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few())
    df = dyad._get_coherence_df()
    assert len(df['roi1']) == 4
    assert len(df['roi2']) == 4

def test_wtc_downsampling():
    recording = get_test_recording()
    dyad = Dyad(recording, recording)
    n = 100
    Study([dyad]).compute_wtcs(ch_match=get_test_ch_match_one(), downsample=n)
    assert len(dyad.wtcs[0].times) <= n
    

def test_save_study_to_disk():
    subject = get_test_recording()
    dyad = Dyad(subject, subject)
    study = Study([dyad])

    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name
        study.save_pickle(file_path)
        study_reloaded = Study.from_pickle(file_path)
    
    assert len(study_reloaded.dyads) == len(study.dyads)

def test_save_study_df_to_disk():
    recording = get_test_recording()
    dyad = Dyad(recording, recording)
    study = Study([dyad])
    study.compute_wtcs(show_time_estimation=False)

    with tempfile.NamedTemporaryFile(suffix='.feather') as temp_file:
        file_path = temp_file.name
        study.save_feather(file_path)
        df = CoherenceDataFrame.from_feather(file_path)
    
    assert np.all(df['subject1'] == recording.subject_label)
    
def test_study_run_estimation(capsys):
    recording = get_test_recording()
    dyads = []
    for _ in range(10):
        dyads.append(Dyad(recording, recording))
    study = Study(dyads)
    study.estimate_wtcs_run_time()
    out = capsys.readouterr()
    assert 'time' in str(out)

def test_lionirs_channel_grouping():
    roi_file_path = 'data/NIRS/lionirs/channel_grouping_7ROI.mat'
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

def test_ordered_recording_ch_names():
    roi_file_path = 'data/NIRS/lionirs/channel_grouping_7ROI.mat'
    croi = ChannelROI.from_lionirs_file(roi_file_path)
    recording = Recording(channel_roi=croi).load_file(snirf_file1)
    ch_names = recording.ordered_ch_names
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

