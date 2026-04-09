import pytest
from unittest.mock import patch
import logging
import re
import tempfile

import numpy as np

from hypyp.fnirs.fnirs_study import FNIRSStudy
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame
from hypyp.wavelet.implementations.pywavelets_wavelet import ComplexMorletWavelet
from hypyp.fnirs.fnirs_recording import FNIRSRecording
from hypyp.fnirs.fnirs_dyad import FNIRSDyad
from hypyp.fnirs.preprocessor.implementations.mne_preprocessor_raw_to_haemo import MnePreprocessorRawToHaemo
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
    return FNIRSRecording(tasks=tasks).load_file(snirf_file1)

def get_test_recordings(count:int=2):
    tasks = [Task('task1', onset_time=0, duration=20)]
    recordings = [FNIRSRecording(tasks=tasks, subject_label=f's{i+1}').load_file(snirf_file1) for i in range(count)]
    return recordings

def get_test_ch_match_one():
    return 'S1_D1 760'

def get_test_ch_match_few():
    return re.compile(r'^S1_.*760')

#
# Dyad
#

def test_recording_dyad():
    s1 = FNIRSRecording().load_file(snirf_file1, preprocess=False)
    s2 = FNIRSRecording().load_file(snirf_file2, preprocess=False)
    dyad = FNIRSDyad(s1, s2)
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
    s1 = FNIRSRecording(tasks=tasks).load_file(snirf_file1)
    s2 = FNIRSRecording(tasks=tasks).load_file(snirf_file2)
    dyad = FNIRSDyad(s1, s2)

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
    s1 = FNIRSRecording(tasks=[tasks[0], tasks[1]])
    s2 = FNIRSRecording(tasks=[tasks[1], tasks[2]])
    assert len(FNIRSDyad(s1, s1).tasks) == 2
    assert len(FNIRSDyad(s2, s2).tasks) == 2
    assert len(FNIRSDyad(s1, s2).tasks) == 1
    
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
    s1 = FNIRSRecording(tasks=tasks1)
    s2 = FNIRSRecording(tasks=tasks2)
    merged_task_names = [task.name for task in FNIRSDyad(s1, s2).tasks]
    assert 'my_task1' in merged_task_names
    assert 'my_task2' in merged_task_names
    assert 'my_task3' in merged_task_names
    assert 'different_name' not in merged_task_names
    assert 'unknown_task' not in merged_task_names

def test_dyad_check_sfreq_same():
    s1 = FNIRSRecording().load_file(snirf_file1)
    s1.preprocessed.resample(sfreq=5)
    s2 = FNIRSRecording().load_file(snirf_file2)
    dyad = FNIRSDyad(s1, s2)
    with pytest.raises(Exception):
        dyad.get_pairs(dyad.s1, dyad.s2)
    
def test_dyad_compute_pair_wtc():
    # test with the same subject, so we can check we have a high coherence
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    pair = dyad.get_pairs(dyad.s1, dyad.s2)[0].sub((0, 10)) # Take 10% of the file
    wtc = ComplexMorletWavelet().wtc(pair)
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(wtc.W) == pytest.approx(1)
    assert wtc.label_dyad == dyad.label

def test_dyad_cwt_cache_during_wtc():
    s1, s2 = get_test_recordings()
    dyad = FNIRSDyad(s1, s2)
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
    dyad = FNIRSDyad(s1, s2)
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
    s1 = FNIRSRecording(subject_label='subject1', tasks=[Task('my_task', onset_time=0, duration=10)]).load_file(snirf_file1)
    s2 = FNIRSRecording(subject_label='subject2', tasks=[Task('my_task', onset_time=0, duration=20)]).load_file(snirf_file2)
    # Force a different task length for subject2 to have a different lenght
    dyad = FNIRSDyad(s1, s1)
    dyad.s2 = s2 # hack for the test
    dyad.is_intra = False # hack for the test
    dyad.compute_wtcs(ch_match=get_test_ch_match_one(), with_intra=True)

def test_dyad_compute_all_wtc():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    assert dyad.is_wtc_computed == False
    dyad.compute_wtcs(only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == len(recordings[0].preprocessed.ch_names)**2
    # Should have a mean of 1 since the first pair is the same signal
    assert np.mean(dyad.wtcs[0].W) == pytest.approx(1)

    assert len(dyad.df['channel1'].unique()) == 32
    
def test_dyad_computes_intra_subject():
    s1 = FNIRSRecording().load_file(snirf_file1)
    s2 = FNIRSRecording().load_file(snirf_file2)
    dyad = FNIRSDyad(s1, s2)
    dyad.compute_wtcs(only_time_range=(0,10), with_intra=True)
    assert s1.is_wtc_computed == True
    assert s2.is_wtc_computed == True
    assert len(dyad.wtcs) == len(s1.intra_wtcs)
    assert len(dyad.wtcs) == len(s2.intra_wtcs)
    
def test_dyad_computes_intra_subject_channel_match():
    s1 = FNIRSRecording().load_file(snirf_file1)
    s2 = FNIRSRecording().load_file(snirf_file2)
    dyad = FNIRSDyad(s1, s2)
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
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    dyad.compute_wtcs(ch_match='760', only_time_range=(0,10))
    assert dyad.is_wtc_computed == True
    assert len(dyad.wtcs) == (len(recordings[0].preprocessed.pick('all').ch_names)/2)**2

def test_dyad_compute_regex_match_wtc():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few(), only_time_range=(0,10))
    assert len(dyad.wtcs) == 4
    assert dyad.wtcs[0].label_pair == dyad.get_pairs(dyad.s1, dyad.s2)[0].label

def test_dyad_compute_tuple_match_wtc():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    regex1 = re.compile(r'^S1_D1.*760')
    regex2 = re.compile(r'.*760')
    dyad.compute_wtcs(ch_match=(regex1, regex2), only_time_range=(0,10))
    assert len(dyad.wtcs) == 16
    #[print(wtc.label) for wtc in dyad.wtcs]

def test_dyad_compute_list_match_wtc():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    ch_list = ['S1_D1 760', 'S1_D2 760', 'S2_D1 760']
    dyad.compute_wtcs(ch_match=ch_list, only_time_range=(0,10))
    assert len(dyad.wtcs) == len(ch_list) * len(ch_list)
    
def test_dyad_compute_list_per_subject_match_wtc():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    ch_list1 = ['S1_D1 760', 'S1_D2 760']
    ch_list2 = ['S2_D1 760', 'S2_D2 760']
    dyad.compute_wtcs(ch_match=(ch_list1, ch_list2), only_time_range=(0,10), with_intra=False)
    assert len(dyad.wtcs) == len(ch_list1) * len(ch_list2)
    assert len(dyad.df['channel1'].unique()) == 2

    assert ch_list1[0] in dyad.df['channel1'].unique()
    assert ch_list2[0] not in dyad.df['channel1'].unique()

    assert ch_list1[0] not in dyad.df['channel2'].unique()
    assert ch_list2[0] in dyad.df['channel2'].unique()
    

def test_dyad_is_intra_when_same_subject():
    recording = FNIRSRecording().load_file(snirf_file1)
    dyad = FNIRSDyad(recording, recording)
    assert dyad.is_intra == True
    # make sure wtcs are marked as "is_intra"

    dyad.compute_wtcs()
    assert dyad.df['is_intra'][0] == True

def test_access_cwts_after_computation():
    recording = FNIRSRecording().load_file(snirf_file1)
    dyad = FNIRSDyad(recording, recording)
    dyad.compute_wtcs()
    assert len(recording.cwts) == len(recording.mne_raw.ch_names)

def test_dyad_wtc_per_task():
    tasks = [
        Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT), # these 2 events have different duration
        Task('task3', onset_event_id=3, offset_event_id=TASK_NEXT_EVENT),
    ]
    recording1 = FNIRSRecording(tasks=tasks).load_file(snirf_file1)
    recording2 = FNIRSRecording(tasks=tasks).load_file(snirf_file2)
    dyad = FNIRSDyad(recording1, recording2)
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
    recording1 = FNIRSRecording(tasks=tasks)
    recording1.load_file(snirf_file1)
    recording2 = FNIRSRecording(tasks=tasks)
    recording2.load_file(snirf_file2)

    dyad = FNIRSDyad(recording1, recording2)
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
    s1 = FNIRSRecording(tasks=tasks).load_file(snirf_file1)
    s2 = FNIRSRecording(tasks=tasks).load_file(snirf_file2)
    dyad = FNIRSDyad(s1, s2)

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


    assert dyad.wtcs[0].times[0] == 0
    # Make sure wtcs for sections keep the original times
    assert dyad.wtcs[1].times[0] != 0
    assert dyad.wtcs[1].times[0] > dyad.wtcs[0].times[-1]
    assert dyad.wtcs[2].times[0] > dyad.wtcs[1].times[-1]

    
    mat = dyad.get_all_wtcs_as_ts_matrix(dyad.wtcs)
    assert mat.shape[0] == len(dyad.wtcs)

    synchronies = dyad.get_synchrony_time_series()
    # We should have rows for "frequency range" and cols for time series
    assert len(synchronies.by_condition['task1'].time_series_per_range.shape) == 2
    assert synchronies.by_condition['task1'].time_series_per_range.shape[0] == 1
    assert np.nanmean(synchronies.by_condition['task1'].time_series_per_range) > 0

def test_study_wtc():
    s1, s2 = get_test_recordings()
    s3, s4 = get_test_recordings()
    s5, s6 = get_test_recordings()
    dyad1 = FNIRSDyad(s1, s2)
    dyad2 = FNIRSDyad(s3, s4)
    dyad3 = FNIRSDyad(s5, s6)

    # Add a bunch of "dyad3" to our list, so we have a number of "others" for our first dyad
    dyads = [dyad1, dyad2, dyad3, dyad3, dyad3, dyad3]
    study = FNIRSStudy(dyads)
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

    study = FNIRSStudy([FNIRSDyad(recordings[0], recordings[1]), FNIRSDyad(recordings[2], recordings[3])])
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
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    dyad.compute_wtcs(ch_match=get_test_ch_match_one())
    assert len(dyad.wtcs) == 1

def test_dyad_does_not_compute_tasks_when_epochs_not_loaded():
    recording = FNIRSRecording(tasks=[Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT)])
    recording.load_file(snirf_file1, preprocess=False)
    dyad = FNIRSDyad(recording, recording)
    with pytest.raises(Exception):
        # This should raise an exception, since the epochs have not been loaded from annotations
        dyad.compute_wtcs(ch_match=get_test_ch_match_one(), only_time_range=(0,10))

def test_dyad_coherence_pandas():
    s1, s2 = get_test_recordings()
    dyad = FNIRSDyad(s1, s2)
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
    dyad = FNIRSDyad(s1, s2)

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
    dyad1 = FNIRSDyad(s1, s2, label='dyad1')
    dyad2 = FNIRSDyad(s1, s3, label='dyad2')
    dyad3 = FNIRSDyad(s2, s3, label='dyad3')
    study = FNIRSStudy([dyad1, dyad2, dyad3])
    study.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=False, show_time_estimation=False)
    df = study.df
    assert len(df['task'].unique()) == len(dyad1.s1.task_keys)
    assert len(df['channel1'].unique()) == 2
    assert len(df['channel2'].unique()) == 2
    assert len(df['dyad'].unique()) == 3
    assert np.all(df['is_intra'] == False)

def test_study_coherence_pandas_with_intra():
    study = FNIRSStudy([FNIRSDyad(*get_test_recordings(), label='dyad1')])
    study.compute_wtcs(ch_match=get_test_ch_match_few(), with_intra=True, show_time_estimation=False)
    df = study.df
    assert len(df['is_intra'].unique()) == 2


def test_dyad_coherence_pandas_on_roi():
    s1, s2 = get_test_recordings()
    dyad = FNIRSDyad(s1, s2)
    dyad.compute_wtcs(ch_match=get_test_ch_match_few())
    df = dyad._get_coherence_df()
    assert len(df['roi1']) == 4
    assert len(df['roi2']) == 4

def test_wtc_downsampling():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    n = 100
    FNIRSStudy([dyad]).compute_wtcs(ch_match=get_test_ch_match_one(), downsample=n)
    assert len(dyad.wtcs[0].times) <= n
    

def test_save_study_to_disk():
    subject = get_test_recording()
    dyad = FNIRSDyad(subject, subject)
    study = FNIRSStudy([dyad])

    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name
        study.save_pickle(file_path)
        study_reloaded = FNIRSStudy.from_pickle(file_path)
    
    assert len(study_reloaded.dyads) == len(study.dyads)

def test_save_study_df_to_disk():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    study = FNIRSStudy([dyad])
    study.compute_wtcs(show_time_estimation=False, with_intra=False)

    with tempfile.NamedTemporaryFile(suffix='.feather') as temp_file:
        file_path = temp_file.name
        study.save_feather(file_path)
        df = CoherenceDataFrame.from_feather(file_path)

    assert np.all(df['subject1'] == recordings[0].subject_label)
    
def test_study_run_estimation(capsys):
    recordings = get_test_recordings()
    dyads = []
    for _ in range(10):
        dyads.append(FNIRSDyad(*recordings))
    study = FNIRSStudy(dyads)
    study.estimate_wtcs_run_time()
    out = capsys.readouterr()
    assert 'time' in str(out)


def test_synchrony_time_series():
    recordings = get_test_recordings()
    dyad = FNIRSDyad(*recordings)
    dyad.compute_wtcs(period_cuts=[3, 10])

    mat = dyad.get_all_wtcs_as_ts_matrix(dyad.wtcs)
    assert mat.shape[0] == len(dyad.wtcs)

    synchronies = dyad.get_synchrony_time_series()
    assert len(synchronies.by_condition['task1'].time_series_per_range.shape) == 2
    assert synchronies.by_condition['task1'].time_series_per_range.shape[0] == 3 # the cuts makes 3 "ranges"

    # Check that we mask values when we don't have a valid WTC (cone of influence)
    assert np.isnan(synchronies.by_condition['task1'].time_series_per_range[0, 0]) == True
    assert np.isnan(synchronies.by_condition['task1'].time_series_per_range[0, -1]) == True
    assert np.isnan(synchronies.by_condition['task1'].time_series_per_range[0, 100]) == False
    assert np.isnan(synchronies.by_condition['task1'].time_series_per_range[-1, 100]) == True

def test_synchrony_time_series_perfect_coherence():
    import matplotlib.pyplot as plt
    # use the same recording to have perfect coherence,
    # but load it twice, otherwise if it is the same python object, it will be computed as "intra-subject"
    s1 = get_test_recording()
    s2 = get_test_recording()
    # use twice the same recording
    dyad = FNIRSDyad(s1, s2)
    # use only one channel
    dyad.compute_wtcs(period_cuts=[3, 4, 10], ch_match=['S1_D1 850'])

    mat = dyad.get_all_wtcs_as_ts_matrix(dyad.wtcs)
    #print(mat)
    assert mat.shape[0] == len(dyad.wtcs)

    synchronies = dyad.get_synchrony_time_series()

    for freq_range, ts in synchronies.by_condition['task1'].by_freq_band.items():
        value = np.nanmean(ts)
        print(f"synchrony for freq_range: {freq_range}: {value}")

    assert np.nanmean(synchronies.by_condition['task1'].time_series_per_range[0,:]) == pytest.approx(1.0)
    assert np.nanmean(synchronies.by_condition['task1'].time_series_per_range[1,:]) == pytest.approx(1.0)
    assert np.nanmean(synchronies.by_condition['task1'].time_series_per_range[2,:]) == pytest.approx(1.0)
    # the last one is period of 10sec and above, should not have any value since it is all masked (cone of influence)
    assert np.isnan(np.mean(synchronies.by_condition['task1'].time_series_per_range[3,:]))
