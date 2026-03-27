import pytest
import warnings
import logging
import re

import numpy as np
import mne

from hypyp.fnirs.fnirs_recording import FNIRSRecording
from hypyp.dataclasses.channel_roi import ChannelROI
from hypyp.data_browser import DataBrowser
from hypyp.fnirs.fnirs_step import FNIRSStep, PREPROCESS_STEP_BASE, PREPROCESS_STEP_HAEMO_FILTERED
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
    step = FNIRSStep(raw, key, desc)
    assert step.obj.get_data().shape[0] == 1
    assert step.obj.get_data().shape[1] == 2
    assert step.name == key
    assert step.desc == desc
    assert step.duration == 2

#
# Subject Recordings
#

def test_load_from_raw():
    raw = mne.io.read_raw_snirf(snirf_file1, preload=True)
    recording = FNIRSRecording().load_raw(raw)
    assert recording.subject_label == 'default'
    assert len(recording.mne_raw.times) == len(raw.times)


def test_recording():
    filepath = snirf_file1
    recording = FNIRSRecording(subject_label='my_subject')
    recording.load_file(filepath, MnePreprocessorRawToHaemo(), preprocess=False)
    assert recording.subject_label == 'my_subject'
    assert recording.filepath == filepath
    assert recording.mne_raw is not None
    assert len(recording.tasks) == 1 # default task, which is the complete record
    assert recording.preprocess_steps is None
    assert recording.is_preprocessed == False
    assert recording.epochs_per_task is None # need preprocessing to extract epochs

def test_recording_load_participant_name():
    filepath = snirf_file1
    recording = FNIRSRecording()
    recording.load_file(filepath, preprocess=False)
    assert recording.subject_label == 'default'

def test_recording_tasks():
    recording = FNIRSRecording(tasks=[Task('my_task_in_time', onset_time=1, duration=2)])
    assert len(recording.tasks) == 1
    assert recording.task_keys[0] == 'my_task_in_time'

def test_recording_epochs():
    tasks = [
        Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT),
        Task('task2', onset_event_id=3, offset_event_id=TASK_NEXT_EVENT),
    ]
    recording = FNIRSRecording(tasks=tasks)
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
    recording = FNIRSRecording(tasks=tasks)
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
    recording = FNIRSRecording(tasks=tasks)
    recording.load_file(snirf_file1, MnePreprocessorRawToHaemo())
    epochs = recording.get_epochs_for_task('task1')
    assert len(epochs) == 3
    n_events = epochs.events.shape[0]
    assert n_events == 3
    

def test_upstream_preprocessor():
    recording = FNIRSRecording(tasks=[Task('task1', onset_event_id=1, offset_event_id=TASK_NEXT_EVENT)]).load_file(snirf_file1, MnePreprocessorAsIs())
    assert len(recording.preprocess_steps) == 1
    assert recording.is_preprocessed == True
    assert recording.preprocess_steps[0].name == PREPROCESS_STEP_BASE
    assert recording.epochs_per_task is not None
    assert len(recording.epochs_per_task) > 0

def test_mne_preprocessor():
    preprocessor = MnePreprocessorRawToHaemo()
    recording = FNIRSRecording().load_file(snirf_file1, preprocessor, preprocess=False)
    recording.preprocess(preprocessor)
    assert len(recording.preprocess_steps) > 1
    assert recording.is_preprocessed == True
    assert recording.preprocess_steps[0].name == PREPROCESS_STEP_BASE
    assert recording.preprocess_steps[-1].name == PREPROCESS_STEP_HAEMO_FILTERED
    assert recording.preprocessed is not None
    assert recording.preprocess_step_keys[0] == recording.preprocess_steps[-1].name

    # can get step by key
    assert recording.get_preprocess_step(PREPROCESS_STEP_BASE).name == PREPROCESS_STEP_BASE
    assert recording.get_preprocess_step(PREPROCESS_STEP_HAEMO_FILTERED).name == PREPROCESS_STEP_HAEMO_FILTERED

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
    recording = FNIRSRecording(channel_roi=croi).load_file(snirf_file1)
    ch_names = recording.ordered_ch_names
    assert ch_names[0] == 'S2_D2 760'


def test_positions_to_standard_montage():
    #recording = get_test_recording()
    recording = FNIRSRecording().load_file('data/NIRS/slow_breathing.snirf')
    map = recording.get_channel_to_standard_montage_map(as_data_frame=False)
    assert map['S1']['name'] == 'AF7'
    assert map['S1_D2']['name'] == 'AF7'

def test_positions_to_standard_montage_as_data_frame():
    #recording = get_test_recording()
    recording = FNIRSRecording().load_file(snirf_file1)
    df = recording.get_channel_to_standard_montage_map()
    assert df[df.columns[0]][0] == 'S1'


# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    browser = DataBrowser()
    previous_count = len(browser.paths)
    browser.download_demo_fnirs_dataset()
    new_paths = browser.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():

