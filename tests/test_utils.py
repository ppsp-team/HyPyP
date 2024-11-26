#!/usr/bin/env python
# coding=utf-8

import pytest
import numpy as np
import mne
from hypyp import utils
from hypyp.signal import SynteticSignal
from hypyp.wavelet.pywavelets_wavelet import PywaveletsWavelet

def get_fake_raw():
    sfreq = 100
    duration = 10
    n_channels = 5
    n_samples = int(sfreq * duration)
    data = np.random.randn(n_channels, n_samples)

    ch_names = [f'Foo {i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

def test_epochs_per_task():
    raw = get_fake_raw()
    # Define events using annotations (e.g., at 2s, 5s, and 8s with a duration of 0s)
    onsets = [2, 5, 8, 9]
    durations = [0, 0, 0, 0]
    descriptions = np.arange(1, len(onsets)+1)
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

    tasks = [
        ('task1', 1, 2),
        ('rest', 2, 3),
        ('task2', 3, 4),
    ]

    all_epochs = utils.epochs_from_tasks_annotations(raw, tasks)
    assert len(all_epochs) == len(tasks)
    assert all_epochs[0].tmax == pytest.approx(3)
    assert all_epochs[2].tmax == pytest.approx(1)

def test_epochs_recurring_task():
    raw = get_fake_raw()
    onsets = [2, 5, 6, 9]
    durations = [0, 0, 0, 0]
    descriptions = [1, 2, 1, 2]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

    tasks = [
        ('task1', 1, 2),
    ]

    all_epochs = utils.epochs_from_tasks_annotations(raw, tasks)
    assert len(all_epochs) == 1
    assert len(all_epochs[0]) == 2

def test_epochs_recurring_task_crop_time():
    raw = get_fake_raw()
    onsets = [2, 5, 6, 8] # Have the 2nd task occurence shorter than first one
    durations = [0, 0, 0, 0]
    descriptions = [1, 2, 1, 2]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

    tasks = [
        ('task1', 1, 2),
    ]

    all_epochs = utils.epochs_from_tasks_annotations(raw, tasks)
    epochs = all_epochs[0]
    assert len(epochs.times) >= 200
    assert len(epochs.times) < 300

def test_task_start_end_combinaison():
    raw = get_fake_raw()
    # Define events using annotations (e.g., at 2s, 5s, and 8s with a duration of 0s)
    onsets = [2, 5, 8, 9]
    durations = [0, 0, 0, 0]
    descriptions = np.arange(1, len(onsets)+1)
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

    assert utils.epochs_from_tasks_annotations(raw, [('task1', 1, utils.TASK_NEXT_EVENT)])[0].tmax == pytest.approx(3)
    assert utils.epochs_from_tasks_annotations(raw, [('task1', 1, utils.TASK_END)])[0].tmax == pytest.approx(8, abs=0.01)
    assert utils.epochs_from_tasks_annotations(raw, [('task1', utils.TASK_BEGINNING, utils.TASK_NEXT_EVENT)])[0].tmax == pytest.approx(2, abs=0.01)
    assert utils.epochs_from_tasks_annotations(raw, [('task1', utils.TASK_BEGINNING, utils.TASK_END)])[0].tmax == pytest.approx(10, abs=0.01)

def test_task_from_time_range():
    raw = get_fake_raw()
    tasks = [
        ('task1', 0, 1), # task from start to 1 second
        ('task2', 3, 10), # task from 3 seconds to 5 seconds
    ]

    all_epochs = utils.epochs_from_tasks_time_range(raw, tasks)
    print(all_epochs[0])
    print(all_epochs[1])
    assert len(all_epochs) == 2
    duration = all_epochs[1].times[-1] - all_epochs[1].times[0]
    assert duration == tasks[1][2] - tasks[1][1]

def test_downsampling():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(n_points=2000).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    assert res.W.shape[0] == len(res.scales)
    assert res.W.shape[1] == len(signal.y)

    bins = 1000
    times, coif, W, factor = utils.downsample_in_time(res.times, res.coif, res.W, bins=bins)
    assert factor == len(signal.y) // bins
    assert len(times) == bins
    assert len(coif) == bins
    assert W.shape[0] == res.W.shape[0] # no change
    assert W.shape[1] == bins

def test_downsampling_low_values():
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=10, n_points=50).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    times, coif, W, factor = utils.downsample_in_time(res.times, res.coif, res.W)
    assert factor == 1
    assert len(times) == len(res.times)
    assert len(coif) == len(res.coif)
    assert W.shape[0] == res.W.shape[0]
    assert W.shape[1] == res.W.shape[1]
    
def test_downsampling_threshold():
    bins = 1000
    wavelet = PywaveletsWavelet()
    signal = SynteticSignal(tmax=10, n_points=bins+1).add_sin(1)
    res = wavelet.cwt(signal.y, signal.period)

    times, coif, W, factor = utils.downsample_in_time(res.times, res.coif, res.W, bins=bins)
    assert factor == 2
    expected_len = bins / 2 + 1
    assert len(times) == expected_len
    assert len(coif) == expected_len
    assert W.shape[1] == expected_len
    