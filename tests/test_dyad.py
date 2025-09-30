import pytest
import os

import mne
import numpy as np

from hypyp.eeg_classes.eeg_dyad import EEGDyad
from hypyp.signal import SyntheticSignal
from hypyp.utils import generate_random_epoch

epo_file1 = os.path.join("data", "participant1-epo.fif")
epo_file2 = os.path.join("data", "participant2-epo.fif")

def get_test_dyad() -> EEGDyad:
    duration = 10.1
    sfreq = 100
    n = int(duration * sfreq)
    ch_names = ['ch1', 'ch2']
    ch_types = ['eeg'] * len(ch_names)

    data1 = np.array([SyntheticSignal(duration, n).add_noise().y for _ in range(len(ch_names))])
    data2 = np.array([SyntheticSignal(duration, n).add_noise().y for _ in range(len(ch_names))])

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    print(data1.shape)

    raw1 = mne.io.RawArray(data1, info)
    raw2 = mne.io.RawArray(data2, info)
    return EEGDyad.from_raws(raw1, raw2)


def test_dyad():
    dyad = get_test_dyad()
    assert len(dyad.raws) == 2

def test_dyad_epochs():
    dyad = get_test_dyad()
    dyad.create_epochs_from_raws(duration=1)
    assert len(dyad.epo1) == 10
    assert len(dyad.epo2) == 10

    dyad.create_epochs_from_raws(duration=2)
    assert len(dyad.epo1) == 5
    assert len(dyad.epo2) == 5

def test_dyad_epochs_merged():
    dyad = get_test_dyad()
    epochs = dyad.epochs_merged
    assert len(epochs) == 10
    assert epochs.get_data(copy=False).shape[:2] == (10, 4)
    assert epochs.ch_names[0] == 'ch1_S1'
    #assert epochs

def test_dyad_from_raw_merge():
    base_dyad = get_test_dyad()
    raw1 = base_dyad.raw1
    raw2 = base_dyad.raw2
    merged_data = np.vstack([raw1.get_data(), raw2.get_data()])
    ch_names = [f"{ch}_S1" for ch in raw1.ch_names] + [f"{ch}_S2" for ch in raw2.ch_names]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, raw1.info['sfreq'], ch_types)
    raw = mne.io.RawArray(merged_data, info)

    new_dyad = EEGDyad.from_raw_merge(raw)
    assert np.all(new_dyad.raw1.get_data() == base_dyad.raw1.get_data())
    assert np.all(new_dyad.raw2.get_data() == base_dyad.raw2.get_data())

def test_dyad_create_from_epochs():
    epo_template = get_test_dyad().epo1
    epos = [
        generate_random_epoch(epo_template),
        generate_random_epoch(epo_template),
    ]
    dyad = EEGDyad.from_epochs(*epos)
    assert dyad.epo1.get_data(copy=False).shape == epos[0].get_data(copy=False).shape
    assert dyad.epo2.get_data(copy=False).shape == epos[0].get_data(copy=False).shape

def test_dyad_create_from_epo_file():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.epo1.get_data(copy=False).shape == dyad.epo2.get_data(copy=False).shape

def test_prep_ica_fit():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.icas == None
    dyad.prep_ica_fit(2)
    assert len(dyad.icas) == 2

def test_prep_ica_apply():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    dyad.prep_ica_fit(2)
    subject_idx = 0
    component_idx = 0

    data_before = dyad.epochs_merged.get_data()
    dyad.prep_ica_apply(subject_idx, component_idx, threshold=0.01, label='dummy')
    data_after = dyad.epochs_merged.get_data()

    assert len(dyad.ica1.labels_['dummy']) > 0
    # if some component has been removed, we should have a lower amplitude
    assert np.sum(np.abs(data_before)) > np.sum(np.abs(data_after))

def test_pipeline_track_steps():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert len(dyad.steps) == 1

    assert dyad.epos == dyad.steps[-1].epos
    dyad.prep_ica_fit(2)
    dyad.prep_ica_apply(0, 0, label='dummy')
    assert len(dyad.steps) == 2
    # Make sure we kept all the stages
    assert dyad.epos == dyad.steps[-1].epos

def test_prep_autoreject():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    # Truncate to 20 epochs to run faster
    dyad.epos = [epo[:20] for epo in dyad.epos]
    dyad.prep_autoreject()
    assert dyad.dic_ar['dyad'] > 0

def test_analyse_pow():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.psds is None
    dyad.analyse_pow(8, 12)
    assert len(dyad.psds) == 2
    assert dyad.psd1.freqs[0] == 8
    assert dyad.psd1.freqs[-1] == 12

def test_analyse_connectivity():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    dyad.analyse_connectivity_ccorr()
    assert dyad.connectivity['ccorr'] is not None
    conn = dyad.connectivity['ccorr']
    assert conn.mode == 'ccorr'
    n_ch = len(dyad.epo1.ch_names)
    # TODO improve adressing
    assert conn.inter[0].values.shape == (n_ch, n_ch)
    assert conn.intras[0][0].values.shape == (n_ch, n_ch)
