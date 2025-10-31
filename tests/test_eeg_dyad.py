import pytest
import os

import mne
import numpy as np

from hypyp.dyad import Dyad
from hypyp.eeg.eeg_dyad import EEGDyad, PREPROCESS_STEP_ICA_FIT, PREPROCESS_STEP_RAW
from hypyp.signal.synthetic_signal import SyntheticSignal
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

def test_dyad_factory_from_raws():
    template_dyad = get_test_dyad()
    dyad = Dyad.from_eeg_raws(template_dyad.raw1, template_dyad.raw2)
    assert np.all(dyad.raw1.get_data() == template_dyad.raw1.get_data())
    assert np.all(dyad.raw2.get_data() == template_dyad.raw2.get_data())


def test_dyad_epochs_merged():
    dyad = get_test_dyad()
    epochs = dyad.epochs_merged
    assert len(epochs) == 10
    assert epochs.get_data(copy=False).shape[:2] == (10, 4)
    assert epochs.ch_names[0] == 'ch1_S1'
    #assert epochs


@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_raw_merge,
   Dyad.from_eeg_raw_merge,
])  
def test_dyad_from_raw_merge(dyad_factory):
    base_dyad = get_test_dyad()
    raw1 = base_dyad.raw1
    raw2 = base_dyad.raw2
    merged_data = np.vstack([raw1.get_data(), raw2.get_data()])
    ch_names = [f"{ch}_S1" for ch in raw1.ch_names] + [f"{ch}_S2" for ch in raw2.ch_names]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, raw1.info['sfreq'], ch_types)
    raw = mne.io.RawArray(merged_data, info)

    new_dyad = dyad_factory(raw)
    assert np.all(new_dyad.raw1.get_data() == base_dyad.raw1.get_data())
    assert np.all(new_dyad.raw2.get_data() == base_dyad.raw2.get_data())

@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_epochs,
   Dyad.from_eeg_epochs,
])  
def test_dyad_create_from_epochs(dyad_factory):
    epo_template = get_test_dyad().epo1
    epos = [
        generate_random_epoch(epo_template),
        generate_random_epoch(epo_template),
    ]
    dyad = dyad_factory(*epos)
    assert dyad.epo1.get_data(copy=False).shape == epos[0].get_data(copy=False).shape
    assert dyad.epo2.get_data(copy=False).shape == epos[0].get_data(copy=False).shape

@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_files,
   Dyad.from_eeg_files,
])  
def test_dyad_create_from_epo_file(dyad_factory):
    dyad: EEGDyad = dyad_factory(epo_file1, epo_file2)
    assert dyad.epo1.get_data(copy=False).shape == dyad.epo2.get_data(copy=False).shape

@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_files,
   Dyad.from_eeg_files,
])  
def test_prep_ica_fit(dyad_factory):
    dyad: EEGDyad = dyad_factory(epo_file1, epo_file2)
    assert dyad.icas == None
    dyad.prep_ica_fit(2)
    assert len(dyad.icas) == 2

@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_files,
   Dyad.from_eeg_files,
])  
def test_prep_ica_apply(dyad_factory):
    dyad: EEGDyad = dyad_factory(epo_file1, epo_file2)
    dyad.prep_ica_fit(2)
    subject_idx = 0
    component_idx = 0

    data_before = dyad.epochs_merged.get_data(copy=False)
    dyad.prep_ica_apply(subject_idx, component_idx, threshold=0.01, label='dummy')
    data_after = dyad.epochs_merged.get_data(copy=False)

    assert len(dyad.ica1.labels_['dummy']) > 0
    # if some component has been removed, we should have a lower amplitude
    assert np.sum(np.abs(data_before)) > np.sum(np.abs(data_after))

@pytest.mark.parametrize('dyad_factory', [
   EEGDyad.from_files,
   Dyad.from_eeg_files,
])  
def test_pipeline_track_steps(dyad_factory):
    dyad: EEGDyad = dyad_factory(epo_file1, epo_file2)
    assert len(dyad.steps) == 1
    assert dyad.steps[0].name == PREPROCESS_STEP_RAW

    assert dyad.epos == dyad.steps[-1].epos
    dyad.prep_ica_fit(2)
    dyad.prep_ica_apply(0, 0, label='dummy')
    assert len(dyad.steps) == 2
    # Make sure we kept all the stages
    assert dyad.epos == dyad.steps[-1].epos

def test_prep_autoreject():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    # Truncate to 20 epochs to run faster
    dyad.epos_add_step([epo[:20] for epo in dyad.epos])
    dyad.prep_autoreject()
    assert dyad.dic_ar['dyad'] > 0

def test_analyse_pow_average():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.psds is None
    dyad.analyse_pow(8, 12)
    assert len(dyad.psds) == 2
    assert dyad.psd1.freqs[0] == 8
    assert dyad.psd1.freqs[-1] == 12
    assert len(dyad.psd1.psd.shape) == 2
    assert len(dyad.psd1.ch_names) == len(dyad.epo1.ch_names)

def test_analyse_pow_not_average():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.psds is None
    dyad.analyse_pow(8, 12, epochs_average=False)
    assert len(dyad.psds) == 2
    assert dyad.psd1.freqs[0] == 8
    assert dyad.psd1.freqs[-1] == 12
    assert len(dyad.psd1.psd.shape) == 3
    assert len(dyad.psd1.ch_names) == len(dyad.epo1.ch_names)


@pytest.mark.parametrize('mode', [
   'plv',
   'envelope_corr',
   'pow_corr',
   'coh',
   'imaginary_coh',
   'ccorr',  
   'pli',
   'wpli',
])  
def test_analyse_connectivity(mode):
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    dyad.compute_complex_signal_freq_bands()
    dyad.analyse_connectivity(mode)
    assert dyad.connectivities[mode] is not None
    conn = dyad.connectivities[mode]
    assert conn.mode == mode
    n_ch = len(dyad.epo1.ch_names)
    # TODO improve adressing
    assert conn.inter[0].values.shape == (n_ch, n_ch)
    assert conn.intra1[0].values.shape == (n_ch, n_ch)
    assert conn.intra2[0].values.shape == (n_ch, n_ch)

    assert conn.intra1[0].ch_names[0] == dyad.epo1.ch_names

def test_factory_class():
    dyad = Dyad.from_eeg_files(epo_file1, epo_file2)

#def test_frequency_bands():
#    assert 'TODO' == True