import pytest
import os
import logging

import mne
import numpy as np

from hypyp.eeg.eeg_dyad import EEGDyad, PREPROCESS_STEP_ICA_APPLY, PREPROCESS_STEP_RAW
from hypyp.signal.synthetic_signal import SyntheticSignal
from hypyp.utils import generate_random_epoch
from hypyp.dataclasses.freq_band import FreqBand, FreqBands

# avoid all the output from mne
logging.disable()

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

    raw1 = mne.io.RawArray(data1, info)
    raw2 = mne.io.RawArray(data2, info)
    return EEGDyad.from_raws(raw1, raw2, label='test')

def test_dyad():
    dyad = get_test_dyad()
    assert len(dyad.raws) == 2
    assert dyad.label == 'test'
    assert dyad.is_icas_computed == False
    assert dyad.is_autoreject_applied == False
    assert dyad.is_psds_computed == False
    assert dyad.is_connectivity_computed == False

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
    dyad = EEGDyad.from_raws(template_dyad.raw1, template_dyad.raw2)
    assert np.all(dyad.raw1.get_data() == template_dyad.raw1.get_data())
    assert np.all(dyad.raw2.get_data() == template_dyad.raw2.get_data())


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
    dyad: EEGDyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.icas == None
    dyad.prep_ica_fit(2)
    assert len(dyad.icas) == 2

def test_prep_ica_apply():
    dyad: EEGDyad = EEGDyad.from_files(epo_file1, epo_file2)
    dyad.prep_ica_fit(2)
    subject_idx = 0
    component_idx = 0

    assert dyad.is_icas_computed == True
    assert len(dyad.icas_applied) == 0

    data_before = dyad.epochs_merged.get_data(copy=False)
    dyad.prep_ica_apply(subject_idx, component_idx, threshold=0.01, label='dummy')
    assert len(dyad.icas_applied) == 1
    assert dyad.icas_applied[0] == 'dummy'

    data_after = dyad.epochs_merged.get_data(copy=False)

    assert len(dyad.ica1.labels_['dummy']) > 0
    # if some component has been removed, we should have a lower amplitude
    assert np.sum(np.abs(data_before)) > np.sum(np.abs(data_after))

def test_pipeline_track_steps():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
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
    dyad.prep_autoreject_apply()
    assert dyad.dic_ar['dyad'] > 0

def test_analyse_pow_average():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.psds is None
    dyad.analyse_pow(FreqBands.from_simple_min_max(8, 12), epochs_average=True)
    assert len(dyad.psds) == 2
    assert dyad.psds1[0].freqs[0] == 8
    assert dyad.psds1[0].freqs[-1] == 12
    assert len(dyad.psds1[0].psd.shape) == 2
    assert len(dyad.psds1[0].ch_names) == len(dyad.epo1.ch_names)

def test_analyse_pow_not_average():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    assert dyad.psds is None
    dyad.analyse_pow(FreqBands.from_simple_min_max(8, 12), epochs_average=False)
    assert len(dyad.psds) == 2
    assert dyad.psds1[0].freqs[0] == 8
    assert dyad.psds1[0].freqs[-1] == 12
    assert len(dyad.psds1[0].psd.shape) == 3
    assert len(dyad.psds1[0].ch_names) == len(dyad.epo1.ch_names)


# Test all the modes
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
    assert dyad.connectivities_per_mode[mode] is not None
    conn = dyad.connectivities_per_mode[mode]
    assert conn.mode == mode
    n_ch = len(dyad.epo1.ch_names)
    assert conn.inter[0].values.shape == (n_ch, n_ch)
    assert conn.intra1[0].values.shape == (n_ch, n_ch)
    assert conn.intra2[0].values.shape == (n_ch, n_ch)

    assert conn.intra1[0].ch_names[0] == dyad.epo1.ch_names

def test_connectivity_mode_keys():
    dyad = EEGDyad.from_files(epo_file1, epo_file2)
    dyad.compute_complex_signal_freq_bands()
    dyad.analyse_connectivity('ccorr')
    dyad.analyse_connectivity('plv')

    assert dyad.connectivity_modes == ['ccorr', 'plv']

def test_frequency_bands():
    freq_bands = FreqBands({
        'Alpha-Low': [7.5, 11],
        'Alpha-High': [11.5, 13]
    })
    # test indexing
    assert freq_bands[0].fmin == 7.5
    assert freq_bands['Alpha-Low'].fmin == 7.5
    assert freq_bands['Alpha-Low'][0] == 7.5
    assert freq_bands['Alpha-Low'][1] == 11

def test_synchrony_time_series():
    dyad = get_test_dyad()
    dyad.compute_complex_signal_freq_bands()
    dyad.analyse_connectivity('plv', epochs_average=False)

    synchronies = dyad.get_synchrony_time_series()
    freq_bands = synchronies[0].freq_bands
    assert len(freq_bands) == len(dyad.complex_signal.freq_bands)
    assert synchronies[0].by_freq_band[freq_bands[0]].shape[0] == len(dyad.epo1)

def test_synchrony_time_series_for_mode():
    dyad = get_test_dyad()
    dyad.compute_complex_signal_freq_bands()
    dyad.analyse_connectivity('plv', epochs_average=False)

    synchrony = dyad.get_synchrony_time_series_for_mode('plv')
    freq_bands = synchrony.freq_bands
    assert len(freq_bands) == len(dyad.complex_signal.freq_bands)
    assert synchrony.by_freq_band[freq_bands[0]].shape[0] == len(dyad.epo1)

def test_synchrony_time_series_exceptions():
    dyad = get_test_dyad()
    dyad.compute_complex_signal_freq_bands()

    # Must raise an exception if connectivity is not computed yet
    with pytest.raises(ValueError):
        dyad.get_synchrony_time_series()

    # Must raise an exception if epochs are averaged
    dyad.analyse_connectivity('plv', epochs_average=True)
    with pytest.raises(ValueError):
        dyad.get_synchrony_time_series_for_mode('plv')

def test_synchrony_discontinuity():
    dyad = get_test_dyad()

    count_before_drop = len(dyad.epo1)
    dyad.epo1.drop(indices=[3, 4], reason='TEST')
    assert 3 not in dyad.epo1.selection
    dyad.align_epochs()

    dyad.compute_complex_signal_freq_bands()
    dyad.analyse_connectivity('plv', epochs_average=False)
    synchronies = dyad.get_synchrony_time_series()
    synchrony = synchronies[0]

    assert synchrony.time_series_per_range.shape[1] == count_before_drop
