import pytest
import os
import logging

import mne
import numpy as np

from hypyp.multimodal.multimodal_dyad import MultimodalDyad
from hypyp.eeg import EEGDyad
from hypyp.fnirs import FNIRSDyad, FNIRSRecording

# avoid all the output from mne
logging.disable()

epo_file1 = os.path.join('data', 'participant1-epo.fif')
epo_file2 = os.path.join('data', 'participant2-epo.fif')

snirf_file1 = os.path.join('data', 'NIRS', 'DCARE_02_sub1.snirf')
snirf_file2 = os.path.join('data', 'NIRS', 'DCARE_02_sub2.snirf')

def get_test_eeg_dyad() -> EEGDyad:
    return EEGDyad.from_files(epo_file1, epo_file2)

def get_test_fnirs_dyad():
    s1 = FNIRSRecording().load_file(snirf_file1, preprocess=False)
    s2 = FNIRSRecording().load_file(snirf_file2, preprocess=False)
    return FNIRSDyad(s1, s2)

def test_multimodal_dyad():
    eeg_dyad = get_test_eeg_dyad()
    fnirs_dyad = get_test_fnirs_dyad()
    dyad = MultimodalDyad(eeg=eeg_dyad, fnirs=fnirs_dyad)
    assert dyad.eeg == eeg_dyad
    assert dyad.fnirs == fnirs_dyad

def test_add_modality():
    eeg_dyad = get_test_eeg_dyad()
    fnirs_dyad = get_test_fnirs_dyad()
    dyad = MultimodalDyad()
    assert dyad.eeg is None
    assert dyad.fnirs is None
    dyad.add_eeg(eeg_dyad)
    assert dyad.eeg == eeg_dyad
    dyad.add_fnirs(fnirs_dyad)
    assert dyad.fnirs == fnirs_dyad

def test_generic_list_of_modalities():
    dyad = MultimodalDyad(eeg=get_test_eeg_dyad(), fnirs=get_test_fnirs_dyad())
    assert len(dyad.modalities) == 2

#def test_synchrony_time_series():
#    dyad = MultimodalDyad(eeg=get_test_eeg_dyad(), fnirs=get_test_fnirs_dyad())
#    synchronies = dyad.get_synchrony_time_series()
#    assert len(synchronies) == 2
