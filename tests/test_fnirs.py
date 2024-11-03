import pytest
import warnings
import logging

from hypyp.fnirs.subject_fnirs import SubjectFNIRS
from hypyp.fnirs.dyad_fnirs import DyadFNIRS
from hypyp.fnirs.data_loader_fnirs import DataLoaderFNIRS

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

logging.disable()

# Try to load every file types we have
@pytest.mark.parametrize("file_path", fnirs_files)
def test_data_loader(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = DataLoaderFNIRS().get_mne_raw(file_path)
    assert raw.info['sfreq'] > 0
    assert len(raw.ch_names) > 0
    

def test_subject():
    # filename does not end with _raw.fif
    # ignore the warning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    s1 = SubjectFNIRS()
    s1.load_file(DataLoaderFNIRS(), fif_file)
    assert s1.filepath == fif_file
    assert s1.raw is not None

    s1.best_ch_names = ch_list_s1

    s1.load_epochs(tmin, tmax, baseline)
    assert s1.events is not None
    assert s1.event_dict is not None
    assert s1.epochs is not None
    assert len(s1.epochs.ch_names) == len(ch_list_s1)

def test_instanciate():
    # filename does not end with _raw.fif
    # ignore the warning
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    loader = DataLoaderFNIRS()

    s1 = SubjectFNIRS()
    s1.load_file(loader, fif_file)
    s1.set_best_ch_names(ch_list_s1)

    s2 = SubjectFNIRS()
    s2.load_file(loader, fif_file)
    s2.set_best_ch_names(ch_list_s2)

    dyad = DyadFNIRS(s1, s2)
    assert dyad.s1 is not None
    assert dyad.s2 is not None

    dyad.load_epochs(tmin, tmax, baseline)

    assert dyad.s1.events is not None
    assert dyad.s1.event_dict is not None
    assert dyad.s1.epochs is not None
    assert len(dyad.s1.epochs.ch_names) == len(ch_list_s1)

    assert dyad.s2.events is not None
    assert dyad.s2.event_dict is not None
    assert dyad.s2.epochs is not None
    assert len(dyad.s2.epochs.ch_names) == len(ch_list_s2)

def test_list_paths():
    loader = DataLoaderFNIRS()
    paths = loader.paths
    assert len(paths) > 0

def test_list_paths():
    loader = DataLoaderFNIRS()
    previous_count = len(loader.paths)
    loader.add_source('/foo')
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

def test_list_files():
    loader = DataLoaderFNIRS()
    assert len(loader.paths) > 0
    assert len(loader.list_fif_files()) > 0
    # path should be absolute
    assert loader.list_fif_files()[0].startswith('/')
    
# Skip this test because it downloads data. We don't want this on the CI
@pytest.mark.skip(reason="Downloads data")
def test_download_demos():
    loader = DataLoaderFNIRS()
    previous_count = len(loader.paths)
    loader.download_demo_dataset()
    new_paths = loader.paths
    assert len(new_paths) == previous_count + 1

#def test_load_lionirs():



    

