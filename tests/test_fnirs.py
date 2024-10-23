import pytest

from hypyp.fnirs import DataLoaderFNIRS, DyadFNIRS, SubjectFNIRS


#set events
tmin = 0 
tmax = 300
baseline = (0, 0)

ch_list_s1 = ["S4_D4 hbo"] 
ch_list_s2 = ["S7_D6 hbo"]

fif_file = './data/sub-110_session-1_pre_raw.fif'
snirf_file = './data/FNIRS/DCARE_02_sub1.snirf'

def test_subject():

    s1 = SubjectFNIRS()
    s1.load_fif_file(fif_file)
    assert s1.filepath == fif_file
    assert s1.raw is not None

    s1.best_ch_names = ch_list_s1

    s1.load_epochs(tmin, tmax, baseline)
    assert s1.events is not None
    assert s1.event_dict is not None
    assert s1.epochs is not None
    assert len(s1.epochs.ch_names) == len(ch_list_s1)

def test_instanciate():
    s1 = SubjectFNIRS()
    s1.load_fif_file(fif_file)
    s1.set_best_ch_names(ch_list_s1)

    s2 = SubjectFNIRS()
    s2.load_fif_file(fif_file)
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

def test_list_files():
    print(DataLoaderFNIRS.list_all_files())
    assert len(DataLoaderFNIRS.list_all_files()) > 0
    assert len(DataLoaderFNIRS.list_fif_files()) > 0
    assert DataLoaderFNIRS.list_fif_files()[0].startswith('data')

def test_load_snirf():
    s1 = SubjectFNIRS()
    s1.load_snirf_file(snirf_file)
    #s1.set_best_ch_names(['S1_D1 760'])
    #s1.load_epochs(tmin=tmin, tmax=tmax, baseline=baseline)
    #print(s1.epochs)
    

    