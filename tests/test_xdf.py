import pytest
from hypyp.xdf import XDFImport, XDFStream

#file_path = 'data/dyad-example.xdf'
#file_path = 'data/dyad-example-synthetic.xdf'
file_path = 'data/ExampleWorkshopData/FingerTappingExample4.xdf'

def test_instanciate():
    xdf = XDFImport(file_path)
    assert xdf.file_path == file_path

    #print(xdf.available_stream_names)
    # 1 keyboard events and 2 eeg
    assert len(xdf.available_streams) == 3
    assert xdf.available_streams[0].name == 'Keyboard Events'
    assert xdf.available_streams[1].name == 'EEG_subject_1'
    assert xdf.available_streams[2].name == 'EEG_subject_2'

    assert xdf.available_streams[0].type == 'Markers'
    assert xdf.available_streams[1].type == 'EEG'
    assert xdf.available_streams[2].type == 'EEG'

    assert xdf.available_streams[0].id == 1
    assert xdf.available_streams[1].id == 2
    assert xdf.available_streams[2].id == 3

    assert xdf.map_id_to_idx[1] == 0
    assert xdf.map_id_to_idx[2] == 1
    assert xdf.map_id_to_idx[3] == 2

    assert len(xdf.raw_all.keys()) == 2
    assert xdf.montage is None

def test_instanciate_no_matching_stream():
    with pytest.raises(Exception):
        XDFImport(file_path, stream_matches=[0])

def test_xdf_stream_indices_per_type():
    # should not be able to create mne object when using stream of type Markers
    xdf = XDFImport(file_path, stream_matches=[1], convert_to_mne=False)
    assert 1 in xdf.get_stream_ids_for_type('Markers')
    assert 2 in xdf.get_stream_ids_for_type('EEG')
    assert 3 in xdf.get_stream_ids_for_type('EEG')

def test_instanciate_marker_stream():
    # should not be able to create mne object when using stream of type Markers
    xdf = XDFImport(file_path, stream_matches=[1], convert_to_mne=False)
    xdf_stream_ids = xdf.get_stream_ids_for_type('Markers')
    assert len(xdf_stream_ids) == 1
    with pytest.raises(Exception):
        xdf = XDFImport(file_path, stream_matches=xdf_stream_ids)

def test_all_eeg_streams():
    xdf = XDFImport(file_path)
    _, raw = list(xdf.raw_all.items())[0]
    assert len(raw.ch_names) > 0

def test_match_stream_by_idx():
    xdf = XDFImport(file_path, stream_matches=[2])
    assert len(list(xdf.raw_all.items())) == 1
    _name, raw = list(xdf.raw_all.items())[0]
    assert len(raw.ch_names) > 0
    
def test_match_stream_by_name():
    # first grab the name of the first channel
    xdf_tmp = XDFImport(file_path)
    key = list(xdf_tmp.raw_all.keys())[0]

    # now do the test
    xdf = XDFImport(file_path, stream_matches=[key])
    assert list(xdf.raw_all.keys())[0] == key

def test_match_stream_by_both():
    # first grab the name of the first channel
    xdf_tmp = XDFImport(file_path)
    key = list(xdf_tmp.raw_all.keys())[0]

    # now do the test
    xdf = XDFImport(file_path, stream_matches=[key, 3])
    assert len(list(xdf.raw_all.keys())) == 2

def test_match_unexistent_type():
    with pytest.raises(Exception):
        xdf = XDFImport(file_path, stream_type='foo')

def test_stream_type_to_mne_type():
    # Available mne types are : 
    #[ 'grad', 'mag', 'ref_meg', 'eeg', 'seeg', 'dbs', 'ecog', 'eog', 'emg', 'ecg', 
    # 'resp', 'bio', 'misc', 'stim', 'exci', 'syst', 'ias', 'gof', 'dipole', 'chpi',
    # 'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 'fnirs_fd_phase', 'fnirs_od', 'hbo', 'hbr',
    # 'csd', 'temperature', 'gsr', 'eyegaze', 'pupil' ]

    assert XDFStream.stream_type_to_mne_type('EEG') == 'eeg'
    assert XDFStream.stream_type_to_mne_type('fNIRS') == 'fnirs_cw_amplitude'
    assert XDFStream.stream_type_to_mne_type('markers') == 'stim'
    assert XDFStream.stream_type_to_mne_type('stim') == 'stim'

def test_stream_type_map_explicit():
    mne_force_type = 'misc'
    my_map = {'EEG': mne_force_type}
    assert XDFStream.stream_type_to_mne_type('EEG', my_map) == mne_force_type

    xdf = XDFImport(file_path, mne_type_map=my_map)
    assert xdf.available_streams[1].type == 'EEG'
    _, raw = list(xdf.raw_all.items())[0]
    assert raw.get_channel_types()[0] == 'misc'

def test_channel_names():
    xdf = XDFImport(file_path, convert_to_mne=False)
    ch_names = xdf.available_streams[1].ch_names
    assert ch_names[0] == 'Fp1'

def test_ch_names_to_ch_types():
    ch_types = XDFStream.get_mne_ch_types('EEG', ['Fp1', 'Fp2', 'AccX', 'GyroX', 'QuatX'])
    assert ch_types == ['eeg', 'eeg', 'misc', 'misc', 'misc']

def test_ch_names():
    xdf = XDFImport(file_path)
    _, raw = list(xdf.raw_all.items())[0]
    assert raw.ch_names[0] == 'Fp1'

def test_ch_types_misc():
    xdf = XDFImport(file_path)
    _, raw = list(xdf.raw_all.items())[0]
    ch_types = raw.get_channel_types()

    assert ch_types[0] == 'eeg'
    assert ch_types[-1] == 'misc'


# test montage setup
# test fif file
# test have markers in raw
# test time of every stream is aligned
