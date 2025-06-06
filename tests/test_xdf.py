import tempfile
import pytest

from hypyp.xdf import XDFImport, XDFStream
import mne

file_path = 'data/XDF/dyad-example-noise.xdf'
file_path_with_markers = 'data/XDF/dyad-example-with-markers.xdf'

## These have been extracted with print(xdf), and are used for testing select_match by stream id and mapping
#   Stream id 1 of type 'Accelerometer' with name 'LSLOutletHS1-Accelerometer'
#   Stream id 2 of type 'EEG' with name 'LSLOutletHS1-EEG'
#   Stream id 3 of type 'Markers' with name 'LSLOutletHS1-Markers'
#   Stream id 4 of type 'Markers' with name 'LSLOutletHS2-Markers'
#   Stream id 5 of type 'Quality' with name 'LSLOutletHS2-Quality'
#   Stream id 6 of type 'Accelerometer' with name 'LSLOutletHS2-Accelerometer'
#   Stream id 7 of type 'EEG' with name 'LSLOutletHS2-EEG'
#   Stream id 8 of type 'Quality' with name 'LSLOutletHS1-Quality'
STREAM_ID_ACC_1 = 1
STREAM_ID_EEG_1 = 2
STREAM_ID_MARKERS_1 = 3
STREAM_ID_MARKERS_2 = 4
STREAM_ID_QUALITY_2 = 5
STREAM_ID_ACC_2 = 6
STREAM_ID_EEG_2 = 7
STREAM_ID_QUALITY_1 = 8

def test_instanciate():
    xdf = XDFImport(file_path)
    assert xdf.file_path == file_path

    assert len(xdf.available_streams) == 8
    assert xdf.available_streams[0].name == 'LSLOutletHS1-Markers'
    assert xdf.available_streams[1].name == 'LSLOutletHS1-Accelerometer'
    assert xdf.available_streams[2].name == 'LSLOutletHS1-Quality'
    assert xdf.available_streams[3].name == 'LSLOutletHS1-EEG'
    assert xdf.available_streams[4].name == 'LSLOutletHS2-Markers'
    assert xdf.available_streams[5].name == 'LSLOutletHS2-Accelerometer'
    assert xdf.available_streams[6].name == 'LSLOutletHS2-Quality'
    assert xdf.available_streams[7].name == 'LSLOutletHS2-EEG'

    assert xdf.available_streams[0].type == 'Markers'
    assert xdf.available_streams[1].type == 'Accelerometer'
    assert xdf.available_streams[2].type == 'Quality'
    assert xdf.available_streams[3].type == 'EEG'
    assert xdf.available_streams[4].type == 'Markers'
    assert xdf.available_streams[5].type == 'Accelerometer'
    assert xdf.available_streams[6].type == 'Quality'
    assert xdf.available_streams[7].type == 'EEG'

    assert xdf.available_streams[0].id == STREAM_ID_MARKERS_1
    assert xdf.available_streams[1].id == STREAM_ID_ACC_1
    assert xdf.available_streams[2].id == STREAM_ID_QUALITY_1
    assert xdf.available_streams[3].id == STREAM_ID_EEG_1
    assert xdf.available_streams[4].id == STREAM_ID_MARKERS_2
    assert xdf.available_streams[5].id == STREAM_ID_ACC_2
    assert xdf.available_streams[6].id == STREAM_ID_QUALITY_2
    assert xdf.available_streams[7].id == STREAM_ID_EEG_2

    assert xdf.map_id_to_idx[STREAM_ID_MARKERS_1] == 0
    assert xdf.map_id_to_idx[STREAM_ID_ACC_1] == 1
    assert xdf.map_id_to_idx[STREAM_ID_QUALITY_1] == 2
    assert xdf.map_id_to_idx[STREAM_ID_EEG_1] == 3
    assert xdf.map_id_to_idx[STREAM_ID_MARKERS_2] == 4
    assert xdf.map_id_to_idx[STREAM_ID_ACC_2] == 5
    assert xdf.map_id_to_idx[STREAM_ID_QUALITY_2] == 6
    assert xdf.map_id_to_idx[STREAM_ID_EEG_2] == 7

def test_convert_by_default():
    xdf = XDFImport(file_path, select_type='EEG')
    assert xdf.file_path == file_path
    assert len(xdf.mne_raws) == 2

def test_instanciate_no_matching_stream():
    with pytest.raises(Exception):
        # use a non-existing stream_id
        non_existing_stream_id = 0
        XDFImport(file_path, select_matches=[non_existing_stream_id])

def test_xdf_stream_indices_per_type():
    # should not be able to create mne object when using stream of type Markers
    xdf = XDFImport(file_path, convert_to_mne=False)
    assert STREAM_ID_MARKERS_1 in xdf.get_stream_ids_for_type('Markers')
    assert STREAM_ID_MARKERS_2 in xdf.get_stream_ids_for_type('Markers')
    assert STREAM_ID_EEG_1 in xdf.get_stream_ids_for_type('EEG')
    assert STREAM_ID_EEG_2 in xdf.get_stream_ids_for_type('EEG')

def test_all_eeg_streams():
    xdf = XDFImport(file_path)
    assert len(xdf.mne_raws[0].ch_names) == len(xdf.selected_signal_streams[0].ch_names)

def test_match_stream_by_idx():
    xdf = XDFImport(file_path, select_matches=[STREAM_ID_EEG_1])
    assert len(xdf.mne_raws) == 1
    assert len(xdf.mne_raws[0].ch_names) == len(xdf.selected_streams[0].ch_names)
    
def test_match_stream_by_name():
    # first grab the name of the first channel
    xdf_tmp = XDFImport(file_path)
    key = list(xdf_tmp.mne_raws_dict.keys())[0]

    # now do the test
    xdf = XDFImport(file_path, select_matches=[key])
    assert list(xdf.mne_raws_dict.keys())[0] == key

def test_match_stream_by_both():
    # first grab the name of the first channel
    xdf_tmp = XDFImport(file_path)
    key = list(xdf_tmp.mne_raws_dict.keys())[0]

    # now do the test
    xdf = XDFImport(file_path, select_matches=[key, STREAM_ID_EEG_2])
    assert len(list(xdf.mne_raws_dict.keys())) == 2

def test_match_unexistent_type():
    with pytest.raises(Exception):
        xdf = XDFImport(file_path, select_type='foo')

def test_stream_type_to_mne_type():
    # Available mne types are : 
    #[ 'grad', 'mag', 'ref_meg', 'eeg', 'seeg', 'dbs', 'ecog', 'eog', 'emg', 'ecg', 
    # 'resp', 'bio', 'misc', 'stim', 'exci', 'syst', 'ias', 'gof', 'dipole', 'chpi',
    # 'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 'fnirs_fd_phase', 'fnirs_od', 'hbo', 'hbr',
    # 'csd', 'temperature', 'gsr', 'eyegaze', 'pupil' ]

    assert XDFStream.get_default_mne_ch_type_for_stream_type('EEG') == 'eeg'
    assert XDFStream.get_default_mne_ch_type_for_stream_type('fNIRS') == 'fnirs_cw_amplitude'
    assert XDFStream.get_default_mne_ch_type_for_stream_type('markers') == 'stim'
    assert XDFStream.get_default_mne_ch_type_for_stream_type('stim') == 'stim'

def test_stream_type_map_explicit():
    mne_force_type = 'misc'
    my_map = {'EEG': mne_force_type}
    assert XDFStream.get_default_mne_ch_type_for_stream_type('EEG', my_map) == mne_force_type

    # Force EEG data as 'misc' in mne
    xdf = XDFImport(file_path, mne_type_map=my_map)
    assert xdf.available_streams[xdf.map_id_to_idx[STREAM_ID_EEG_1]].type == 'EEG'
    assert xdf.mne_raws[0].get_channel_types()[0] == 'misc'

def test_channel_names():
    xdf = XDFImport(file_path, convert_to_mne=False)
    ch_names = xdf.available_streams[xdf.map_id_to_idx[STREAM_ID_EEG_1]].ch_names
    assert ch_names[0] == 'Ch1'

def test_channel_names_standard_metadata():
    # the description of channels in the "file_path" xdf does not follow the spec here https://github.com/sccn/xdf/wiki/EEG-Meta-Data
    # let's try with another xdf file which follows the spec
    xdf = XDFImport(file_path_with_markers, convert_to_mne=False)
    ch_names = xdf.selected_signal_streams[0].ch_names
    assert ch_names[0] == 'Ch1'
    assert ch_names[1] == 'Ch2'
    assert ch_names[2] == 'Ch3'

def test_ch_names_to_ch_types():
    # look at automatic detection of mne types
    ch_types = XDFStream.get_mne_ch_types('EEG', ['Fp1', 'Fp2', 'AccX', 'GyroX', 'QuatX'])
    assert ch_types == ['eeg', 'eeg', 'misc', 'misc', 'misc']

def test_ch_types_in_raw():
    rawEEG = XDFImport(file_path, select_type='EEG').mne_raws[0]
    assert rawEEG.get_channel_types()[0] == 'eeg'

    rawAcc = XDFImport(file_path, select_type='Accelerometer').mne_raws[0]
    assert rawAcc.get_channel_types()[0] == 'misc'

def test_fif_file_for_stream():
    xdf = XDFImport(file_path)
    xdf_stream = xdf.selected_signal_streams[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_file = xdf_stream.save_to_fif_file(tmp_dir)
        raw_reloaded = mne.io.read_raw_fif(saved_file)
        assert raw_reloaded.get_data()[0][0] == xdf.mne_raws_dict[xdf_stream.name].get_data()[0][0]
    
def test_mne_compatible_stream_types():
    xdf = XDFImport(file_path, convert_to_mne=False)
    marker_streams = [stream for stream in xdf.available_streams if stream.type == 'Markers']
    assert marker_streams[0].is_mne_raw_compatible == False
    
def test_subject_unique_name():
    xdf = XDFImport(file_path, select_matches=[1], convert_to_mne=False)
    assert xdf.selected_streams[0].get_unique_name() == 'LSLOutletHS1-Accelerometer'
    assert xdf.selected_streams[0].get_unique_name(append_stream_id=True) == 'LSLOutletHS1-Accelerometer-1'
    
def test_markers():
    xdf = XDFImport(file_path_with_markers)
    keys = list(xdf.markers_dict)
    assert len(keys) == 1
    assert keys[0] == XDFImport(file_path_with_markers, select_type='Markers').selected_streams[0].name
    markers = list(xdf.markers_dict.values())[0]
    annotations = markers.as_mne_annotations()
    assert len(annotations) > 0

def test_merge_markers():
    xdf = XDFImport(file_path_with_markers)
    annotations = xdf.markers.as_mne_annotations()
    markers_dup = xdf.markers + xdf.markers
    assert len(markers_dup.as_mne_annotations()) == 2 * len(annotations)

def test_markers_timing():
    xdf = XDFImport(file_path_with_markers)
    markers = xdf.markers
    offset = markers.timestamps[0]
    annotations = markers.as_mne_annotations(offset)
    assert annotations[0]['onset'] == 0.0

    annotations_for_stream = markers.as_mne_annotations(xdf.selected_signal_streams[0].reference_time)
    assert annotations_for_stream[0]['onset'] > 0
    assert annotations_for_stream[0]['onset'] < 10
    xdf.mne_raws[0].set_annotations(annotations_for_stream)

def test_markers_negative():
    xdf = XDFImport(file_path_with_markers)
    annotations_all = xdf.markers.as_mne_annotations()
    first_time = xdf.markers.timestamps[0]
    # let's say a data stream started AFTER the first marker, we want to exclude the first markers
    annotations_partial = xdf.markers.as_mne_annotations(first_time + 2)
    assert len(annotations_partial) < len(annotations_all)

    # have a big time offset to make sure we receive no annotations
    assert len(xdf.markers.as_mne_annotations(first_time + 2000)) == 0

def test_fif_files():
    xdf = XDFImport(file_path, select_type='EEG')
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_files = xdf.save_to_fif_files(tmp_dir)
        assert len(saved_files) == 2
        # make sure we can load without an error
        _ = mne.io.read_raw_fif(saved_files[0])
        _ = mne.io.read_raw_fif(saved_files[1])
    
def test_montage():
    xdf = XDFImport(file_path, select_type='EEG')
    xdf.rename_channels(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'POz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10'])
    xdf.set_montage('standard_1020')
