from typing import List
import os

import mne
import numpy as np
from pyxdf.pyxdf import StreamData

class XDFStream():
    stream: StreamData
    metadata_desc: dict
    raw: mne.io.RawArray | None
    annotations: mne.Annotations | None
    time_offset: float

    @staticmethod
    def stream_type_to_mne_ch_type(stream_type: str, type_map: dict = None):
        if type_map is not None and stream_type in type_map.keys():
            return type_map[stream_type]

        mne_types = [
            'grad', 'mag', 'ref_meg', 'eeg', 'seeg', 'dbs', 'ecog', 'eog', 'emg', 'ecg', 
            'resp', 'bio', 'misc', 'stim', 'exci', 'syst', 'ias', 'gof', 'dipole', 'chpi',
            'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 'fnirs_fd_phase', 'fnirs_od', 'hbo', 'hbr',
            'csd', 'temperature', 'gsr', 'eyegaze', 'pupil',
        ]

        type_lower = stream_type.lower()
        if type_lower in mne_types:
            return type_lower

        # A few extra cases for convenience
        if type_lower == 'fnirs':
            return 'fnirs_cw_amplitude'

        if type_lower == 'markers' or type_lower == 'stim':
            return 'stim'

        # return the stream type received, maybe there are new mne types not listed above
        return type_lower

    @staticmethod
    def get_mne_ch_types(stream_type: str, ch_names: list, type_map: dict = None):
        ch_types = []
        type_from_stream = XDFStream.stream_type_to_mne_ch_type(stream_type, type_map=type_map)
        for ch_name in ch_names:
            ch_type = type_from_stream
            if stream_type.startswith('Acc'):
                ch_type = 'misc'
            if ch_name.startswith('Markers'):
                ch_type = 'stim'
            if ch_name.startswith('Acc'):
                ch_type = 'misc'
            if ch_name.startswith('Gyro'):
                ch_type = 'misc'
            if ch_name.startswith('Quat'):
                ch_type = 'misc'
            ch_types.append(ch_type)
                

        return ch_types

    def __init__(self, stream, mne_type_map: dict = None):
        self.stream = stream
        self.mne_type_map = mne_type_map
        self.metadata_desc = {}
        self.raw = None
        self.annotations = None
        if self.stream["info"]["desc"] and self.stream["info"]["desc"][0]:
            self.metadata_desc = self.stream["info"]["desc"][0]
        
        self.time_offset = float(self.stream['info']['created_at'][0])
        
        
    @property
    def name(self):
        return self.stream["info"]["name"][0]

    @property
    def type(self):
        return self.stream["info"]["type"][0]
    
    @property
    def id(self):
        return self.stream["info"]["stream_id"]

    @property
    def srate(self):
        return float(self.stream["info"]["nominal_srate"][0])

    @property
    def is_mne_compatible(self):
        return self.is_mne_raw_compatible or self.is_mne_annotations_compatible

    @property
    def is_mne_raw_compatible(self):
        if self.type.lower() == 'quality':
            return False
        return self.srate > 0

    @property
    def is_mne_annotations_compatible(self):
        return self.srate == 0.0

    @property
    def time_series(self):
        # Here we check wether the data is in the correct shape () and transpose it if necessary
        # ! We assume that no EEG recoding would have more channels than sample point !
        # TODO debug time stamps
        # print("BBBBBBBBBBBBBBBBBBB")
        # print(self.time_offset)  # Time stamps of the markers
        # print(self.stream['time_stamps'])  # Time stamps of the markers
        if self.stream["time_series"].shape[0] > self.stream["time_series"].shape[1]:
            return self.stream["time_series"].T

        return self.stream["time_series"]
        

    @property
    def ch_names(self):

        # The specification for metadata is to have xml nodes as desc->channels->channel->label...
        # Some manufacturers have desc->channel->name...
        # See https://github.com/sccn/xdf/wiki/EEG-Meta-Data 
        # Let's handle both

        lookup_fields = ["label", "name"]
        channels_children = None
        if "channels" in self.metadata_desc:
            channels_children = self.metadata_desc["channels"][0]["channel"]
        elif "channel" in self.metadata_desc:
            channels_children = self.metadata_desc["channel"]
        
        if channels_children is not None:
            for field in lookup_fields:
                if field in channels_children[0]:
                    return [channel[field][0] for channel in channels_children]
        
        # fallback to generic naming
        n_channels = int(self.stream['info']['channel_count'][0])
        return [f'{self.type}_{n:03}' for n in range(1, 1+n_channels)]
    
    @property
    def ch_types(self):
        return XDFStream.get_mne_ch_types(self.type, self.ch_names, self.mne_type_map)
        
    def create_mne_annotations(self):
        if self.type == 'Markers':
            timestamps = self.stream['time_stamps']  # Time stamps of the markers
            descriptions = self.stream['time_series']  # Marker descriptions (the event labels)

            # Ensure timestamps are in seconds (if they aren't already)
            timestamps = np.array(timestamps) - self.time_offset
            onset_times = timestamps
            duration = np.zeros_like(onset_times)

            # Create the description for each marker (can be customized, here we use the marker text)
            description = [str(desc[0]) for desc in descriptions]  # Flatten description if necessary

            # Create the annotations object
            self.annotations = mne.Annotations(onset=onset_times, duration=duration, description=description)
            # TODO debug time stamps
            # print("AAAAAAAAAAAAAAAAAAAAAAAAAaaa")
            # print(onset_times)

        else:
            self.markers = None

    def get_unique_name(self, append_stream_id: bool = False):
        unique_stream_name = self.name
        if append_stream_id:
            unique_stream_name += f'-{self.id}'
        return unique_stream_name
        
        
    def convert_to_mne(self, scale: float | str | None, append_stream_id: bool = False, verbose: bool = False):
        self.unique_name = self.get_unique_name(append_stream_id=append_stream_id)
        if self.is_mne_raw_compatible:
            mne_info = self.create_mne_info(verbose=verbose)
            self.raw = self.create_mne_raw(mne_info, scale, verbose=verbose)

        if self.is_mne_annotations_compatible:
            self.create_mne_annotations()
        
    def create_mne_info(self, verbose: bool = False):
        """
        Create a mne.info object from the XDF's EEG stream metadata.

        Arguments:
            idx: The index of the stream to create mne.info for.
            type: Type of the stream to create info for (e.g., "EEG", "MEG").
        """
        ch_names = self.ch_names
        ch_types = self.ch_types

        if self.srate <= 0:
            raise RuntimeError(f"invalid sampling frequency {self.srate} for stream '{self.id}' with name '{self.name}' and type '{self.type}'")

        mne_info = mne.create_info(ch_names, ch_types=ch_types, sfreq=self.srate, verbose=verbose)
        
        mne_info["subject_info"] = {
            "id": self.id, # integer identifier of the subject
            "his_id": self.unique_name, # string identifier of the subject
        }

        return mne_info

    def create_mne_raw(self, mne_info, scale, verbose: bool = False):
        """
        Create a mne.Raw object.

        Arguments:
            idx: The index of the stream to create info for.
            mne_info: The `mne.info` to create the mne.Raw with.
        """
        # Apply automatic scaling to the data
        scaled_data = self.scale_data(self.time_series, scale, verbose=verbose)
        return mne.io.RawArray(scaled_data, mne_info) 
    
    def scale_data(self, data, scale: float | str | None, std_threshold=1e-5, amplitude_threshold=1.0, verbose=False):
        """
        Automatically scale EEG data to V if necessary based on both standard deviation and amplitude range.

        Arguments:
            data: EEG data array.
            std_threshold: Threshold for standard deviation to determine scaling.
                           Default is set to 1e-5.
            amplitude_threshold: Threshold for amplitude range to determine scaling.
                                 Default is set to 1.0.

        Returns:
            Scaled EEG data array.
        """
        
        # Check for NaN and Inf values in the data
        data = data.astype(np.float64)

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print("Data contains NaN or Inf values: No convertion will be done")
            return data

        std = data.std()

        if scale=='auto':
            amplitude_range = data.max() - data.min()

            if std > std_threshold and amplitude_range > amplitude_threshold:
                # Data has both a small standard deviation and small amplitude range, indicating it's in mV
                print(f"- Std ({std}) and amplitude range ({amplitude_range}) are larges >> converting to V (e.g., * 10e-6)")
                return data * 10e-6  # Scale to V
            else:
                # Data has a large standard deviation or amplitude range, indicating it's already in V
                print(f"- Std ({std}) or amplitude range ({amplitude_range}) are smalls >> NOT converting to V")
                return data
        
        if isinstance(scale, (int, float)):
            if verbose:
                print(f"- Returning the data scaled with {scale}")
            return data * scale
        
        if verbose:
            print("- Returning the data with no transformation, user input is 'None'")

        return data
    
    def set_montage(self, montage):
        self.raw.set_montage(montage)
    
    def rename_channels(self, new_names):
        mapping = dict()
        old_names = self.ch_names
        assert len(old_names) == len(new_names)

        for i, new_name in enumerate(new_names):
            mapping[old_names[i]] = new_name
        
        self.raw.rename_channels(mapping)
    
    def save_to_fif_file(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        save_file_path = os.path.join(dir_path, f"{self.unique_name}_raw.fif")
        self.raw.save(save_file_path, overwrite=True)
        return save_file_path
    
    
    def __str__(self):
        ch_names = ','.join(self.ch_names)
        ch_types = ','.join(self.ch_types)
        return f"""Stream id {self.id} of type '{self.type}' with name '{self.name}'
    Sampling Rate: {self.srate}
    Channel names: [{ch_names}]
    Channel types: [{ch_types}]"""