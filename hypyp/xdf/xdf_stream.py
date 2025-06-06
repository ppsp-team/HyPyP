from typing import List
import os
import warnings
from dataclasses import dataclass

import mne
import numpy as np
from pyxdf import pyxdf

@dataclass
class Markers():
    """
    Data class to keep the markers data from non-signal XDF streams

    Parameters:
        timestamps (1D np.array): The time of markers
        descriptions (List[str]): The string value for every marker. Must have the same length as timestamps
    """
    timestamps: np.array
    descriptions: List[str]

    def as_mne_annotations(self, reference_time: float = 0.0):
        """
        Get the markers as mne.Annotations. Since every signal stream have a different beginning time,
        we need to set the "onset" relative to the beginning of the every signal stream

        Parameters:
            reference_time (float, optional): Time of the first value of the signal stream on which the annotations will be set. Defaults to 0.0.

        Returns:
            mne.Annotations: The annotations, with onset time relative to the reference_time
        """
        onset_times = np.array(self.timestamps) - reference_time
        empty = mne.Annotations(onset=[], duration=[], description=[])
        if len(onset_times) == 0:
            return empty

        i = 0
        while onset_times[i] < 0:
            i += 1
            if i == len(onset_times):
                # No event after the reference_time
                return empty

        durations = np.zeros_like(onset_times)
        return mne.Annotations(onset=onset_times[i:], duration=durations[i:], description=self.descriptions[i:])
        
    def __add__(self, other):
        if not isinstance(other, Markers):
            return NotImplemented
        timestamps = np.concatenate([self.timestamps, self.timestamps])
        descriptions = self.descriptions + other.descriptions
        return Markers(timestamps, descriptions)

class XDFStream():
    """
    Class representation of a single stream from a XDF import

    Parameters:
        pyxdf_stream (pyxdf.StreamData): The original loaded stream from pyxdf library
        metadata_desc (dict): Channel info from xdf loaded stream
        mne_raw (mne.io.RawArray)
    """
    pyxdf_stream: pyxdf.StreamData
    metadata_desc: dict
    mne_raw: mne.io.RawArray | None
    markers: mne.Annotations | None

    @staticmethod
    def get_default_mne_ch_type_for_stream_type(stream_type: str, type_map: dict = None):
        """
        Get a mne channel type, given a stream type.

        Parameters:
            stream_type (str): The loaded type for XDF stream
            type_map (dict, optional): A lookup table for stream_types that should always return a predefined channel type. Defaults to None.

        Returns:
            str: A channel type recognized by MNE
        """
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
        """
        Map every channel to its mne channel type

        Some vender may have accelerometer, quality and EEG signal in the same stream.

        Parameters:
            stream_type (str): XDF stream typ
            ch_names (list): List of channels, loaded from XDF stream
            type_map (dict, optional): A lookup table for stream_types that should always return a predefined channel type. Defaults to None.

        Returns:
            List[str]: List of type for every channel
        """
        ch_types = []
        default_ch_type = XDFStream.get_default_mne_ch_type_for_stream_type(stream_type, type_map=type_map)
        for ch_name in ch_names:
            ch_type = default_ch_type
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
        self.pyxdf_stream = stream
        self.mne_type_map = mne_type_map
        self.metadata_desc = {}
        self.mne_raw = None

        if self.pyxdf_stream["info"]["desc"] and self.pyxdf_stream["info"]["desc"][0]:
            self.metadata_desc = self.pyxdf_stream["info"]["desc"][0]
        
        if self.is_markers_compatible:
            self.init_markers()
        
    @property
    def name(self):
        return self.pyxdf_stream["info"]["name"][0]

    @property
    def type(self):
        return self.pyxdf_stream["info"]["type"][0]
    
    @property
    def id(self):
        return self.pyxdf_stream["info"]["stream_id"]

    @property
    def srate(self):
        return float(self.pyxdf_stream["info"]["nominal_srate"][0])
    
    @property
    def reference_time(self):
        return self.pyxdf_stream['time_stamps'][0]

    @property
    def is_markers_compatible(self):
        return self.srate == 0.0
        
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
        return self.is_markers_compatible

    @property
    def time_series(self):
        if self.pyxdf_stream["time_series"].shape[0] > self.pyxdf_stream["time_series"].shape[1]:
            return self.pyxdf_stream["time_series"].T

        return self.pyxdf_stream["time_series"]
        

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
        n_channels = int(self.pyxdf_stream['info']['channel_count'][0])
        return [f'{self.type}_{n:03}' for n in range(1, 1+n_channels)]
    
    @property
    def ch_types(self):
        return XDFStream.get_mne_ch_types(self.type, self.ch_names, self.mne_type_map)
        
    def init_markers(self):
        """
        Save markers from a stream of type "Markers" (which contains strings and have a sampling rate of 0)

        Stream MUST NOT be a signal stream
        """
        assert self.is_markers_compatible

        timestamps = np.array(self.pyxdf_stream['time_stamps'])

        descriptions = self.pyxdf_stream['time_series']
        descriptions = [str(desc[0]) for desc in descriptions]  # Flatten description

        self.markers = Markers(timestamps, descriptions)

    def get_unique_name(self, append_stream_id: bool = False):
        unique_stream_name = self.name
        if append_stream_id:
            unique_stream_name += f'-{self.id}'
        return unique_stream_name
        
        
    def convert_to_mne_raw(
        self,
        scale: float | str | None,
        append_stream_id: bool = False,
        verbose: bool = False
    ):
        """
        Create and store in self a mne.io.RawArray object from 

        Parameters:
            scale (float | str | None): Scaling factor or 'auto' for automatic scaling, None for no scaling.
            append_stream_id (bool, optional): Should the id of the stream be added to the name (when we stream may have a duplicated name). Defaults to False.
            verbose (bool, optional): Verbose flag. Defaults to False.
        """
        self.unique_name = self.get_unique_name(append_stream_id=append_stream_id)
        if self.is_mne_raw_compatible:
            mne_info = self.create_mne_info(verbose=verbose)
            scaled_data = self.scale_data(self.time_series, scale, verbose=verbose)
            self.mne_raw = mne.io.RawArray(scaled_data, mne_info) 

    def create_mne_info(self, verbose: bool = False):
        """Create a mne.info object from the XDF's stream metadata."""
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

    def scale_data(self, data, scale: float | str | None, std_threshold=1e-5, amplitude_threshold=1.0, verbose=False):
        """
        Automatically scale signal data to V if necessary based on both standard deviation and amplitude range.

        Arguments:
            data: signal data array.
            std_threshold: Threshold for standard deviation to determine scaling.
                           Default is set to 1e-5.
            amplitude_threshold: Threshold for amplitude range to determine scaling.
                                 Default is set to 1.0.

        Returns:
            Scaled signal data array.
        """
        
        # Check for NaN and Inf values in the data
        data = data.astype(np.float64)

        if scale is None:
            return data

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            warnings.warn("Data contains NaN or Inf values: No convertion will be done")
            return data

        if isinstance(scale, (int, float)):
            if verbose:
                print(f"- Returning the data scaled with {scale}")
            return data * scale

        if scale=='auto':
            std = data.std()
            amplitude_range = data.max() - data.min()

            if std > std_threshold and amplitude_range > amplitude_threshold:
                # Data has both a small standard deviation and small amplitude range, indicating it's in mV
                print(f"- Std ({std}) and amplitude range ({amplitude_range}) are larges >> converting to V (e.g., * 10e-6)")
                return data * 10e-6  # Scale to V

            # Data has a large standard deviation or amplitude range, indicating it's already in V
            print(f"- Std ({std}) or amplitude range ({amplitude_range}) are smalls >> NOT converting to V")
            return data
        
        raise ValueError(f"Invalid value for scale: {scale}")
    
    def set_montage(self, montage):
        """
        Add the mne montage

        Parameters:
            montage (str): See MNE documentation
        """
        self.mne_raw.set_montage(montage)
    
    def rename_channels(self, new_names):
        """
        Set the name of all the channels for signal stream. Useful when they were not correctly loaded (or not present) from the XDF file

        Parameters:
            new_names (List[str]): The list of new names to set
        """
        mapping = dict()
        old_names = self.ch_names
        assert len(old_names) == len(new_names)

        for i, new_name in enumerate(new_names):
            mapping[old_names[i]] = new_name
        
        self.mne_raw.rename_channels(mapping)
    
    def save_to_fif_file(self, dir_path: str):
        """
        Save the mne.io.RawArray object as .fif file, using the unique name for file naming

        Parameters:
            dir_path (str): Relative or absolute path of the folder where to save the .fif file

        Returns:
            str: The file name created
        """
        os.makedirs(dir_path, exist_ok=True)
        save_file_path = os.path.join(dir_path, f"{self.unique_name}_raw.fif")
        self.mne_raw.save(save_file_path, overwrite=True)
        return save_file_path
    
    def __str__(self):
        ch_names = ','.join(self.ch_names)
        ch_types = ','.join(self.ch_types)
        return f"""Stream id {self.id} of type '{self.type}' with name '{self.name}'
    Sampling Rate: {self.srate}
    Channel names: [{ch_names}]
    Channel types: [{ch_types}]"""