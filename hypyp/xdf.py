"""
XDFImport class

| Option | Description |
| ------ | ----------- |
| title           | io.py |
| authors         | Franck Porteous, Jonas Mago, Guillaume Dumas |
| date            | 2023-02-01 |
"""

from typing import Dict, List
import os
import warnings

import mne
import numpy as np
import pyxdf
from pyxdf.pyxdf import StreamData

class XDFImport():
    """
    Read an XDF file and enable to export stream in a convenient format (e.g., an EEG stream into an mne.Raw instance).

    Arguments:
      file_path: Path to LSL data (i.e., XDF file). Can be absolute or relative.
      stream_type: Define which type of stream the user is looking to convert.
      stream_matches: List of the stream index(es) in the XDF the user wishes to convert (can be `str` which the class will try to match to the name of an existing stream or an `int` which will be interpreted as such). Do not set to convert all of the request type
      montage: A path to a local Dig montage or a mne standard montage.
      scale: Scaling factor or 'auto' for automatic scaling, None for no scaling.
      save_FIF_path: Boolean indicating whether to save the converted data to FIF format.
    """

    @staticmethod
    def stream_type_to_mne_type(stream_type: str, type_map: dict = None):
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
        base_type = XDFImport.stream_type_to_mne_type(stream_type, type_map=type_map)
        for ch_name in ch_names:
            ch_type = base_type
            if ch_name.startswith('Acc'):
                ch_type = 'misc'
            if ch_name.startswith('Gyro'):
                ch_type = 'misc'
            if ch_name.startswith('Quat'):
                ch_type = 'misc'
            ch_types.append(ch_type)
                

        return ch_types

    file_path: str
    mne_type_map: dict | None
    available_stream_names: List[str]
    available_stream_types: List[str]
    selected_stream_indices: List[int]
    map_idx_to_id: dict
    map_id_to_idx: dict
    raw_all: Dict[str, mne.io.RawArray] | None
    verbose: bool

    def __init__(self,
        file_path: str,
        stream_type: str = 'EEG',
        stream_matches: list = None,
        mne_type_map: dict = None,
        montage: str = None,
        scale: float | str = None,
        save_FIF_path: bool = None,
        verbose: bool = False,
        convert_to_mne: bool = True,
        ):
        
        self.file_path = file_path 
        self.scale = scale
        self.save_FIF_path = save_FIF_path
        self.montage = montage
        self.mne_type_map = mne_type_map

        self.available_stream_names = []
        self.available_stream_types = []

        self.selected_stream_indices = []
        self.map_idx_to_id = {}
        self.map_id_to_idx = {}

        self.raw_all = None
        self.verbose = verbose

        # Load file
        self.available_streams, self.header = pyxdf.load_xdf(file_path, verbose=(None if verbose == False else verbose))

        self.load_streams_info()

        if verbose: 
            self.print_streams()

        if stream_matches is None:
            self.select_streams_by_type(stream_type)
        else:
            self.select_streams_by_matches(stream_matches)

        if convert_to_mne:
            self.convert_streams_to_mne()
    
    def load_streams_info(self):
        for idx, stream in enumerate(self.available_streams):
            self.available_stream_names.append(stream["info"]["name"][0])
            self.available_stream_types.append(stream["info"]["type"][0])
            self.map_idx_to_id[idx] = stream["info"]["stream_id"]
            self.map_id_to_idx[stream["info"]["stream_id"]] = idx

    def print_streams(self):
        print(f"List of available streams in XDF file {self.file_path}:")
        for stream in self.available_streams:
            id = stream["info"]["stream_id"]
            idx = self.map_id_to_idx[id]
            stream_name = self.available_stream_names[idx]
            stream_type = self.available_stream_types[idx]
            print(f"  Stream id {id} of type '{stream_type}' with name '{stream_name}'")

    def select_streams_by_type(self, stream_type: str) -> list:
        """
        Read the XDF file to find & store the XDF stream's indexes that match the `type` (e.g., "EEG"). 

        Arguments:
          type: The string (e.g., "EEG", "video") that will be matched to XDF stream's `type` to find their indexes.
        """

        if self.verbose:
            print(f"Looking for streams of type '{stream_type}'")

        stream_ids = self.get_stream_ids_by_type(stream_type)
        
        for stream_id in stream_ids:
            self.selected_stream_indices.append(self.map_id_to_idx[stream_id])
        
        # Assert that we have found at least one real stream of selected type
        if len(self.selected_stream_indices) == 0:
            raise ValueError(f"No stream of type '{stream_type}' were found in this XDF file")

    def get_stream_ids_by_type(self, stream_type: str):
        ids = []
        for idx in range(len(self.available_stream_types)):
            if self.available_stream_types[idx] == stream_type:
                ids.append(self.map_idx_to_id[idx])
        return ids
        
    def select_streams_by_matches(self, keyword_matches: list):
        """
        Interpret the query made by the user (a list of indexes, or `str` that matches 
        streams' name) into a list containing the indexes within the XDF file.

        Arguments:
            idx: List containing the index that the user is trying to convert.
        """
        for keyword_match in keyword_matches:
            if type(keyword_match) == int: # if usr gives stream_id
                stream_id = keyword_match
                self.selected_stream_indices.append(self.map_id_to_idx[stream_id])

            else: # If usr gives anything other than the stream_id
                found = False
                for idx, stream_name in enumerate(self.available_stream_names):
                    if stream_name == keyword_match:
                        found = True
                        self.selected_stream_indices.append(idx) 
                
                if not found:
                    raise ValueError(f"No stream matching keyword '{keyword_match}'")

    def create_mne_info(self, idx: int, append_stream_id: bool = False):
        """
        Create a mne.info object from the XDF's EEG stream metadata.

        Arguments:
            idx: The index of the stream to create mne.info for.
            type: Type of the stream to create info for (e.g., "EEG", "MEG").
        """
        
        #gather metadata
        stream = self.available_streams[idx]
        stream_name = self.available_stream_names[idx]
        stream_type = self.available_stream_types[idx]

        ch_names = self.get_ch_names_for_stream(idx)
        ch_types = XDFImport.get_mne_ch_types(stream_type, ch_names, self.mne_type_map)

        # Create mne.Info object
        sfreq = float(stream["info"]["nominal_srate"][0])
        if sfreq <= 0:
            raise RuntimeError(f"invalid sampling frequency {sfreq} for stream '{self.map_idx_to_id[idx]}' with name '{stream_name}' and type '{stream_type}'")

        mne_info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq, verbose=self.verbose)
        
        unique_stream_name = stream_name
        if append_stream_id:
            warnings.warn("Multiple streams have the same name. Adding original stream_id as suffixes to the generated raws") 
            unique_stream_name += f'-{self.map_idx_to_id[idx]}'
        
        mne_info["subject_info"] = {
            "stream_name": unique_stream_name,
            "stream_id": self.map_idx_to_id[idx],
        }

        return mne_info

    def get_ch_names_for_stream(self, idx: int):
        # Rename channels if the information is available
        desc_info = self.available_streams[idx]["info"]["desc"]
        if desc_info and desc_info[0] and "channels" in desc_info[0]:
            return [channel["label"][0] for channel in self.available_streams[idx]["info"]["desc"][0]["channels"][0]["channel"]]
        
        # fallback to generic naming
        stream = self.available_streams[idx]
        stream_type = self.available_stream_types[idx]
        n_channels = int(stream['info']['channel_count'][0])
        return [f'{stream_type}_{n:03}' for n in range(1, 1+n_channels)]
    
    def create_mne_raw(self, idx: int, mne_info):
        """
        Create a mne.Raw object.

        Arguments:
            idx: The index of the stream to create info for.
            info: The `mne.info` to create the mne.Raw with.
        """

        # Here we check wether the data is in the correct shape () and transpose it if necessary
        # ! We assume that no EEG recoding would have more channels than sample point !
        if self.available_streams[idx]["time_series"].shape[0] > self.available_streams[idx]["time_series"].shape[1]:
            data = self.available_streams[idx]["time_series"].T
        else: 
            data = self.available_streams[idx]["time_series"]

        # Apply automatic scaling to the data
        scaled_data = self.scale_data(data)
        
        return mne.io.RawArray(scaled_data, mne_info) 
    
    def convert_streams_to_mne(self):
        """
        A function that centralizes the pipeline for creating a dictionary containing converted
        XDF stream into `mne.Raw`.

        Note:
            The returned dictionary has the name of the stream as a key and the `mne.Raw` object as the value.
        """
        self.raw_all = {}
        
        # Find if all the stream have unique names (true if any stream name is duplicated)
        selected_stream_names = [self.available_stream_names[idx] for idx in self.selected_stream_indices]

        for idx in self.selected_stream_indices:
            stream_name = self.available_stream_names[idx]
            if self.verbose:
                print(f'Converting {stream_name}')

            has_duplicate_names = selected_stream_names.count(stream_name) > 1
            mne_info = self.create_mne_info(idx, append_stream_id=has_duplicate_names)
            raw = self.create_mne_raw(idx, mne_info)

            # name might have changed if we have duplicates
            # TODO what is this "stream_name"
            unique_stream_name = mne_info['subject_info']['stream_name']
            
            # Save the object/stream_name pair in raw_all
            self.raw_all[unique_stream_name] = raw

            # Save file is asked too
            if self.save_FIF_path is not None: 
                os.makedirs(self.save_FIF_path, exist_ok=True)
                save_file_path = os.path.join([self.save_FIF_path, f"{unique_stream_name}.fif"])
                raw.save(save_file_path, overwrite=True)

                print(f'Saved {stream_name} at {save_file_path}')

        self.setup_montage()       # Set the given montage (local path or mne default montage)

        if self.verbose:
            print("Convertion done.")
        
    def scale_data(self, data, std_threshold=1e-5, amplitude_threshold=1.0):
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

        if self.scale=='auto':
            amplitude_range = data.max() - data.min()

            if std > std_threshold and amplitude_range > amplitude_threshold:
                # Data has both a small standard deviation and small amplitude range, indicating it's in mV
                print(f"- Std ({std}) and amplitude range ({amplitude_range}) are larges >> converting to V (e.g., * 10e-6)")
                return data * 10e-6  # Scale to V
            else:
                # Data has a large standard deviation or amplitude range, indicating it's already in V
                print(f"- Std ({std}) or amplitude range ({amplitude_range}) are smalls >> NOT converting to V")
                return data
        
        if isinstance(self.scale, (int, float)):
            if self.verbose:
                print(f"- Returning the data scaled with {self.scale}")
            return data * self.scale
        
        if self.verbose:
            print("- Returning the data with no transformation, user input is 'None'")

        return data
    
    
    def setup_montage(self):
        """
        Set the montage of the raw(s) using a custom mne montage label, or the path to a dig.montage file.

        Arguments:
            self: The instance of the class.
        """
        if self.montage is None:
            if self.verbose:
                print("- No channels information was found. The montage can be set manually/individually by using the MNE function set_montage on the mne.Raw objects.")
            return

        if self.verbose:
            print(f"Setting '{self.montage}' as the montage for all EEG/sEEG/ECoG/DBS/fNIRS stream(s).")

        for key, raw in self.raw_all.items():
            try: 
                raw.set_montage(self.montage)
            except ValueError as e:
                warnings.warn(f"Invalid montage given to mne.set_montage(): {self.montage}")
                raise e
