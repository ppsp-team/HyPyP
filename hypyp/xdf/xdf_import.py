from typing import List, Dict
import warnings

import pyxdf
import numpy as np

from .xdf_stream import XDFStream, Markers

class XDFImport():
    """
    Read an XDF file and (optionally) create mne Raws and Annotations

    Parameters:
      file_path: Path to XDF file (LSL data recorded with LabRecorder). Can be absolute or relative.
      select_type: Define which type of stream the user is looking to convert. Should match the stream type in LSL
      select_matches: List of the stream index(es) in the XDF the user wishes to convert (can be `str` which the class will try to match to the name of an existing stream or an `int` which will be interpreted as such)
      mne_type_map: Dict to map stream types to mne channel types
      scale: Scaling factor or 'auto' for automatic scaling, None for no scaling.
      convert_to_mne: Flag to disable the automatic conversion to mne Raws and Annotations
      verbose: Verbose flag
    """
    
    file_path: str
    mne_type_map: dict | None
    selected_stream_indices: List[int]
    map_id_to_idx: dict # Stream identifier to the index in our list
    verbose: bool

    def __init__(self,
        file_path: str,
        select_type: str = None,
        select_matches: list = None,
        mne_type_map: dict = None,
        scale: float | str | None = None,
        verbose: bool = False,
        convert_to_mne: bool = True,
    ):
        
        self.file_path = file_path 
        self.scale = scale

        self.selected_stream_indices = []
        self.map_id_to_idx = {}

        self.verbose = verbose

        # Load file
        streams, self.header = pyxdf.load_xdf(file_path, verbose=(None if verbose == False else verbose))

        self.available_streams = [XDFStream(stream, mne_type_map=mne_type_map) for stream in streams]

        self.init_map_streams()

        # Prepare the "selected_streams" list
        if select_matches is not None:
            self.select_streams_by_matches(select_matches)
        elif select_type is not None:
            self.select_streams_by_type(select_type)
        else:
            self.select_all_streams()

        # Assert that we have found at least one real stream of selected type
        if len(self.selected_stream_indices) == 0:
            raise ValueError(f"No stream selected. select_type: {select_type}, select_matches: {select_matches}")

        if verbose: 
            print(self)

        # Create MNE objects
        if convert_to_mne:
            self.convert_streams_to_mne()
    
    @property
    def selected_streams(self):
        """List of selected streams, given the select_match or select_type"""
        return [self.available_streams[idx] for idx in self.selected_stream_indices]
    
    @property
    def selected_signal_streams(self):
        """List of selected streams that are signal streams (srate>0), given the select_match or select_type"""
        return [self.available_streams[idx] for idx in self.selected_stream_indices if self.available_streams[idx].is_mne_raw_compatible]
    
    @property
    def selected_markers_streams(self):
        """List of selected streams that are markers streams (srate==0), given the select_match or select_type"""
        return [self.available_streams[idx] for idx in self.selected_stream_indices if self.available_streams[idx].is_mne_annotations_compatible]
    
    @property
    def mne_raws_dict(self):
        """Dictionary of mne.io.RawArray object for selected streams, by stream name"""
        ret = dict()
        for stream in self.selected_signal_streams:
            ret[stream.name] = stream.mne_raw
        return ret
    
    @property
    def mne_raws(self):
        """List of mne.io.RawArray object for selected streams"""
        return [stream.mne_raw for stream in self.selected_signal_streams]
    
    @property
    def markers_dict(self) -> Dict[str, Markers]:
        """
        Dictionary of Markers object from selected streams that are not signals (srate==0), by stream name

        Markers can be converted to mne.Annotations for signal streams
        """
        ret = dict()
        for stream in self.selected_markers_streams:
            ret[stream.name] = stream.markers
        return ret
    
    @property
    def markers(self) -> Markers:
        """
        Merge list of all Markers from selected streams that are not signals (srate==0)

        Markers can be converted to mne.Annotations for signal streams
        """
        markers = None
        for stream in self.selected_markers_streams:
            if markers is None:
                markers = stream.markers
            else:
                markers += stream.markers
        return markers
    
    def get_stream_indices_for_type(self, stream_type: str):
        """For a given type of stream, get the list of index in the available streams which are of this type"""
        return [idx for idx, stream in enumerate(self.available_streams) if stream.type == stream_type]
        
    def get_stream_ids_for_type(self, stream_type: str):
        """For a given type of stream, get the XDF identifiers in the available streams which are of this type"""
        return [stream.id for stream in self.available_streams if stream.type == stream_type]
        
    def init_map_streams(self):
        """
        Create mapping between stream indentifiers and indices in our available_streams list

        'id' is the stream identifier

        'idx' is the index of the stream in our list "available_streams"
        """
        for idx, stream in enumerate(self.available_streams):
            self.map_id_to_idx[stream.id] = idx

        
    def select_all_streams(self) -> list:
        """
        Find in the available streams loaded from the XDF file all the streams that can be converted to MNE
        
        Subsequent calls to class methods will only apply to the selected streams
        """
        self.selected_stream_indices = [idx for idx, stream in enumerate(self.available_streams) if stream.is_mne_compatible]
        

    def select_streams_by_type(self, stream_type: str) -> list:
        """
        Find in the available streams loaded from the XDF file the streams that are of a specific type.
        
        Subsequent calls to class methods will only apply to the selected streams

        Parameters:
          type: The string (e.g., "EEG", "video") that will be matched to XDF stream's `type` to find their indexes.
        """

        if self.verbose:
            print(f"Looking for streams of type '{stream_type}'")
        
        self.selected_stream_indices = self.get_stream_indices_for_type(stream_type)

    def select_streams_by_matches(self, keyword_matches: list):
        """
        Interpret the query made by the user (a list of indexes, or `str` that matches 
        streams' name) into a list containing the indexes within the XDF file.

        Parameters:
            idx: List containing the index that the user is trying to convert.
        """
        for keyword_match in keyword_matches:
            # match stream_id
            found_idx = None
            if type(keyword_match) == int:
                stream_id = keyword_match
                found_idx = self.map_id_to_idx[stream_id]

            # match stream name
            else:
                for idx, stream in enumerate(self.available_streams):
                    if stream.name == keyword_match:
                        found_idx = idx
                
            if found_idx is None:
                raise ValueError(f"No stream matching keyword '{keyword_match}'")

            self.selected_stream_indices.append(found_idx) 

    def convert_streams_to_mne(self):
        """
        Create mne.io.RawArray objects from every signal streams, and add all markers as mne.Annotations to the mne.io.RawArray objects

        mne.io.RawArray objects are then available using obj.mne_raws or obj.mne_raws_dict
        """
        
        # Find if all the stream have unique names (true if any stream name is duplicated)
        names = [stream.name for stream in self.selected_streams]

        for stream in self.selected_streams:
            if self.verbose:
                print(f'Converting {stream.name} to MNE')

            has_duplicate_names = names.count(stream.name) > 1
            if has_duplicate_names:
                warnings.warn("Multiple streams have the same name. Adding original stream_id as suffixes to the generated raws") 
            
            stream.convert_to_mne_raw(self.scale, append_stream_id=has_duplicate_names)
        
        # add annotations from markers streams to data streams
        for data_stream in self.selected_signal_streams:
            if self.markers is not None:
                data_stream.mne_raw.set_annotations(self.markers.as_mne_annotations(data_stream.reference_time))

        if self.verbose:
            print("All convertion to MNE done.")
        
    def rename_channels(self, new_names):
        """
        Set the name of all the channels for every signal streams. Useful when they were not correctly loaded (or not present) from the XDF file

        Parameters:
            new_names (List[str]): The list of new names to set
        """
        for stream in self.selected_signal_streams:
            stream.rename_channels(new_names)
    
    def set_montage(self, montage):
        """
        Set the montage of the raw(s) using a custom mne montage label, or the path to a dig.montage file.

        Parameters:
            self: The instance of the class.
            montage: A path to a local Dig montage or a mne standard montage.
        """
        if self.verbose or True:
            names = [stream.name for stream in self.selected_signal_streams]
            print(f"Setting '{montage}' as the montage for streams: {','.join(names)}")


        for stream in self.selected_signal_streams:
            try: 
                stream.set_montage(montage)
            except ValueError as e:
                warnings.warn(f"Invalid montage given to mne.set_montage(): {montage}")
                raise e

    def save_to_fif_files(self, dir_path):
        """
        Save all the mne.io.RawArray objects as .fif files

        Parameters:
            dir_path (str): Relative or absolute path of the folder where to save the .fif files

        Returns:
            List[str]: The list of file names that have been created
        """
        return [stream.save_to_fif_file(dir_path) for stream in self.selected_signal_streams]
    
    def __str__(self):
        available_streams_str = "\n  ".join([str(stream) for stream in self.available_streams])
        selected_streams_str = ",".join([stream.name for stream in self.selected_streams])
        return f"""XDFImport with {len(self.available_streams)} available streams and {len(self.selected_streams)} selected streams (loaded from {self.file_path})
Available streams:
  {available_streams_str}
Selected streams: [{selected_streams_str}]
"""
    