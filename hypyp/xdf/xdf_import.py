from typing import List
import warnings

import pyxdf

from .xdf_stream import XDFStream

class XDFImport():
    """
    Read an XDF file and enable to export stream in a convenient format (e.g., an EEG stream into an mne.Raw instance).

    Arguments:
      file_path: Path to XDF file (LSL data recorded with LabRecorder). Can be absolute or relative.
      stream_type: Define which type of stream the user is looking to convert.
      stream_matches: List of the stream index(es) in the XDF the user wishes to convert (can be `str` which the class will try to match to the name of an existing stream or an `int` which will be interpreted as such). Do not set to convert all of the request type
      mne_type_map: Dict to map stream types to mne channel types
      scale: Scaling factor or 'auto' for automatic scaling, None for no scaling.
    """
    
    file_path: str
    mne_type_map: dict | None
    selected_stream_indices: List[int]
    map_id_to_idx: dict
    verbose: bool

    def __init__(self,
        file_path: str,
        select_type: str = None,
        select_matches: list = None,
        mne_type_map: dict = None,
        scale: float | str = None,
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

        self.map_streams()

        # Prepare the "selected_streams" list
        if select_matches is not None:
            self.select_streams_by_matches(select_matches)
        else:
            self.select_streams_by_type(select_type)

        if verbose: 
            print(self)

        # Create MNE objects
        if convert_to_mne:
            self.convert_streams_to_mne()
    
    @property
    def raws_dict(self):
        ret = dict()
        for stream in self.selected_data_streams:
            ret[stream.name] = stream.raw
        return ret
    
    @property
    def raws(self):
        return [stream.raw for stream in self.selected_data_streams]
    
    @property
    def annotations_dict(self):
        ret = dict()
        for stream in self.selected_markers_streams:
            ret[stream.name] = stream.annotations
        return ret
    
    @property
    def annotations(self):
        return [stream.annotations for stream in self.selected_markers_streams]
    
    @property
    def annotations_flat(self):
        ret = []
        for x in self.annotations:
            ret += x    
        return ret
    
    @property
    def selected_streams(self):
        return [self.available_streams[idx] for idx in self.selected_stream_indices]
    
    @property
    def selected_data_streams(self):
        return [self.available_streams[idx] for idx in self.selected_stream_indices if self.available_streams[idx].is_mne_raw_compatible]
    
    @property
    def selected_markers_streams(self):
        return [self.available_streams[idx] for idx in self.selected_stream_indices if self.available_streams[idx].is_mne_annotations_compatible]
    
    @property
    def selected_stream_names(self):
        return [stream.name for stream in self.selected_streams]
    
    @property
    def selected_data_stream_names(self):
        return [stream.name for stream in self.selected_data_streams]
    
    @property
    def selected_markers_stream_names(self):
        return [stream.name for stream in self.selected_markers_streams]
    
    def get_streams_for_type(self, stream_type: str):
        return [stream for stream in self.available_streams if stream.type == stream_type]
        
    def get_stream_indices_for_type(self, stream_type: str):
        return [idx for idx, stream in enumerate(self.available_streams) if stream.type == stream_type]
        
    def get_stream_ids_for_type(self, stream_type: str):
        return [stream.id for stream in self.available_streams if stream.type == stream_type]
        
    def map_streams(self):
        # create mapping between stream indentifiers and indices in our available_streams list
        # id is the stream identifier
        # idx is the index of the stream in our list "available_streams"
        for idx, stream in enumerate(self.available_streams):
            self.map_id_to_idx[stream.id] = idx

        
    def select_streams_by_type(self, stream_type: str) -> list:
        """
        Read the XDF file to find & store the XDF stream's indexes that match the `type` (e.g., "EEG"). 

        Arguments:
          type: The string (e.g., "EEG", "video") that will be matched to XDF stream's `type` to find their indexes.
        """

        if self.verbose:
            print(f"Looking for streams of type '{stream_type}'")
        
        if stream_type is not None:
            self.selected_stream_indices = self.get_stream_indices_for_type(stream_type)
        else:
            # Use all the mne raw compatible
            # TODO this if is complicated
            self.selected_stream_indices = [idx for idx, stream in enumerate(self.available_streams) if stream.is_mne_compatible]
        
        # Assert that we have found at least one real stream of selected type
        if len(self.selected_stream_indices) == 0:
            raise ValueError(f"No stream of type '{stream_type}' were found in this XDF file")

    def select_streams_by_matches(self, keyword_matches: list):
        """
        Interpret the query made by the user (a list of indexes, or `str` that matches 
        streams' name) into a list containing the indexes within the XDF file.

        Arguments:
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
        A function that centralizes the pipeline for creating a dictionary containing converted
        XDF stream into `mne.Raw`.

        Note:
            The returned dictionary has the name of the stream as a key and the `mne.Raw` object as the value.
        """
        
        # Find if all the stream have unique names (true if any stream name is duplicated)
        names = self.selected_stream_names

        for stream in self.selected_streams:
            if self.verbose:
                print(f'Converting {stream.name}')

            has_duplicate_names = names.count(stream.name) > 1
            if has_duplicate_names:
                warnings.warn("Multiple streams have the same name. Adding original stream_id as suffixes to the generated raws") 
            
            stream.convert_to_mne(self.scale, append_stream_id=has_duplicate_names)

        if self.verbose:
            print("Convertion done.")
        
    def save_to_fif_files(self, dir_path):
        return [stream.save_to_fif_file(dir_path) for stream in self.selected_data_streams]
    
    def rename_channels(self, new_names):
        return [stream.rename_channels(new_names) for stream in self.selected_data_streams]
    
    def set_montage(self, montage):
        """
        Set the montage of the raw(s) using a custom mne montage label, or the path to a dig.montage file.

        Arguments:
            self: The instance of the class.
            montage: A path to a local Dig montage or a mne standard montage.
        """
        if self.verbose or True:
            print(f"Setting '{montage}' as the montage for streams: {','.join(self.selected_data_stream_names)}")


        for stream in self.selected_data_streams:
            try: 
                stream.set_montage(montage)
            except ValueError as e:
                warnings.warn(f"Invalid montage given to mne.set_montage(): {montage}")
                raise e

    def __str__(self):
        available_streams_str = "\n  ".join([str(stream) for stream in self.available_streams])
        selected_streams_str = ",".join([stream.name for stream in self.selected_streams])
        return f"""XDFImport with {len(self.available_streams)} available streams and {len(self.selected_streams)} selected streams (loaded from {self.file_path})
Available streams:
  {available_streams_str}
Selected streams: [{selected_streams_str}]
"""
    