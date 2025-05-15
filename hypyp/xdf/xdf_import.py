from typing import List
import warnings

import pyxdf

from .xdf_stream import XDFStream

class XDFImport():
    """
    Read an XDF file and enable to export stream in a convenient format (e.g., an EEG stream into an mne.Raw instance).

    Arguments:
      file_path: Path to LSL data (i.e., XDF file). Can be absolute or relative.
      stream_type: Define which type of stream the user is looking to convert.
      stream_matches: List of the stream index(es) in the XDF the user wishes to convert (can be `str` which the class will try to match to the name of an existing stream or an `int` which will be interpreted as such). Do not set to convert all of the request type
      scale: Scaling factor or 'auto' for automatic scaling, None for no scaling.
      save_FIF_path: Boolean indicating whether to save the converted data to FIF format.
    """
    
    file_path: str
    mne_type_map: dict | None
    selected_stream_indices: List[int]
    map_id_to_idx: dict
    verbose: bool

    def __init__(self,
        file_path: str,
        stream_type: str = 'EEG',
        stream_matches: list = None,
        mne_type_map: dict = None,
        scale: float | str = None,
        save_FIF_path: bool = None,
        verbose: bool = False,
        convert_to_mne: bool = True,
        ):
        
        self.file_path = file_path 
        self.scale = scale
        self.save_FIF_path = save_FIF_path

        self.selected_stream_indices = []
        self.map_id_to_idx = {}

        self.verbose = verbose

        # Load file
        streams, self.header = pyxdf.load_xdf(file_path, verbose=(None if verbose == False else verbose))

        self.available_streams = [XDFStream(stream, mne_type_map=mne_type_map) for stream in streams]


        if verbose: 
            self.print_available_streams()

        self.map_streams()

        # Prepare the "selected_streams" list
        if stream_matches is None:
            self.select_streams_by_type(stream_type)
        else:
            self.select_streams_by_matches(stream_matches)

        # Create MNE objects
        if convert_to_mne:
            self.convert_streams_to_mne()
    
    @property
    def selected_streams(self):
        return [self.available_streams[idx] for idx in self.selected_stream_indices]
    
    @property
    def selected_stream_names(self):
        return [stream.name for stream in self.selected_streams]
    
    @property
    def raw_all(self):
        ret = dict()
        for stream in self.selected_streams:
            ret[stream.name] = stream.raw
        return ret
    
    def print_available_streams(self):
        print(f"List of available streams in XDF file {self.file_path}:")
        for stream in self.available_streams:
            print(f"  Stream id {stream.id} of type '{stream.type}' with name '{stream.name}'")
            print(f"    Channel names: {','.join(stream.ch_names)}")
            print(f"    Channel types: {','.join(stream.ch_types)}")

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

        self.selected_stream_indices = self.get_stream_indices_for_type(stream_type)
        
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
        names = [stream.name for stream in self.selected_streams]

        for stream in self.selected_streams:
            if self.verbose:
                print(f'Converting {stream.name}')

            has_duplicate_names = names.count(stream.name) > 1
            if has_duplicate_names:
                warnings.warn("Multiple streams have the same name. Adding original stream_id as suffixes to the generated raws") 
            
            stream.convert_to_mne(self.scale, append_stream_id=has_duplicate_names)

            # Save file is asked too
            if self.save_FIF_path is not None: 
                stream.save_fif_file(self.save_FIF_path)

        if self.verbose:
            print("Convertion done.")
        
    def save_fif_files(self, dir_path):
        return [stream.save_fif_file(dir_path) for stream in self.selected_streams]
    
    def rename_channels(self, new_names):
        return [stream.rename_channels(new_names) for stream in self.selected_streams]
    
    def set_montage(self, montage):
        """
        Set the montage of the raw(s) using a custom mne montage label, or the path to a dig.montage file.

        Arguments:
            self: The instance of the class.
            montage: A path to a local Dig montage or a mne standard montage.
        """
        if self.verbose or True:
            print(f"Setting '{montage}' as the montage for streams: {','.join(self.selected_stream_names)}")


        for stream in self.selected_streams:
            try: 
                stream.set_montage(montage)
            except ValueError as e:
                warnings.warn(f"Invalid montage given to mne.set_montage(): {montage}")
                raise e
