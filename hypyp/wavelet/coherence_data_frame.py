from typing import TypedDict, cast, List

import pandas as pd
import pyarrow.feather as feather

COHERENCE_FRAME_COLUMNS = [
    'dyad',
    'is_intra',
    'is_intra_of',
    'is_pseudo',
    'subject1',
    'subject2',
    'roi1',
    'roi2',
    'channel1',
    'channel2',
    'task',
    'epoch',
    'section',
    'bin',
    'coherence',
    'coherence_masked',
    'bin_time_range',
    'bin_period_range',
    'wavelet_library',
    'wavelet_name',
]

class CoherenceDataFrame(TypedDict, total=False):
    # row properties
    dyad: pd.Categorical

    is_intra: bool
    is_intra_of: int
    is_pseudo: bool

    subject1: str
    subject2: str
    roi1: str
    roi2: str
    channel1: str
    channel2: str

    task: pd.Categorical
    epoch: int
    section: int
    bin: int

    coherence: float
    coherence_masked: float
    bin_time_range: pd.Categorical
    bin_period_range: pd.Categorical

    wavelet_library: pd.Categorical
    wavelet_name: pd.Categorical

    @staticmethod
    def from_wtc_frame_rows(data:list):
        """
        Get a typed pandas DataFrame from wavelet transform coherence data

        Args:
            data (list): WTCs data formated as frame rows (see `WTC.as_frame_rows`)

        Returns:
            CoherenceDataFrame: a typed pandas DataFrame
        """
        df = pd.DataFrame(
            data,
            columns=COHERENCE_FRAME_COLUMNS,
        )
        #CoherenceDataFrame._set_dtype_categories(df)

        return cast(CoherenceDataFrame, df)
    
    @staticmethod
    def _set_dtype_categories(df):
        df['dyad'] = df['dyad'].astype('category')
        df['task'] = df['task'].astype('category')
        df['bin_time_range'] = df['bin_time_range'].astype('category')
        df['bin_period_range'] = df['bin_period_range'].astype('category')
        df['wavelet_library'] = df['wavelet_library'].astype('category')
        df['wavelet_name'] = df['wavelet_name'].astype('category')
        
    
    @staticmethod
    def concat(dfs: list[pd.DataFrame]):
        """
        Takes a list of dataframes and concatenate them into one dataframe

        Args:
            dfs (List[CoherenceDataFrame]): the list of dataframes (CoherenceDataFrame)

        Returns:
            CoherenceDataFrame: a new CoherenceDataFrame
        """
        df = pd.concat(dfs, ignore_index=True)
        return cast(CoherenceDataFrame, df)
        
    @staticmethod
    def from_feather(feather_path:str):
        """
        Read pandas feather file and return as CoherenceDataFrame

        Args:
            feather_path (str): path on disk

        Returns:
            CoherenceDataFrame: a typed pandas DataFrame
        """
        with open(feather_path, 'rb') as f:
            df = feather.read_feather(f)
        
        return cast(CoherenceDataFrame, df)
    
    @staticmethod
    def save_feather(df:pd.DataFrame, feather_path:str):
        """
        Save to disk

        Args:
            df (CoherenceDataFrame): the pandas dataframe to save
            feather_path (str): path on disk
        """
        with open(feather_path, 'wb') as f:
            feather.write_feather(df, f)

