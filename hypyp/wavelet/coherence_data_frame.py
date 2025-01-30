from typing import TypedDict, cast, Self, List

import pandas as pd
import pyarrow.feather as feather

COHERENCE_FRAME_COLUMNS = [
    'dyad',
    'is_intra',
    'is_shuffle',
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
    is_shuffle: bool

    subject1: pd.Categorical
    subject2: pd.Categorical
    roi1: pd.Categorical
    roi2: pd.Categorical
    channel1: pd.Categorical
    channel2: pd.Categorical

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
    def from_wtc_frame_rows(data:list) -> Self:
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
    
    #@staticmethod
    #def _set_dtype_categories(df):
    #    df['dyad'] = df['dyad'].astype('category')
    #    df['subject1'] = df['subject1'].astype('category')
    #    df['subject2'] = df['subject2'].astype('category')
    #    df['roi1'] = df['roi1'].astype('category')
    #    df['roi2'] = df['roi2'].astype('category')
    #    df['channel1'] = df['channel1'].astype('category')
    #    df['channel2'] = df['channel2'].astype('category')
    #    df['task'] = df['task'].astype('category')
    #    df['bin_time_range'] = df['bin_time_range'].astype('category')
    #    df['bin_period_range'] = df['bin_period_range'].astype('category')
    #    df['wavelet_library'] = df['wavelet_library'].astype('category')
    #    df['wavelet_name'] = df['wavelet_name'].astype('category')
        
    
    @staticmethod
    def concat(dfs: List[Self]) -> Self:
        """
        Takes a list of dataframes and concatenate them into one dataframe

        Args:
            dfs (List[Self]): the list of dataframes (CoherenceDataFrame)

        Returns:
            CoherenceDataFrame: _description_
        """
        df = pd.concat(dfs, ignore_index=True)
        return cast(CoherenceDataFrame, df)
        
    @staticmethod
    def from_feather(feather_path:str) -> Self:
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
    def save_feather(df:Self, feather_path:str):
        """
        Save to disk

        Args:
            df (CoherenceDataFrame): the pandas dataframe to save
            feather_path (str): path on disk
        """
        with open(feather_path, 'wb') as f:
            feather.write_feather(df, f)

