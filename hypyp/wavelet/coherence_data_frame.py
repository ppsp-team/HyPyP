from typing import Tuple, TypedDict, cast

#import numpy as np
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
    'coherence',
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

    coherence: float

    @staticmethod
    def from_wtcs(data):
        df = pd.DataFrame(
            data,
            columns=COHERENCE_FRAME_COLUMNS,
        )
        CoherenceDataFrame.set_dtype_categories(df)

        return cast(CoherenceDataFrame, df)
    
    @staticmethod
    def set_dtype_categories(df):
        df['dyad'] = df['dyad'].astype('category')
        df['subject1'] = df['subject1'].astype('category')
        df['subject2'] = df['subject2'].astype('category')
        df['roi1'] = df['roi1'].astype('category')
        df['roi2'] = df['roi2'].astype('category')
        df['channel1'] = df['channel1'].astype('category')
        df['channel2'] = df['channel2'].astype('category')
        df['task'] = df['task'].astype('category')
        
    
    @staticmethod
    def concat(dfs):
        df = pd.concat(dfs, ignore_index=True)
        return df
        
    
    @staticmethod
    def from_feather(feather_path: str):
        with open(feather_path, 'rb') as f:
            df = feather.read_feather(f)
        
        return cast(CoherenceDataFrame, df)
    
    @staticmethod
    def save_feather(df, feather_path: str):
        with open(feather_path, 'wb') as f:
            feather.write_feather(df, f)

