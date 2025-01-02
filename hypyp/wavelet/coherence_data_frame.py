#import os
from typing import Tuple, TypedDict, cast

#import numpy as np
import pandas as pd
#import feather

COHERENCE_FRAME_COLUMNS = [
    'dyad',
    'is_intra',
    'is_shuffle',
    'task',
    'epoch',
    'section',
    'subject1',
    'subject2',
    'roi1',
    'roi2',
    'channel1',
    'channel2',
    'coherence',
]

class CoherenceDataFrame(TypedDict, total=False):
    # row properties
    dyad: str
    is_intra: bool
    is_shuffle: bool
    task: str
    epoch: int
    section: int
    subject1: str
    subject2: str
    roi1: str
    roi2: str
    channel1: str
    channel2: str
    coherence: float

    @staticmethod
    def from_wtcs(data):
        # TODO: use factors
        df = pd.DataFrame(data, columns=COHERENCE_FRAME_COLUMNS)
        return cast(CoherenceDataFrame, df)
    
    #@staticmethod
    #def from_feather(feather_path: str):
    #    json_path = feather_path.replace('.feather', '.json')

    #    with open(json_path, 'r') as f:
    #        meta = json.load(f)

    #    with open(feather_path, 'rb') as f:
    #        df = feather.read_dataframe(f)
    #    
    #    return cast(Tuple[SensorDataFrame, SensorDataFrameMeta], (df, meta))
    
    #@staticmethod
    #def save_to_feather(df, meta: SensorDataFrameMeta, feather_path: str):
    #    ensure_folder_exists_for(feather_path)
    #    json_path = feather_path.replace('.feather', '.json')

    #    with open(json_path, 'w') as f:
    #        json.dump(meta, f)

    #    with open(feather_path, 'wb') as f:
    #        feather.write_dataframe(df, f)

