from typing import Tuple

import numpy as np

class PairSignals:
    x: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    n: int
    dt: float
    sfreq: float

    is_intra: bool
    is_intra_of: int
    is_pseudo: bool

    label_ch1: str
    label_ch2: str
    label_task: str

    epoch_idx: int
    section_idx: int

    label_dyad: str
    label_s1: str
    label_s2: str
    label_roi1: str
    label_roi2: str

    def __init__(self,
                 x,
                 y1,
                 y2,
                 label_ch1='',
                 label_ch2='',
                 label_roi1='',
                 label_roi2='',
                 label_s1='',
                 label_s2='',
                 label_dyad='',
                 label_task='',
                 epoch_idx=0,
                 section_idx=0,
                 is_intra:bool=False,
                 is_intra_of:int=0,
                 is_pseudo:bool=False,
        ):
        """
        A pair of signal that are already aligned and can be compared

        Args:
            x (_type_): times
            y1 (_type_): signal 1 values
            y2 (_type_): signal 2 values
            label_ch1 (str, optional): label for channel of subject 1. Defaults to ''.
            label_ch2 (str, optional): label for channel of subject 2. Defaults to ''.
            label_roi1 (str, optional): label for region of interest of subject 1. Defaults to ''.
            label_roi2 (str, optional): label for region of interest of subject 2. Defaults to ''.
            label_s1 (str, optional): label for subject 1. Defaults to ''.
            label_s2 (str, optional): label for subject 2. Defaults to ''.
            label_dyad (str, optional): label for the dyad. Defaults to ''.
            label_task (str, optional): label for the task of this signal section. Defaults to ''.
            epoch_id (int, optional): identifier of the epoch of this signal pair. Defaults to 0.
            section_id (int, optional): identifier of the section of this signal pair, when an epoch had to be splitted in smaller sections. Defaults to 0.
            is_intra (bool, optional): if the pair is from an intra-subject. Defaults to False.
            is_intra_of (int, optional): if the pair is from an intra-subject, which subject is this. Defaults to 0 when not intra-subject.
            is_pseudo (bool, optional): if the pair is from a shuffled dyad. Defaults to False.
        """
        self.x = x
        self.n = len(x)
        self.dt = x[1] - x[0]
        self.sfreq = 1 / self.dt

        self.y1 = y1
        self.y2 = y2

        self.is_intra = is_intra
        self.is_intra_of = is_intra_of
        self.is_pseudo = is_pseudo

        self.label_ch1 = label_ch1
        self.label_ch2 = label_ch2

        self.label_task = label_task
        self.epoch_idx = epoch_idx
        self.section_idx = section_idx

        self.label_dyad = label_dyad
        self.label_s1 = label_s1
        self.label_s2 = label_s2
        self.label_roi1 = label_roi1
        self.label_roi2 = label_roi2
    
    @property
    def label(self):
        ret = f'{self.label_ch1} - {self.label_ch2}'

        prefix = self.label_task
        if self.epoch_idx > 0:
            prefix = f'{prefix}[{self.epoch_idx}]'

        if self.section_idx > 0:
            prefix = f'{prefix}(section:{self.section_idx})'

        if prefix != '':
            ret = f'{prefix} - {ret}'
        
        return ret

    def sub(self, time_range:Tuple[float, float], section_idx:int|None=None):
        """
        Get a new PairSignals from a portion of the initial PairSignals

        Args:
            time_range (Tuple[float, float]): from_time and to_time in a tuple
            section_id (int | None, optional): new section_id to set on the pair, useful for splitting a signal. Defaults to self.section_id.

        Returns:
            PairSignals: a new PairSignals
        """
        if time_range[0] == 0 and time_range[1] == self.n/self.sfreq:
            return self

        signal_from = int(self.sfreq * time_range[0])
        signal_to = int(self.sfreq * time_range[1]) + 1

        if section_idx is None:
            section_idx = self.section_idx
        
        return PairSignals(
            self.x[signal_from:signal_to],
            self.y1[signal_from:signal_to],
            self.y2[signal_from:signal_to],
            is_intra=self.is_intra,
            is_intra_of=self.is_intra_of,
            is_pseudo=self.is_pseudo,
            label_ch1=self.label_ch1,
            label_ch2=self.label_ch2,
            label_roi1=self.label_roi1,
            label_roi2=self.label_roi2,
            label_s1=self.label_s1,
            label_s2=self.label_s2,
            label_dyad=self.label_dyad,
            label_task=self.label_task,
            epoch_idx=self.epoch_idx,
            section_idx=section_idx,
        )
    
    def __repr__(self):
        return f'Pair({self.label})'
    

