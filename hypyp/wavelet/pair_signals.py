from typing import Tuple

class PairSignals:
    def __init__(self,
                 x,
                 y1,
                 y2,
                 label_ch1='',
                 label_ch2='',
                 label_s1='', # TODO be more consistent in argument names
                 label_s2='',
                 label_roi1='',
                 label_roi2='',
                 label_dyad='',
                 task='',
                 epoch=0,
                 section=0,
                 is_intra:bool=False,
                 is_shuffle:bool=False,
                 range:Tuple[float, float]|None=None,
        ):
        self.x = x
        self.n = len(x)
        self.dt = x[1] - x[0]
        self.fs = 1 / self.dt

        self.y1 = y1
        self.y2 = y2

        self.is_intra = is_intra
        self.is_shuffle = is_shuffle

        self.label_ch1 = label_ch1
        self.label_ch2 = label_ch2

        self.task = task
        self.epoch = epoch
        self.section = section

        self.label_dyad = label_dyad
        self.label_s1 = label_s1
        self.label_s2 = label_s2
        self.label_roi1 = label_roi1
        self.label_roi2 = label_roi2

        self.range = range
    
    @property
    def label(self):
        ret = f'{self.label_ch1} - {self.label_ch2}'

        prefix = self.task
        if self.epoch > 0:
            prefix = f'{prefix}[{self.epoch}]'

        if self.section > 0:
            prefix = f'{prefix}(section:{self.section})'

        if prefix != '':
            ret = f'{prefix} - {ret}'
        
        return ret

    def sub(self, range):
        if range[0] == 0 and range[1] == self.n/self.fs:
            return self

        signal_from = int(self.fs * range[0])
        signal_to = int(self.fs * range[1])
        
        return PairSignals(
            self.x[signal_from:signal_to],
            self.y1[signal_from:signal_to],
            self.y2[signal_from:signal_to],
            is_shuffle=self.is_shuffle,
            task=self.task,
            epoch=self.epoch,
            section=self.section,
            label_dyad=self.label_dyad,
            label_s1=self.label_s1,
            label_s2=self.label_s2,
            label_ch1=self.label_ch1,
            label_ch2=self.label_ch2,
            label_roi1=self.label_roi1,
            label_roi2=self.label_roi2,
            range=range, # keep track that this is a range in an original PairSignals
        )
    
    def __repr__(self):
        return f'Pair({self.label})'
    

