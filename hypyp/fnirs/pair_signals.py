class PairSignals:
    def __init__(self, x, y1, y2, ch_name1='', ch_name2='', label='', info_table1=[], info_table2=[]):
        self.x = x
        self.n = len(x)
        self.dt = x[1] - x[0]
        self.fs = 1 / self.dt

        self.y1 = y1
        self.y2 = y2

        self.ch_name1 = ch_name1
        self.ch_name2 = ch_name2

        self.label = label

        self.info_table1 = info_table1
        self.info_table2 = info_table2

    def sub(self, range):
        if range[0] == 0 and range[1] == self.n/self.fs:
            return self

        signal_from = int(self.fs * range[0])
        signal_to = int(self.fs * range[1])
        
        return PairSignals(
            self.x[signal_from:signal_to],
            self.y1[signal_from:signal_to],
            self.y2[signal_from:signal_to],
            self.info_table1,
            self.info_table2,
        )
    

