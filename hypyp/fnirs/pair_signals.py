class PairSignals:
    def __init__(self, x, y1, y2, info_table1=[], info_table2=[]):
        self.x = x
        self.n = len(x)
        self.dt = x[1] - x[0]
        self.fs = 1 / self.dt
        self.y1 = y1
        self.y2 = y2
        self.info_table1 = info_table1
        self.info_table2 = info_table2

    def sub_hundred(self, range):
        if range[0] == 0 and range[1] == 100:
            return self

        signal_from = self.n * range[0] // 100
        signal_to = self.n * range[1] // 100
        
        return PairSignals(
            self.x[signal_from:signal_to],
            self.y1[signal_from:signal_to],
            self.y2[signal_from:signal_to],
            self.info_table1,
            self.info_table2,
        )
    

