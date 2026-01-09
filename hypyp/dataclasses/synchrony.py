from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SynchronyForATask():
    time_series_per_range: np.array
    task_name: str
    freq_ranges: list[str]
    dt: float

    @property
    def by_freq_range(self):
        ret = dict()
        for i in range(self.time_series_per_range.shape[0]):
            k = self.freq_ranges[i]
            ret[k] = self.time_series_per_range[i]
        return ret

    @property
    def times(self):
        return np.arange(self.time_series_per_range.shape[1]) * self.dt
    
    def plot(self):
        for range_idx in range(self.time_series_per_range.shape[0]):
            #x = np.arange(s.shape[1]) * self.wtcs[0].dt
            ts = self.time_series_per_range[range_idx]
            plt.plot(self.times, ts, label=self.freq_ranges[range_idx])
        plt.title(f"Synchrony for task '{self.task_name}'")
        plt.xlabel('time (s)')
        plt.ylabel('synchrony [0-1]')
        plt.ylim((0, None))
        plt.legend()
        plt.show()


@dataclass 
class SynchronyTimeSeries():
    items: Optional[list[SynchronyForATask]] = field(default_factory=lambda: [])

    @property
    def by_task(self) -> dict[str, SynchronyForATask]:
        ret = dict()
        for synchrony in self.items:
            ret[synchrony.task_name] = synchrony
        return ret
    
    def add_task(self, synchrony:SynchronyForATask):
        self.items.append(synchrony)

    def plot(self):
        for s in self.items:
            s.plot()
            plt.show()

