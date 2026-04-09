from typing import Optional, Dict, Iterator
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SynchronyForCondition():
    time_series_per_range: np.ndarray
    condition_name: str
    freq_bands: list[str]
    dt: float

    @property
    def by_freq_band(self) -> Dict[str, np.ndarray]:
        ret = dict()
        for i in range(self.time_series_per_range.shape[0]):
            k = self.freq_bands[i]
            ret[k] = self.time_series_per_range[i]
        return ret

    @property
    def times(self):
        return np.arange(self.time_series_per_range.shape[1]) * self.dt
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for range_idx in range(self.time_series_per_range.shape[0]):
            #x = np.arange(s.shape[1]) * self.wtcs[0].dt
            ts = self.time_series_per_range[range_idx]
            ax.plot(self.times, ts, label=self.freq_bands[range_idx])
        ax.set_title(f"Synchrony for condition '{self.condition_name}'")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('synchrony [0-1]')
        ax.set_ylim((0, None))
        ax.legend()


@dataclass 
class SynchronyTimeSeries():
    items: Optional[list[SynchronyForCondition]] = field(default_factory=lambda: [])

    @property
    def by_condition(self) -> dict[str, SynchronyForCondition]:
        ret = dict()
        for synchrony in self.items:
            ret[synchrony.condition_name] = synchrony
        return ret
    
    def add_condition(self, synchrony:SynchronyForCondition):
        self.items.append(synchrony)

    def __getitem__(self, idx) -> SynchronyForCondition:
        return self.items[idx]

    def __len__(self) -> SynchronyForCondition:
        return len(self.items)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        for s in self.items:
            s.plot(ax=ax)

