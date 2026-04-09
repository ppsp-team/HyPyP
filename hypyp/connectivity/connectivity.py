from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

from ..dataclasses.freq_band import FreqBand

@dataclass
class Connectivity():
    freq_band: FreqBand
    values: np.ndarray
    zscore: np.ndarray
    ch_names: tuple[list[str], list[str]]

    def plot_zscore(self, ax:Axes = None, title: str = None):
        if title is None:
            title = f"Z Score {self.freq_band.name}"

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # If zscore was not averaged, we need to average it for display
        if len(self.zscore.shape) == 3:
            zscore = np.mean(self.zscore, axis=0)
        else:
            zscore = self.zscore

        sns.heatmap(zscore, xticklabels=self.ch_names[0], yticklabels=self.ch_names[1], cmap='viridis', cbar=True, ax=ax)
        ax.set_title(title)
        return fig

