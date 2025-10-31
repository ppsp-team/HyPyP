from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Connectivity():
    freq_band_name: str
    freq_band: tuple[float, float]
    values: np.ndarray
    zscore: np.ndarray
    ch_names: tuple[list[str], list[str]]

    def plot_zscore(self, ax:Axes = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        sns.heatmap(self.zscore, cmap='viridis', cbar=True, ax=ax)
        return fig

