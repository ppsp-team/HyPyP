from dataclasses import dataclass

import numpy as np
import mne
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PSD():
    freqs: np.array
    psd: np.ndarray
    ch_names: list[str]

    @property
    def is_averaged(self):
        return len(self.psd.shape) == 2

    def plot(self, ax: Axes = None, title: str = 'Average Power in EEG Frequency Bands Across Channels'):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        psd = self.psd
        # average epochs for plotting
        if not self.is_averaged:
            psd = np.mean(psd, axis=0)
        
        # Plot heatmap
        sns.heatmap(psd, xticklabels=self.freqs, yticklabels=self.ch_names, cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Channel')

        return fig


