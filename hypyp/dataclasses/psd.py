from typing import Optional
from dataclasses import dataclass

import numpy as np
import mne
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class SpectralPower():
    freqs: np.array
    psd: np.ndarray
    ch_names: list[str]
    band_name: Optional[str] = ''

    @property
    def is_averaged(self):
        return len(self.psd.shape) == 2

    @property
    def band_display_name(self):
        if self.band_name == '':
            return f"{self.freqs[0]}-{self.freqs[-1]}"
        else:
            return self.band_name

    def plot(self, ax: Axes = None, title: str = None):
        if title is None:
            title = f"Average Power in EEG Frequency Band {self.band_display_name} Across Channels"

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
        ax.set_xlabel(f"Frequency (Hz)")
        ax.set_ylabel('Channel')
        #ax.invert_yaxis()
        #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        return fig


