from matplotlib.figure import Figure
import numpy as np

from ..plots import plot_wavelet_transform_weights

class CWT:
    W: np.ndarray
    times: np.ndarray
    dt: float
    scales: np.ndarray
    periods: np.ndarray
    frequencies: np.ndarray
    coi: np.ndarray # Cone of influence, in periods
    coif: np.ndarray # Cone of influence, in frequencies

    def __init__(self, weights:np.ndarray, times:np.ndarray, scales:np.ndarray, periods:np.ndarray, coi:np.ndarray):
        """
        The CWT object holds the results of a Continuous Wavelet Transform 

        Args:
            weights (np.ndarray): weights of the transforms
            times (np.ndarray): timecodes
            scales (np.ndarray): scales used
            periods (np.ndarray): scales in "seconds"
            coi (np.ndarray): cone of influence
        """
        self.W = weights
        self.times = times
        self.dt = times[1] - times[0]
        self.sfreq = 1 / self.dt

        self.scales = scales

        self.periods = periods
        self.frequencies = 1 / periods

        self.coi = coi
        self.coif = 1 / coi

    def plot(self, **kwargs) -> Figure:
        """
        Plot the Continuous Wavelet Transform weights

        Returns:
            Figure: matplotlib.Figure
        """
        return plot_wavelet_transform_weights(self.W, self.times, self.frequencies, self.coif, self.sfreq, **kwargs)


