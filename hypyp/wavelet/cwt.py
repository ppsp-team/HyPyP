import numpy as np

from ..plots import plot_cwt_weights

class CWT:
    def __init__(self, weights, times, scales, periods, coi):
        self.W: np.ndarray = weights
        self.times: np.ndarray = times
        self.dt = times[1] - times[0]

        self.scales: np.ndarray = scales

        self.periods: np.ndarray = periods
        self.frequencies: np.ndarray = 1 / periods

        self.coi: np.ndarray = coi # Cone of influence, in periods
        self.coif: np.ndarray = 1 / coi # Cone of influence, in frequencies

    def plot(self, **kwargs):
        return plot_cwt_weights(self.W, self.times, self.frequencies, self.coif)


