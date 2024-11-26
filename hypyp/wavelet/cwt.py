import numpy as np

from ..plots import plot_cwt_weights

class CWT:
    def __init__(self, weights, times, scales, frequencies, coif, tracer=None):
        self.W: np.ndarray = weights
        self.times: np.ndarray = times
        self.dt = times[1] - times[0]
        self.scales: np.ndarray = scales
        self.frequencies: np.ndarray = frequencies
        self.coi: np.ndarray = 1 / coif # Cone of influence, in scales
        self.coif: np.ndarray = coif # Cone of influence, in frequencies
        self.tracer: dict = tracer

    def plot(self, **kwargs):
        return plot_cwt_weights(self.W, self.times, self.frequencies, self.coif)


