from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import chirp

class SyntheticSignal:
    x: np.ndarray
    y: np.ndarray
    n_points: int
    sampling_rate: float
    period: float

    def __init__(self, duration: float = 100, n_points: int = 1000):
        """
        Compose a synthetic signal

        Args:
            duration (int, optional): duration of the signal (max time value). Defaults to 100.
            n_points (int, optional): number of points. Defaults to 1000.
        """
        self.n_points = n_points
        self.x = np.linspace(0, duration, n_points)
        self.sampling_rate = n_points / duration
        self.period = 1.0 / self.sampling_rate
        self.y = np.zeros_like(self.x)

    def add_chirp(self, f0:float, f1:float):
        """
        Add a chirp signal from frequencies f0 to f1

        Args:
            f0 (float): start frequency
            f1 (float): stop frequency

        Returns:
            SyntheticSignal: object itself, for chaining
        """
        self.y += chirp(self.x, f0=f0, f1=f1, t1=np.max(self.x), method='linear')
        return self
    
    def add_sin(self, freq: float):
        """
        Add a sinusoid to the signal at a specific frequency

        Args:
            freq (float): frequency of the sinusoid

        Returns:
            SyntheticSignal: object itself, for chaining
        """
        self.y += np.sin(self.x * 2 * np.pi * freq)
        return self
    
    def add_noise(self, level:float=0.1):
        """
        Add noise to the signal

        Args:
            level (float, optional): multiplication factor for the random noise (which is from 0 to 1). Defaults to 0.1.

        Returns:
            SyntheticSignal: object itself, for chaining
        """
        self.y += level * np.random.normal(0, 1, self.n_points)
        return self
    
    def add_custom(self, y:np.ndarray):
        """
        Add all the received values to the signal

        Args:
            y (np.array): values to add

        Returns:
            SyntheticSignal: object itself, for chaining
        """
        self.y += y
        return self
    
    #
    # Plots
    #
    def plot(self, t:np.ndarray|None=None, ax:Axes|None=None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if t is None:
            range = (0, len(self.x))
        else:
            range = (int(t[0] * self.sampling_rate), int(t[1] * self.sampling_rate))

        ax.plot(self.x[range[0]:range[1]], self.y[range[0]:range[1]])
        ax.set_xlabel('Time (s)')
        return fig
    
    def plot_fft(self, ax:Axes|None=None) -> Figure:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        xf = fftfreq(self.n_points, self.period)[:self.n_points//2]
        yf = fft(self.y)

        #plt.plot(xf, 2.0/self.n_points * np.imag(yf[0:self.n_points//2]))
        #plt.plot(xf, 2.0/self.n_points * np.imag(yf[0:self.n_points//2]))
        ax.plot(xf, 2.0/self.n_points * np.abs(yf[0:self.n_points//2]))
        ax.set_xscale('log')
        ax.set_xlabel('Freq (Hz)')
        return fig
        
        
        
