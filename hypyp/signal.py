import numpy as np

class SynteticSignal:
    def __init__(self, tmax: int, n_points: int = 10000):
        self.n_points = n_points
        self.times_vec = np.linspace(0, tmax, n_points)
    
    @property
    def x(self):
        return self.times_vec

    def gen_sin(self, freq):
        self.y = np.sin(self.times_vec * 2 * np.pi * freq)
        return self
        
