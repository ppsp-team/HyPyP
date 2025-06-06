import mne

from ..base_step import BaseStep

class MneStep(BaseStep[mne.io.Raw]):
    @property
    def n_times(self):
        return self.obj.n_times
        
    @property
    def sfreq(self):
        return self.obj.info['sfreq']
        
    @property
    def ch_names(self):
        return self.obj.ch_names
        
    def plot(self, **kwargs):
        return self.obj.plot(**kwargs)

