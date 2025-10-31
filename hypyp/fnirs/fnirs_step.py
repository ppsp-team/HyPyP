import mne

from ..core.base_step import BaseStep

PREPROCESS_STEP_BASE = 'raw'
PREPROCESS_STEP_BASE_DESC = 'Loaded data'

PREPROCESS_STEP_OD = 'od'
PREPROCESS_STEP_OD_DESC = 'Optical density'

PREPROCESS_STEP_OD_CLEAN = 'od_clean'
PREPROCESS_STEP_OD_CLEAN_DESC = 'Optical density cleaned'

PREPROCESS_STEP_HAEMO = 'haemo'
PREPROCESS_STEP_HAEMO_DESC = 'Hemoglobin'

PREPROCESS_STEP_HAEMO_FILTERED = 'haemo_filtered'
PREPROCESS_STEP_HAEMO_FILTERED_DESC = 'Hemoglobin Band-pass Filtered'

class FNIRSStep(BaseStep[mne.io.Raw]):
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

