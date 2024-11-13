from typing import List

import numpy as np
from scipy.stats import ttest_1samp


from .dyad import Dyad

class Cohort():
    def __init__(self, dyads: List[Dyad] = []):
        self.dyads: List[Dyad] = dyads
        self.dyads_shuffle = self.get_dyads_shuffle()

    @property
    def is_wtc_computed(self):
        for dyad in self.dyads:
            if not dyad.is_wtc_computed:
                return False
        return True

    @property
    def is_wtc_shuffle_computed(self):
        for dyad in self.dyads_shuffle:
            if not dyad.is_wtc_computed:
                return False
        return True

    def get_dyads_shuffle(self) -> List[Dyad]:
        dyads_shuffle = []
        for i, dyad1 in enumerate(self.dyads):
            for j, dyad2 in enumerate(self.dyads):
                if i == j:
                    continue
                dyads_shuffle.append(Dyad(dyad1.s1, dyad2.s2))
        return dyads_shuffle

    def compute_wtcs(self, *args, significance=False, **kwargs):
        for dyad in self.dyads:
            dyad.compute_wtcs(*args, **kwargs)
        
        if significance:
            self.compute_wtcs_shuffle(**kwargs)
            self.compute_wtcs_significance()
            
        return self
    
    def compute_wtcs_shuffle(self, *args, **kwargs):
        for dyad in self.dyads_shuffle:
            dyad.compute_wtcs(*args, **kwargs)
        return self
    
    def compute_wtcs_significance(self):
        if not self.is_wtc_computed:
            raise RuntimeError('Must compute wavelet coherence for dyads to have significance')

        if not self.is_wtc_shuffle_computed:
            raise RuntimeError('Must compute wavelet coherence for dyads shuffle to have significance')
        
        # TODO statistic test to have real significance
        # TODO better looping, with dictionaries to have direct access to dyads channels
        for i, dyad in enumerate(self.dyads):
            for j, wtc in enumerate(dyad.wtcs):
                sig_metrics = []
                # get the same pair for each dyad_shuffle
                for k, dyad_shuffle in enumerate(self.dyads_shuffle):
                    # don't include the real dyad
                    if k == j:
                        continue
                    for l, wtc_shuffle in enumerate(dyad_shuffle.wtcs):
                        if wtc_shuffle.label == wtc.label:
                            sig_metrics.append(wtc_shuffle.sig_metric)

                others = np.array(sig_metrics)
                n = len(others)
                if n > 0:
                    # p-value
                    # TODO: this should be better than this
                    wtc.sig_t_stat, wtc.sig_p_value = ttest_1samp(others, wtc.sig_metric)
        
        return self
        
    