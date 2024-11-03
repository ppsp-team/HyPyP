from typing import List, Tuple
from itertools import compress
from enum import Enum

import numpy as np
import mne
import itertools as itertools

from hypyp.fnirs.data_loader_fnirs import DataLoaderFNIRS

class SubjectFNIRS:
    def __init__(self):
        self.filepath: str | None
        self.raw: mne.io.Raw
        self.raw_od: mne.io.Raw
        self.raw_od_clean: mne.io.Raw
        self.quality_sci: np.ndarray
        self.best_ch_names: List[str] | None
        self.events: any # we should know what type this is
        self.epochs: mne.Epochs
        self.ignore_distances = True

    def load_file(self, loader: DataLoaderFNIRS, filepath: str):
        self.filepath = filepath        
        self.raw = loader.get_mne_raw(filepath)
        self.preprocess()
        return self
    
    def get_analysis_properties(self):
        return {
            'raw_haemo_filtered': 'Hemoglobin filtered',
            'raw_haemo': 'Hemoglobin',
            # Don't use them because they have different channel names
            #'raw_od_clean': 'Raw optical density cleaned',
            #'raw': 'Raw',
        }
    
    def get_ch_names_for_property(self, property_name):
        if property_name.startswith('raw_haemo'):
            return self.raw_haemo.ch_names
        return self.raw.ch_names
        
    def preprocess(self):
        picks = mne.pick_types(self.raw.info, meg=False, fnirs=True)

        # TODO: it seems that .snirf files have a different measurem
        if not self.ignore_distances:
            dists = mne.preprocessing.nirs.source_detector_distances(self.raw.info, picks=picks)
            self.raw.pick(picks[dists > 0.01])

        haemo_picks = mne.pick_types(self.raw.info, fnirs=['hbo', 'hbr'])

        # If we have haemo_picks, it means it is already preprocessed
        # TODO: this code flow if confusing
        if len(haemo_picks) > 0:
            self.raw_haemo = self.raw.copy().pick(haemo_picks)
        else:
            self.raw_od = mne.preprocessing.nirs.optical_density(self.raw)
            self.quality_sci = mne.preprocessing.nirs.scalp_coupling_index(self.raw_od)
            self.raw_od.info['bads'] = list(compress(self.raw_od.ch_names, self.quality_sci < 0.1))
            picks = mne.pick_types(self.raw_od.info, meg=False, fnirs=True, exclude='bads')
            self.raw_od_clean = self.raw_od.copy().pick(picks)
            # TODO: see if we want to expose parameters here
            self.raw_haemo = mne.preprocessing.nirs.beer_lambert_law(self.raw_od_clean, ppf=0.1)

        # TODO: have these parameters exposed?
        self.raw_haemo_filtered = self.raw_haemo.copy().filter(
            0.05,
            0.7,
            h_trans_bandwidth=0.2,
            l_trans_bandwidth=0.02)

    def set_best_ch_names(self, ch_names):
        self.best_ch_names = ch_names
        return self
    
    def load_epochs(self, tmin: int, tmax: int, baseline: Tuple[int, int]):
        if self.raw is None:
            raise RuntimeError('Must load raw data first')

        if self.best_ch_names is None:
            raise RuntimeError('No "best channels" has been set')

        ch_picks = mne.pick_channels(self.raw.ch_names, include = self.best_ch_names)
        best_channels = self.raw.copy().pick(ch_picks)
        self.events, self.event_dict = mne.events_from_annotations(best_channels)
        self.epochs = mne.Epochs(
            best_channels,
            self.events,
            event_id = self.event_dict,
            tmin = tmin,
            tmax = tmax,
            baseline = baseline,
            reject_by_annotation=False)
        return self
