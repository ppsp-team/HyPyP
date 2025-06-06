from typing import List
from collections import OrderedDict

import scipy.io

class ChannelROI:
    rois: OrderedDict

    def __init__(self, rois:OrderedDict|None=None):
        """
        Structure for grouping channels into regions of interest

        Args:
            rois (OrderedDict | None, optional): predefined regions of interest. Defaults to None.
        """
        if rois is None:
            self.rois = OrderedDict()
        else:
            self.rois = rois

    @staticmethod
    def from_lionirs_file(roi_file_path:str):
        """
        Create a new ChannelROI object from a file exported 

        Args:
            roi_file_path (str): The path of the file exported by LIONirs toolbox for Region of Interests
                                 It should be a matlab file with structure zone->(label|plotLst)

        Returns:
            ChannelROI: an instance of the class
        """
        obj = ChannelROI()

        mat = scipy.io.loadmat(roi_file_path, struct_as_record=False)

        row_channel_name_map = dict()

        for i, channel_row in enumerate(mat['zone'][0][0].ml):
            row_id = i+1
            source_id = channel_row[0]
            detector_id = channel_row[1]
            # The last column is 1 or 2, which represents hbo or hbr, but we don't need that info here
            row_channel_name_map[row_id] = f'S{source_id}_D{detector_id}'

        mat_rois = mat['zone'][0][0].label[0]
        mat_rows_per_rois = mat['zone'][0][0].plotLst[0]

        for mat_roi, mat_rows_per_roi in zip(mat_rois, mat_rows_per_rois):
            roi_name = mat_roi[0]
            rows_in_roi = mat_rows_per_roi[0]
            channels_in_roi = []
            for row in rows_in_roi:
                channels_in_roi.append(row_channel_name_map[row])
            obj.rois[roi_name] = channels_in_roi
        return obj
        
    @property
    def ordered_ch_names(self):
        channel_names = []
        for channel_names_in_roi in self.rois.values():
            channel_names = channel_names + channel_names_in_roi
        return channel_names
    
    @property
    def group_boundaries_sizes(self):
        boundaries = [0]
        for values in self.rois.values():
            boundaries.append(len(values) + boundaries[-1])
        boundaries.append(len(self.ordered_ch_names))
        return boundaries

    def get_ch_names_in_order(self, ch_names:str) -> List[str]:
        """
        Given a list of channel names, return them in order of Region of Interest

        Args:
            names (str): 

        Returns:
            List[str]: all the received ch_names, ordered according to defined regions of interest
        """
        all_names_found = []

        # OPTIMIZE: this is inefficient
        for ordered_name in self.ordered_ch_names:
            names_found = [name for name in ch_names if name.startswith(ordered_name)]
            for name_found in names_found:
                if name_found not in all_names_found:
                    all_names_found.append(name_found)
        
        for unordered_ch_name in ch_names:
            if not unordered_ch_name in all_names_found:
                all_names_found.append(unordered_ch_name)
        
        return all_names_found

    def get_roi_from_channel(self, ch_name):
        for k, v in self.rois.items():
            # OPTIMIZE: this is inefficient
            names_found = [name for name in v if ch_name.startswith(name)]
            if len(names_found) > 0:
                return k
        return ''
        

