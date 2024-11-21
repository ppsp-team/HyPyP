from collections import OrderedDict
import csv

import scipy.io

class ChannelROI:
    def __init__(self):
        self.rois = OrderedDict()

    @staticmethod
    def from_lionirs(roi_file_path):
        obj = ChannelROI()

        mat = scipy.io.loadmat(roi_file_path, struct_as_record=False)
        #csv_reader = csv.reader(open(channel_map_file_path, "r"), delimiter=',', quoting=csv.QUOTE_NONE)

        row_channel_name_map = dict()

        for i, channel_row in enumerate(mat['zone'][0][0].ml):
            row_id = i+1
            source_id = channel_row[0]
            detector_id = channel_row[1]
            # TODO we have 102 lines, and the last column is 1 or 2, which represents hbo or hbr.
            # Maybe we should also keep track of that info here
            row_channel_name_map[row_id] = f'S{source_id}_D{detector_id}'

        mat_rows_per_groups = mat['zone'][0][0].plotLst[0]
        mat_groups = mat['zone'][0][0].label[0]

        for mat_rows, mat_group in zip(mat_rows_per_groups, mat_groups):
            rows = mat_rows[0]
            group_name = mat_group[0]
            channels = []
            for row in rows:
                channels.append(row_channel_name_map[row])
            obj.rois[group_name] = channels
        return obj
        
    @property
    def ordered_channel_names(self):
        channel_names = []
        for channel_names_roi in self.rois.values():
            channel_names = channel_names + channel_names_roi
        return channel_names

    def get_names_in_order(self, names):
        all_names_found = []

        # TODO this is inefficient
        for ordered_name in self.ordered_channel_names:
            names_found = [name for name in names if name.startswith(ordered_name)]
            for name_found in names_found:
                if name_found not in all_names_found:
                    all_names_found.append(name_found)
        
        for unordered_name in names:
            if not unordered_name in all_names_found:
                all_names_found.append(unordered_name)
        
        return all_names_found


