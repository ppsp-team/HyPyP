#Loading useful libraries

import numpy as np
import mne
import os
from mat4py import loadmat

def load_fnirs(path1, path2, attr=None, preload=False, verbose=None):

  if ".snirf" in path1:
    if attr is None:
      data_1 = mne.io.read_raw_snirf(path1, optode_frame='unknown', preload=preload, verbose=verbose)
      data_2 = mne.io.read_raw_snirf(path2, optode_frame='unknown', preload=preload, verbose=verbose)
    else:
      data_1 = mne.io.read_raw_snirf(path1, optode_frame = attr["optode_frame"], preload=preload, verbose=verbose)
      data_2 = mne.io.read_raw_snirf(path2, optode_frame = attr["optode_frame"], preload=preload, verbose=verbose)

  elif os.path.isdir(path1):
    if attr is None:
      data_1 = mne.io.read_raw_nirx(path1, saturated='annotate', preload=preload, verbose=verbose)
      data_2 = mne.io.read_raw_nirx(path2, saturated='annotate', preload=preload, verbose=verbose)
    else:
      data_1 = mne.io.read_raw_nirx(path1, saturated = attr["saturated"], preload=preload, verbose=verbose)
      data_2 = mne.io.read_raw_nirx(path2, saturated = attr["saturated"], preload=preload, verbose=verbose)

  elif ".csv" in path1:
    data_1 = mne.io.read_raw_hitachi(path1, preload=preload, verbose=verbose)
    data_2 = mne.io.read_raw_hitachi(path2, preload=preload, verbose=verbose)

  elif ".txt" in path1:
    data_1 = mne.io.read_raw_boxy(path1, preload=preload, verbose=verbose)
    data_2 = mne.io.read_raw_boxy(path2, preload=preload, verbose=verbose)

  else:
    print("data type is not supported")
    
  return (data_1, data_2)

#Building a Compatible Montage
def make_fnirs_montage(source_labels:list, detector_labels:list, prob_directory: str, Nz:list, RPA:list, 
                        LPA:list, head_size:float, create_montage:bool = True, mne_standard:str = None):
    if create_montage:
        prob_mat = loadmat(prob_directory)
        f = open("fnirs_montage.elc", "w")
        f.write("# ASA optode file\n")
        f.write("ReferenceLabel	avg\n")
        f.write("UnitPosition	mm\n")
        f.write("NumberPositions= " + str(prob_mat['probeInfo']['probes']['nChannel0']+3)+'\n')
        f.write("Positions\n")
        f.write(str(Nz[0]) + ' ' + str(Nz[1]) + ' ' +str(Nz[2])+ '\n')
        f.write(str(RPA[0]) + ' ' + str(RPA[1]) + ' ' + str(RPA[2]) + '\n')
        f.write(str(LPA[0]) + ' ' + str(LPA[1]) + ' ' + str(LPA[2]) + '\n')
        sensor_coord = prob_mat['probeInfo']['probes']['coords_s3']
        detector_coord = prob_mat['probeInfo']['probes']['coords_d3']
        for j in range(len(sensor_coord)):
            f.write(str(sensor_coord[j][0]) + ' ' + str(sensor_coord[j][1]) + ' ' +str(sensor_coord[j][2])+ '\n')
        for i in range(len(sensor_coord)):
            f.write(str(detector_coord [i][0]) + ' ' + str(detector_coord [i][1]) + ' ' +str(detector_coord [i][2])+ '\n')
        f.write('Labels\n' + 'Nz\n' + 'RPA\n' + 'LPA\n')
        for k in range(len(detector_labels)):
            f.write(str(detector_labels[k]) +'\n')
        for z in range(len(source_labels)):
            f.write(str(source_labels[z]) +'\n')
        f.close()
        loc = mne.channels.read_custom_montage('fnirs_montage.elc', head_size=head_size)
    else:
        if mne_standard is not None:
            loc = mne.channels.make_standard_montage(mne_standard)
        else:
            loc = mne.channels.read_custom_montage(prob_directory, head_size=0.095, coord_frame='mri')

    return loc

# Make Epoch Objects: 
def fnirs_epoch(fnirs_participant_1, fnirs_participant_2, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, event_repeated='merge'):

    fnirs_raw_1 = fnirs_participant_1
    event1 = mne.events_from_annotations(fnirs_raw_1)
    events1 = event1[0]
    events1id = event1[1]
    fnirs_epo1 = mne.Epochs(fnirs_raw_1, events1, events1id, tmin=tmin, tmax=tmax,
                    baseline=baseline, preload=preload, event_repeated=event_repeated)

    fnirs_raw_2 = fnirs_participant_2
    event2 = mne.events_from_annotations(fnirs_raw_2)
    events2 = event2[0]
    events2id = event2[1]
    fnirs_epo2 = mne.Epochs(fnirs_raw_2, events2, events2id, tmin=tmin, tmax=tmax,
                    baseline=baseline, preload=preload, event_repeated=event_repeated)
    return (fnirs_epo1, fnirs_epo2)

# fnirs_epo1 and fnirs_epo2 are now compatible with tools provided by HyPyP (for visualization and statictical analyses check [this](https://github.com/ppsp-team/HyPyP/blob/master/tutorial/getting_started.ipynb) tutorial)

# Example:
# data is in .snirf format without explicit montage file from https://osf.io/75fet/ 

path_1 = "../data/FNIRS/DCARE_02_sub1.snirf"

path_2 = "../data/FNIRS/DCARE_02_sub2.snirf"

fnirs_participant_1 = load_fnirs(path_1, path_2, attr=None, preload=False, verbose=None)[0]
fnirs_participant_2 = load_fnirs(path_1, path_2, attr=None, preload=False, verbose=None)[1]
#Sources ' labels: S#
source_labels = ['S1','S2','S3','S4','S5','S6','S7','S8']
#Sources ' labels: D#
detector_labels = ['D1','D2','D3','D4','D5','D6','D7','D8']
#directory of the probeInfo.mat file
prob_mat_file = '../data/FNIRS/MCARE_01_probeInfo.mat'
#3D Coordination of the tip of the nose: [x, y, z] in mm
Nz_coord = [12.62, 17.33, 16.74]
#3D Coordination of the right preauricular: [x, y, z] in mm
RPA = [21.0121020904262, 15.9632489747085, 17.2796094659563]
#3D Coordination of the left preauricular: [x, y, z] in mm
LPA = [4.55522116441745, 14.6744377188919, 18.3544292678269]
#Head size in mm
head_size = 0.16

location = make_fnirs_montage(source_labels, detector_labels, prob_mat_file,
                              Nz_coord, RPA, LPA, head_size)

fnirs_epo1 = fnirs_epoch(fnirs_participant_1, fnirs_participant_2)[0]
fnirs_epo2 = fnirs_epoch(fnirs_participant_1, fnirs_participant_2)[1]

fnirs_epo1.set_montage(location)
fnirs_epo2.set_montage(location)

fnirs_epo1.plot_sensors()

fnirs_epo2.plot_sensors()

fnirs_epo1.plot()

fnirs_epo2.plot()


