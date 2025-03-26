import numpy as np
import mne
import os
from scipy.io import loadmat


#Loading fnirs data 
def load_fnirs(path1: str, path2: str, attr: dict = None, preload: bool = False, verbose = None) -> tuple:
    """
    Loads fNIRS data files for two participants from various file formats.
    
    This function supports multiple fNIRS data formats including SNIRF, NIRX, 
    Hitachi CSV, and Boxy TXT files. It automatically detects the format based 
    on file extension or directory structure.

    Parameters
    ----------
    path1 : str
        Path to participant #1 fNIRS data file or directory

    path2 : str
      Path to participant #2 fNIRS data file or directory

    attr : dict, optional
        Dictionary containing format-specific optional attributes:
        - For SNIRF: {"optode_frame": str} (e.g., "unknown", "mri", etc.)
        - For NIRX: {"saturated": str} (e.g., "annotate", "nan", etc.)
        If None, default values are used.

    preload : bool, optional
        Whether to preload data into memory (default=False):
        - True: Data is loaded into memory (faster access, higher memory usage)
        - False: Data is loaded on-demand (slower access, lower memory usage)
        - str: Path for memory-mapped file (compromise between speed and memory)

    verbose : bool or None, optional
        Controls verbosity of logging output. If None, uses MNE default verbosity.

    Returns
    -------
    tuple : (mne.io.Raw, mne.io.Raw)
        A tuple containing two Raw objects, one for each participant's fNIRS data.

    Notes
    -----
    The function automatically detects the file format based on:
    - File extension (.snirf, .csv, .txt)
    - Directory structure (for NIRX data)

    If the format is not supported, a message is printed and None is returned.

    Examples
    --------
    >>> # Loading SNIRF files
    >>> raw1, raw2 = load_fnirs('subject1.snirf', 'subject2.snirf')

    >>> # Loading NIRX data with custom attributes
    >>> raw1, raw2 = load_fnirs('nirx_data/subj1/', 'nirx_data/subj2/', 
    ...                         attr={'saturated': 'nan'}, preload=True)
    """

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


#Building the montage
def make_fnirs_montage(source_labels: list, detector_labels: list, prob_directory: str, 
                      Nz: list, RPA: list, LPA: list, head_size: float, 
                      create_montage: bool = True, 
                      mne_standard: str = None) -> mne.channels.DigMontage:
    """
    Builds a compatible montage for fNIRS data analysis with MNE functions.
    
    This function creates a digital montage that maps source and detector positions 
    to standardized head coordinates, allowing for proper visualization and analysis 
    of fNIRS data.
    
    Parameters
    ----------
    source_labels : list
        List of source labels in string format, e.g., ['S1', 'S2', ...]
        
    detector_labels : list
        List of detector labels in string format, e.g., ['D1', 'D2', ...]
        
    prob_directory : str
        Path to the probeInfo.mat file containing optode coordinates, or to a custom 
        montage file in a supported format (if create_montage=False)
        
    Nz : list
        3D coordinates of the nasion [x, y, z] in mm
        
    RPA : list
        3D coordinates of the right preauricular point [x, y, z] in mm
        
    LPA : list
        3D coordinates of the left preauricular point [x, y, z] in mm
        
    head_size : float
        Head circumference in mm
        
    create_montage : bool, optional
        Whether to create a new montage from scratch (default=True):
        - True: Build montage from probeInfo.mat file
        - False: Load an existing montage from prob_directory
        
    mne_standard : str, optional
        Name of a standard MNE montage to use if create_montage=False (default=None)
        If None and create_montage=False, loads a custom montage from prob_directory
    
    Returns
    -------
    montage : mne.channels.DigMontage
        A digital montage object compatible with MNE functions
    
    Notes
    -----
    In MNE-Python, fNIRS channel names MUST follow the structure 'S#_D#_type' where:
    - S# is the source number
    - D# is the detector number
    - type is either 'hbo', 'hbr', or the wavelength (e.g., '850nm')
    
    This function creates a temporary .elc file for creating the montage,
    which is automatically removed after the montage is loaded.
    
    Examples
    --------
    >>> # Creating a new montage from probe information
    >>> sources = ['S1', 'S2', 'S3', 'S4']
    >>> detectors = ['D1', 'D2', 'D3', 'D4']
    >>> montage = make_fnirs_montage(
    ...     sources, detectors, 'probeInfo.mat',
    ...     Nz=[0, 0, 0], RPA=[1, 0, 0], LPA=[-1, 0, 0],
    ...     head_size=580
    ... )
    
    >>> # Using a standard MNE montage
    >>> montage = make_fnirs_montage(
    ...     [], [], '', Nz=[], RPA=[], LPA=[], head_size=0,
    ...     create_montage=False, mne_standard='standard_1020'
    ... )
    """

    if create_montage:
        prob_mat = loadmat(prob_directory)
        f = open("fnirs_montage.elc", "w")
        f.write("# ASA optode file\n")
        f.write("ReferenceLabel	avg\n")
        f.write("UnitPosition	mm\n")
        f.write("NumberPositions= " + str(prob_mat['probeInfo']['probes'].item()['nChannel0'].item()+3)+'\n')
        f.write("Positions\n")
        f.write(str(Nz[0]) + ' ' + str(Nz[1]) + ' ' +str(Nz[2])+ '\n')
        f.write(str(RPA[0]) + ' ' + str(RPA[1]) + ' ' + str(RPA[2]) + '\n')
        f.write(str(LPA[0]) + ' ' + str(LPA[1]) + ' ' + str(LPA[2]) + '\n')
        sensor_coord = prob_mat['probeInfo']['probes'].item()['coords_s3'].item()
        detector_coord = prob_mat['probeInfo']['probes'].item()['coords_d3'].item()
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
        os.remove('fnirs_montage.elc')
    else:
        if mne_standard is not None:
            loc = mne.channels.make_standard_montage(mne_standard)
        else:
            loc = mne.channels.read_custom_montage(prob_directory, head_size=0.095, coord_frame='mri')

    return loc


#Epochs
def fnirs_epoch(fnirs_participant_1: mne.io.Raw, fnirs_participant_2: mne.io.Raw, 
               tmin: float = -0.1, tmax: float = 1, baseline: tuple = (None, 0), 
               preload: bool = True, event_repeated: str = 'merge') -> tuple:
    """
    Extracts epochs from raw fNIRS data for two participants.
    
    This function segments continuous fNIRS data into epochs based on 
    event markers/annotations for both participants in a hyperscanning setup.
    
    Parameters
    ----------
    fnirs_participant_1 : mne.io.Raw
        Raw fNIRS data object for participant #1
        
    fnirs_participant_2 : mne.io.Raw
        Raw fNIRS data object for participant #2
        
    tmin : float, optional
        Start time of epochs relative to events in seconds (default=-0.1)
        
    tmax : float, optional
        End time of epochs relative to events in seconds (default=1)
        
    baseline : tuple, optional
        Time interval (a, b) in seconds for baseline correction (default=(None, 0)):
        - a=None: Use beginning of the epoch
        - b=None: Use end of the epoch
        - (None, None): Use entire epoch
        - (0, 0): No baseline correction
        
    preload : bool, optional
        Whether to preload all epochs into memory (default=True):
        - True: Load all epochs (faster but uses more memory)
        - False: Load epochs on-demand (slower but more memory efficient)
        
    event_repeated : str, optional
        How to handle duplicate events (default='merge'):
        - 'error': Raise an error if duplicates exist
        - 'drop': Keep only the first occurrence of each duplicate
        - 'merge': Combine duplicate events into a single event
    
    Returns
    -------
    tuple : (mne.Epochs, mne.Epochs)
        A tuple containing two Epochs objects, one for each participant
    
    Notes
    -----
    Events are automatically extracted from annotations in the raw data objects.
    The event IDs are determined from the annotation descriptions.
    
    Examples
    --------
    >>> # Extract epochs with default parameters
    >>> epochs1, epochs2 = fnirs_epoch(raw1, raw2)
    
    >>> # Extract longer epochs with custom baseline
    >>> epochs1, epochs2 = fnirs_epoch(
    ...     raw1, raw2, tmin=-0.5, tmax=3.0,
    ...     baseline=(-0.5, 0), event_repeated='drop'
    ... )
    """

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

#Get the inputs for building the montage in a more user freindly way

def fnirs_montage_ui() -> tuple:
    """
    Provides an interactive interface for entering montage parameters.
    
    This function prompts the user to input source and detector labels, 
    fiducial coordinates, and head size through the command line, making
    it easier to create montages without directly calling make_fnirs_montage.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    tuple : (list, list, list, list, list, float)
        A tuple containing:
        - source_labels: List of source labels
        - detector_labels: List of detector labels
        - Nz: Nasion coordinates [x, y, z] in mm
        - RPA: Right preauricular coordinates [x, y, z] in mm
        - LPA: Left preauricular coordinates [x, y, z] in mm
        - head_size: Head circumference in mm
    
    Notes
    -----
    This function uses the input() function to gather information, and thus
    should only be used in interactive environments like Jupyter notebooks
    or command-line sessions.
    
    The returned values can be directly passed to make_fnirs_montage().
    
    Examples
    --------
    >>> # Interactive prompt will appear for each input
    >>> source_labels, detector_labels, Nz, RPA, LPA, head_size = fnirs_montage_ui()
    >>> montage = make_fnirs_montage(
    ...     source_labels, detector_labels, 'probeInfo.mat',
    ...     Nz, RPA, LPA, head_size
    ... )
    """

    source_labels = input("please enter sources names with the S# format: ").split()

    detector_labels = input("please enter detectors names with the D# format: ").split()

    Nz = input("please enter 3D Coordination of tip of the nose x y z in mm: ").split()
    for i in range(len(Nz)):
      Nz[i] = float(Nz[i])

    RPA = input("please enter 3D Coordination of the right preauricular x y z in mm: ").split()
    for i in range(len(RPA)):
      RPA[i] = float(RPA[i])

    LPA = input("please enter 3D Coordination of the left preauricular x y z in mm: ").split()
    for i in range(len(LPA)):
      LPA[i] = float(LPA[i])

    head_size = float(input("please enter the head size in mm "))

    return source_labels, detector_labels, Nz, RPA, LPA, head_size