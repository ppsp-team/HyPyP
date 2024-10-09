import numpy as np
import mne
import os
from scipy.io import loadmat
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import pywt


#Loading fnirs data 

def load_fnirs(path1: str, path2: str, attr: dict = None, preload: bool = False, verbose = None) -> mne.io.Raw:
  """
  Loads fNIRS data files

  Arguments
  ----------
  path1: str
     participant #1 fNIRS data path (directory)

  part2: str
    participant #2 fNIRS data path (directory)

  attr: dict, optional
    dictionary containing optional attributes using to load different fNIRS file
    (default is None, which returns MNE's default attributes)
    
  preload: bool, optional
    Preload data into memory for data manipulation and faster indexing. 
    If True, the data will be preloaded into memory (fast, requires large amount of memory). 
    If preload is a string, preload is the file name of a memory-mapped file which is used to
    store the data on the hard drive: slower, requires less memory. (default is False)
    
  verbose: bool, optional
    Control verbosity of the logging output. If None, use the default verbosity level

  Returns
  --------
  raw:
     instance of RawSNIRF; a Raw object containing fNIRS data.

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

def make_fnirs_montage(source_labels:list, detector_labels:list, prob_directory: str, Nz:list, RPA:list, 
                        LPA:list, head_size:float, create_montage:bool = True, mne_standard:str = None) -> mne.channels.DigMontage:
    """
    Builds a compatible montage with MNE functions

    Arguments
    ---------
    source_labels: list
      list of sources' label in string format 'S#'

    detector_labels: list
      list of detectors' label in string format 'D#'

    prob_directory: str
      directory of the probeInfo.mat file
      file extension can also be: '.loc' or '.locs' or '.eloc' (for EEGLAB files),
      '.sfp' (BESA/EGI files), '.csd', '.elc', '.txt', '.csd', '.elp' (BESA spherical),
      '.bvef' (BrainVision files),
      '.csv', '.tsv', '.xyz' (XYZ coordinates)

    Nz: list
      list of 3D Coordination of the tip of the nose: [x, y, z] in mm; x, y, z are float numbers

    RPA: list
      list of 3D Coordination of the right preauricular: [x, y, z] in mm; x, y, z are float numbers

    LPA: list
      list of 3D Coordination of the left preauricular: [x, y, z] in mm; x, y, z are float numbers

    head_size: float
      Head size in mm

    creat_montage: bool, optional 
      if the montage is already compatible, this argument should be set to False indicating
      there is no need to build the montage from the scratch. (default is True)

    mne_standard: str
      builds the corresponding mne montage (default is None)

    Returns
    -------
    montage: 
      instance of DigMontage, a compatible montage with mne standards

    Note: In MNE-Python the naming of channels MUST follow the structure S#_D# type
          where # is replaced by the appropriate source and detector numbers and type is either hbo, hbr or the wavelength. 
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

def fnirs_epoch(fnirs_participant_1: mne.io.Raw, fnirs_participant_2: mne.io.Raw, tmin: float = -0.1, tmax: float = 1,
                    baseline: tuple = (None, 0), preload: bool = True, event_repeated: str = 'merge'):

    """
    Extracts epochs from the raw instances

    Arguments
    ----------
    fnirs_participant_1: mne.io.Raw
      Raw object containing fNIRS data of participant #1

    fnirs_participant_2: mne.io.Raw
      Raw object containing fNIRS data of participant #2

    tmin, tmax: float, optional
      Start and end time of the epochs in seconds, relative to the time-locked event.
      The closest or matching samples corresponding to the start and end time are included.
      (Defaults are -0.1 and 1, respectively)

    baseline: tuple, optional
      a tuple (a, b). The time interval to consider as “baseline” when applying baseline correction.
      The interval is between a and b (in seconds), including the endpoints.
      If a is None, the beginning of the data is used; and if b is None, it is set to the end of the interval.
      If (None, None), the entire time interval is used. Default: (None, 0)

    preload: bool, optional, optional
      Load all epochs from disk when creating the object or wait before accessing each epoch
      more memory efficient but can be slower (default is True)

    event_repeated: str, optional
      How to handle duplicates in events[:, 0].
      Can be 'error', to raise an error, 
      'drop' to only retain the row occurring first in the events,
      or 'merge' to combine the coinciding events (=duplicates) into a new event. (default is 'merge')

    Returns
    --------
      a tuple containing mne.Epoch data type of participant #1 and mne.Epoch data type of participant #2 

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

def fnirs_montage_ui():

  """
  Get the inputs for building the montage in a more user freindly way

  Arguments
  ---------
    None

  Returns
  --------
    source_labels, detector_labels, Nz, RPA, LPA, head_size:
    see make_fnirs_montage corresponding inputs' description

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



#################################################
def rect(length, normalize=False):
    """ Rectangular function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

    Args:
        length (int): length of the rectangular function
        normalize (bool): normalize or not

    Returns:
        rect (array): the (normalized) rectangular function

    """
    rect = np.zeros(length)
    rect[0] = rect[-1] = 0.5
    rect[1:-1] = 1

    if normalize:
        rect /= rect.sum()

    return rect

def smoothing(coeff, snorm, dj, smooth_factor=0.1):
    """ Smoothing function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

    Args
    ----

    coeff : array
        the wavelet coefficients get from wavelet transform **in the form of a1 + a2*1j**
    snorm : array
        normalized scales
    dj : float
        it satisfies the equation [ Sj = S0 * 2**(j*dj) ]

    Returns
    -------

    rect : array
        the (normalized) rectangular function

    """
    def fft_kwargs(signal, **kwargs):
        return {'n': int(2 ** np.ceil(np.log2(len(signal))))}
    
    W = coeff #.transpose()
    m, n = np.shape(W)

    # Smooth in time
    k = 2 * np.pi * fftfreq(fft_kwargs(W[0, :])['n'])
    k2 = k ** 2
    # Notes by Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    
    F = np.exp(-smooth_factor * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product

    smooth = ifft(F * fft(W, axis=1, **fft_kwargs(W[0, :])),
                      axis=1,  # Along Fourier frequencies
                      **fft_kwargs(W[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT
    if np.isreal(W).all():
        T = T.real

    # Smooth in scale
    wsize = 0.6 / dj * 2
    win = rect(int(np.round(wsize)), normalize=True)
    T = signal.convolve2d(T, win[:, np.newaxis], 'same')

    return T


def xwt_coherence_morl(x1, x2, fs, nNotes=12, detrend=False, normalize=False, tracer=None):
    """
    Calculates the cross wavelet transform coherence between two time series using the Morlet wavelet.

    Arguments:
        x1 : array
            Time series data of the first signal.
        x2 : array
            Time series data of the second signal.
        fs : int
            Sampling frequency of the time series data.
        nNotes : int, optional
            Number of notes per octave for scale decomposition, defaults to 12.
        detrend : bool, optional
            If True, linearly detrends the time series data, defaults to True.
        normalize : bool, optional
            If True, normalizes the time series data by its standard deviation, defaults to True.

    Note:
        This function uses PyWavelets for performing continuous wavelet transforms
        and scipy.ndimage for filtering operations.

    Returns:
        WCT : array
            Wavelet coherence transform values.
        times : array
            Time points corresponding to the time series data.
        frequencies : array
            Frequencies corresponding to the wavelet scales.
        coif : array
            Cone of influence in frequency, reflecting areas in the time-frequency space
            affected by edge artifacts.
    """
    # Assertions and initial computations
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2), "error: arrays not same size"
   
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt
 
    # Data preprocessing: detrend and normalize
    if detrend:
        x1 = signal.detrend(x1, type='linear')
        x2 = signal.detrend(x2, type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2
 
    # Wavelet transform parameters
    #nOctaves = int(np.log2(2 * np.floor(N / 2.0))) 
    nOctaves = int(np.log2(2 * np.floor(N / 2.0))) + 0.6

    def morlet_flambda():
        """From pycwt/mothers.py"""
        """Fourier wavelength as of Torrence and Compo (1998)."""
        f0 = 6
        return (4 * np.pi) / (f0 + np.sqrt(2 + f0**2))

    #scales = 2 ** np.arange(1, (nOctaves), 1.0 / nNotes)
    #def get_scales():
    #    #scales = 2 ** np.arange(1, nOctaves, 1.0 / nNotes)
    #    # Number of scales
    #    s0 = 2 * dt / morlet_flambda()
    #    dj = 1 / 12
    #    J = int(np.round(np.log2(N1 * dt / s0) / dj))
    #    #J = nOctaves - 1
    #    # The scales as of Mallat 1999
    #    sj = 2 ** (np.arange(0, J + 1) * dj)
    #    return sj
    #scales = get_scales()
    # logarithmic scale for scales, as suggested by Torrence & Compo:
    scales = np.geomspace(1, 300, num=116)

    tracer['scales'] = scales

    coef1, freqs1 = pywt.cwt(x1, scales, 'cmor2.5-1.0', sampling_period=dt, tracer=tracer)
    coef2, freqs2 = pywt.cwt(x2, scales, 'cmor2.5-1.0', sampling_period=dt)
    
    dj = 1 / 12

    tracer['W1'] = coef1
    tracer['W2'] = coef2

    #def get_frequencies():
    #    s0 = 2 * dt / morlet_flambda()
    #    J = int(np.round(np.log2(N1 * dt / s0) / dj))
    #    sj = s0 * 2 ** (np.arange(0, J + 1) * dj)
    #    return 1 / (morlet_flambda() * sj)
    
    #frequencies = get_frequencies()
    #print(frequencies.shape)
    #tracer['freq'] = frequencies # TODO: should take it from the return of pywt.cwt

    #frequencies = pywt.scale2frequency('cmor2.5-1.0', scales) / dt
    tracer['freq'] = freqs1
    frequencies = freqs1

    # Compute cross wavelet transform and coherence
    coef12 = coef1 * coef2.conj()
    scaleMatrix = np.ones([1, N]) * scales[:, None]

    snorm = 1 / freqs1 # with "frequencies", we have the same old image
   

    #s0 = snorm[0]
    #sN = snorm[-1]
    #dj = np.log2(sN/s0) / np.size(snorm)

    # def smoothing(X, snorm, dj):
    #     return scipy.ndimage.gaussian_filter(X, sigma=[9, 1])
    
    S1 = smoothing(np.abs(coef1) ** 2 / scaleMatrix, snorm, dj)
    S2 = smoothing(np.abs(coef2) ** 2 / scaleMatrix, snorm, dj)
    S12 = smoothing(coef12 / scaleMatrix, scales, dj)
    if tracer is not None:
      tracer['S1'] = S1
      tracer['S2'] = S2
      tracer['S12'] = S12
    WCT = np.abs(S12) ** 2 / (S1 * S2)

    # Cone of influence calculations
    f0 = 2 * np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    coif = 1.0 / coi
 
    return WCT, times, frequencies, coif

