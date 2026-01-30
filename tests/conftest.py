import pytest
import os
from collections import namedtuple, OrderedDict
import numpy as np
import mne
from hypyp import utils, analyses


@pytest.fixture(scope="module")
def epochs():
    """
    Loading data files & extracting sensor infos
    """
    epo1 = mne.read_epochs(os.path.join("data", "participant1-epo.fif"),
                           preload=True)
    epo2 = mne.read_epochs(os.path.join("data", "participant2-epo.fif"),
                           preload=True)
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    epoch_merge = utils.merge(epo1, epo2)

    epochsTuple = namedtuple('epochs', ['epo1', 'epo2', 'epoch_merge'])

    return epochsTuple(epo1=epo1, epo2=epo2, epoch_merge=epoch_merge)


@pytest.fixture(scope="module")
def preprocessed_epochs():
    """
    Loading preprocessed test data for accorr optimization tests
    """
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")
    
    epo1 = mne.read_epochs(os.path.join(data_dir, "preproc_S1.fif"), preload=True)
    epo2 = mne.read_epochs(os.path.join(data_dir, "preproc_S2.fif"), preload=True)
    
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    epoch_merge = utils.merge(epo1, epo2)
    
    preprocessedTuple= namedtuple('preprocessed_epochs', ['pepo1', 'pepo2', 'pepochs_merge'])
    return preprocessedTuple(pepo1=epo1, pepo2=epo2, pepochs_merge=epoch_merge)


@pytest.fixture(scope="module")
def complex_signal(preprocessed_epochs):
    """
    Compute complex analytic signals for accorr testing.
    
    Returns complex_signal with shape (n_epochs, n_freq, 2*n_channels, n_times)
    ready for accorr computation.
    """
    # Define frequency bands
    freq_bands = OrderedDict({
        'Alpha-Low': [7.5, 11],
        'Alpha-High': [11.5, 13]
    })
    
    # Stack participant data
    data_inter = np.array([preprocessed_epochs.pepo1.get_data(), 
                           preprocessed_epochs.pepo2.get_data()])
    sampling_rate = preprocessed_epochs.pepo1.info['sfreq']
    
    # Compute frequency bands
    # Returns shape: (n_participants=2, n_epochs, n_freq, n_ch, n_times)
    complex_signal_raw = analyses.compute_freq_bands(
        data_inter,
        sampling_rate,
        freq_bands,
        filter_length=int(sampling_rate),
        l_trans_bandwidth=5.0,
        h_trans_bandwidth=5.0
    )
    
    n_epoch, n_ch, n_freq, n_samp = complex_signal_raw.shape[1], complex_signal_raw.shape[2], \
                                    complex_signal_raw.shape[3], complex_signal_raw.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal_reshaped = complex_signal_raw.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    
    return complex_signal_reshaped
