# coding=utf-8

import mne

def filt(epoch_concat):
    """Before ICA : Filter raw data to remove slow drifts
    Parameters
    ----------
    epoch_concat : instance of Epochs

    Returns
    -------
    epochs : instance of Epochs
    """
    epochs=[]
    for epoch_concat in epochs_concat: # per subj
        epochs.append(mne.filter.filter_data(epoch_concat,epoch_concat.info['sfreq'],l_freq=2., h_freq=None))
    return epochs