import pytest
import os
import mne
from hypyp import utils


@pytest.fixture(scope="module")
def epochs():
    """
    Loading data files & extracting sensor infos
    """
    epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"),
                           preload=True)
    epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"),
                           preload=True)
    mne.epochs.equalize_epoch_counts([epo1, epo2])
    epoch_merge = utils.merge(epo1, epo2)

    return epo1
