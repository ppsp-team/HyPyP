import pytest

import mne
import numpy as np

from hypyp import prep

def test_filt():
    n_channels = 10
    n_samples = 100000
    sfreq = 1000

    random_state = np.random.RandomState(seed=42)

    # create noise signal
    info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types='eeg')
    data = random_state.normal(size=(n_channels, n_samples))
    raw = mne.io.RawArray(data, info)
    raw_psd = raw.compute_psd().get_data()

    # compare power on lowest freq
    # Take the first few frequencies and not just index 0
    raw_filt_default, = prep.filt([raw])
    raw_filt_default_psd = raw_filt_default.compute_psd().get_data()
    assert np.sum(raw_filt_default_psd[:,:3]) < np.sum(raw_psd[:,:3])

    # compare power on highest freq
    raw_filt, = prep.filt([raw], (2., 10))
    raw_filt_psd = raw_filt.compute_psd().get_data()
    assert np.sum(raw_filt_psd[:,-1]) < np.sum(raw_filt_default_psd[:,-1])


@pytest.mark.parametrize("fit_kwargs", [
    dict(method='fastica', fit_params=dict(tol=0.01)), # increase tolerance to converge
    dict(method='infomax', fit_params=dict(extended=True))
])
def test_ICA(epochs, fit_kwargs):
    ep = [epochs.epo1, epochs.epo2]
    icas = prep.ICA_fit(ep, n_components=15, **fit_kwargs, random_state=97)
    
    assert len(icas) == len(ep)

    # check that the number of componenents is similar between the two participants
    for i in range(0, len(icas)-1):
        assert mne.preprocessing.ICA.get_components(
            icas[i]).shape == mne.preprocessing.ICA.get_components(icas[i+1]).shape

    cleaned_epochs_ICA = prep.ICA_apply(icas, 0, 0, ep, plot=False)

    # check bad channels are not deleted
    assert epochs.epo1.info['ch_names'] == cleaned_epochs_ICA[0].info['ch_names']
    assert epochs.epo2.info['ch_names'] == cleaned_epochs_ICA[1].info['ch_names']

    # check signal change by comparing total amplitude
    raw_amplitudes = np.mean(np.abs(epochs.epo1.get_data(copy=True)), axis=1)
    processed_amplitudes = np.mean(np.abs(cleaned_epochs_ICA[0].get_data(copy=True)), axis=1)
    assert np.sum(processed_amplitudes) < np.sum(raw_amplitudes)


@pytest.mark.parametrize("AR_local_kwargs", [
    dict(strategy='union'),
    dict(strategy='intersection'),
])
def test_AR_local(epochs, AR_local_kwargs):
    # test on epochs, but usually applied on cleaned epochs with ICA
    cleaned_epochs_AR, dic_AR = prep.AR_local(
        [epochs.epo1, epochs.epo2],
        **AR_local_kwargs,
        threshold=50.0,
        verbose=False
    )
    assert len(epochs.epo1) >= len(cleaned_epochs_AR[0])
    assert len(epochs.epo2) >= len(cleaned_epochs_AR[1])
    assert len(cleaned_epochs_AR[0]) == len(cleaned_epochs_AR[1])
    assert dic_AR['S2'] + dic_AR['S1'] == dic_AR['dyad']
    assert dic_AR['S2'] <= dic_AR['dyad']


@pytest.mark.parametrize("AR_local_kwargs", [
    dict(strategy='union'),
    dict(strategy='intersection'),
])
def test_AR_local_exception(epochs, AR_local_kwargs):
    # test the threshold
    with pytest.raises(Exception):
        prep.AR_local(
            [epochs.epo1, epochs.epo2],
            **AR_local_kwargs,
            threshold=0.0,
            verbose=False
        )


