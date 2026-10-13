# Test data

`preproc_S1.fif` and `preproc_S2.fif` are the first 33 epochs of the two
public sample recordings that `hypyp.datasets.eeg_epochs(participant=1)` and
`eeg_epochs(participant=2)` fetch from the HypypData repository
(`mne/epochs/participant1-epo.fif`, `participant2-epo.fif`): 31 EEG channels
at 500 Hz, one second per epoch, the same measurement date and channel names
as those files. They carry no `subject_info`. They are here so that the tests
that only need an `Info` with sensor positions, or a small amount of epoched
data, run without a download.

Do not add recordings of real participants to this directory; new fixtures
are synthetic (see `tests/xdf_writer.py` and the `make_epochs` helpers in the
test files).
