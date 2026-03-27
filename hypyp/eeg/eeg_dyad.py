import warnings

import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

from .eeg_step import PREPROCESS_STEP_AR, PREPROCESS_STEP_ICA_APPLY, PREPROCESS_STEP_RAW, EEGStep, EEGDyadStep

from ..dataclasses.psd import SpectralPower
from ..dataclasses.freq_band import FreqBands
from ..connectivity.connectivities import Connectivities
from ..connectivity.connectivity import Connectivity
from ..core.base_dyad import BaseDyad
from ..signal.complex_signal import ComplexSignal

from ..utils import create_epochs, merge, split
from ..prep import ICA_fit, ICA_apply, AR_local, DicAR
from ..analyses import pow, compute_sync


DEFAULT_EPOCHS_DURATION = 1

class EEGDyad(BaseDyad):
    label: str
    steps: list[EEGDyadStep]
    sfreq: float
    raws: list[mne.io.Raw] | None
    icas: list[ICA] | None
    icas_applied: list[str]
    dic_ar: DicAR | None
    psds: list[SpectralPower]
    complex_signal: ComplexSignal | None

    # analysis results
    connectivities_per_mode: dict[str, Connectivities]

    def __init__(self, label: str = None):
        super().__init__()
        if label is None:
            self.label = 'no label'
        else:
            self.label = label 

        self.steps = []
        self.sfreq = -1
        self.raws = None
        self.icas = None
        self.icas_applied = []
        self.dic_ar = None
        self.psds = None
        self.complex_signal = None
        self.connectivities_per_mode = {}
    
    @staticmethod
    def from_files(file_path1: str, file_path2: str, label: str = None, verbose: bool = False):
        self = EEGDyad(label)
        if file_path1.endswith('-epo.fif') and file_path2.endswith('-epo.fif'):
            if verbose:
                print(f"Loading EEGDyad from epoch files {file_path1} and {file_path2}")

            self.epos_add_step([
                mne.read_epochs(file_path1, preload=True, verbose=verbose),
                mne.read_epochs(file_path2, preload=True, verbose=verbose),
            ], PREPROCESS_STEP_RAW)
            assert self.epo1.info['sfreq'] == self.epo2.info['sfreq']
            self.sfreq = self.epo1.info['sfreq']
            self._equalize_epoch_counts(verbose=verbose)
        else:
            raise NotImplementedError()

        return self
    
    @staticmethod
    def from_epochs(epo1: mne.Epochs, epo2: mne.Epochs, label: str = None, verbose: bool = False) -> 'EEGDyad':
        assert epo1.info['sfreq'] == epo2.info['sfreq']
        if verbose:
            print(f"Loading EEGDyad from preloaded epochs")

        self = EEGDyad(label)
        self.sfreq = epo1.info['sfreq']
        self.epos_add_step([epo1, epo2], PREPROCESS_STEP_RAW)
        self._equalize_epoch_counts(verbose=verbose)
        return self
    
    @staticmethod
    def from_raws(raw1: mne.io.Raw, raw2: mne.io.Raw, label: str = None, epochs_duration: float = DEFAULT_EPOCHS_DURATION, verbose: bool = False) -> 'EEGDyad':
        assert raw1.info['sfreq'] == raw2.info['sfreq']
        if verbose:
            print(f"Loading EEGDyad from raw data")
        self = EEGDyad(label)
        self.sfreq = raw1.info['sfreq']
        self.raws = [raw1, raw2]
        self.create_epochs_from_raws(epochs_duration, verbose=verbose)
        self._equalize_epoch_counts(verbose=verbose)
        return self
    
    @staticmethod
    def from_raw_merge(raw_merge: mne.io.Raw, label: str = None, epochs_duration: float = DEFAULT_EPOCHS_DURATION, verbose: bool = False) -> 'EEGDyad':
        self = EEGDyad(label)
        if verbose:
            print(f"Loading EEGDyad from raw merge")
        self.sfreq = raw_merge.info['sfreq']
        self.raws = split(raw_merge)
        self.create_epochs_from_raws(epochs_duration, verbose=verbose)
        return self

    def _assert_has_epochs(self):
        if len(self.steps) == 0:
            raise RuntimeError('Epochs not created. Use create_epochs(duration)')
    
    def _assert_has_raws(self):
        if self.raws is None:
            raise RuntimeError('No raw data.')
    
    def _assert_has_icas(self):
        if self.icas is None:
            raise RuntimeError('ICAs is None. Make sure to call "ica_fit()" first.')
    
    def _assert_has_psds(self):
        if self.psds is None:
            raise RuntimeError('ICAs is None. Make sure to call "ica_fit()" first.')
    
    def _equalize_epoch_counts(self, verbose: bool = False):
        if verbose:
            print("Equalizing epochs")
        if len(self.epo1) != len(self.epo2):
            warnings.warn(f"The 2 epochs objects don't have the same length: {len(self.epo1)} != {len(self.epo2)}")
            mne.epochs.equalize_epoch_counts(self.epos)
    
    @property
    def epos(self):
        return self.steps[-1].epos
    
    @property
    def epochs_merged(self) -> mne.Epochs:
        self._assert_has_epochs()
        return merge(self.epo1, self.epo2)
    
    @property
    def epo1(self) -> mne.Epochs:
        self._assert_has_epochs()
        return self.epos[0]
    
    @property
    def epo2(self) -> mne.Epochs:
        self._assert_has_epochs()
        return self.epos[1]
    
    @property
    def raw1(self) -> mne.io.Raw:
        self._assert_has_raws()
        return self.raws[0]
    
    @property
    def raw2(self) -> mne.io.Raw:
        self._assert_has_raws()
        return self.raws[1]
    
    @property
    def ica1(self) -> ICA:
        self._assert_has_icas()
        return self.icas[0]
    
    @property
    def ica2(self) -> ICA:
        self._assert_has_icas()
        return self.icas[1]
    
    @property
    def psds1(self) -> list[SpectralPower]:
        self._assert_has_psds()
        return self.psds[::2]
    
    @property
    def psds2(self) -> list[SpectralPower]:
        self._assert_has_psds()
        return self.psds[1::2]
    
    @property
    def is_icas_computed(self) -> bool:
        return self.icas is not None
    
    @property
    def is_icas_applied(self) -> bool:
        return self.icas is not None
    
    @property
    def is_autoreject_applied(self) -> bool:
        return self.dic_ar is not None
    
    @property
    def is_psds_computed(self) -> bool:
        return self.psds is not None
    
    @property
    def is_complex_signal_computed(self) -> bool:
        return self.complex_signal is not None
    
    @property
    def is_connectivity_computed(self) -> bool:
        return len(self.connectivities_per_mode.keys()) > 0
    
    @property
    def connectivity_modes(self) -> list[str]:
        return list(self.connectivities_per_mode.keys())
    
    def epos_add_step(self, epos, name: str = ''):
        single_steps = [EEGStep(epo.copy(), name=name) for epo in epos]
        self.steps.append(EEGDyadStep(single_steps, name=name))

    def create_epochs_from_raws(self, duration:float=DEFAULT_EPOCHS_DURATION, verbose: bool = False) -> 'EEGDyad':
        list1, list2 = create_epochs([self.raw1], [self.raw2], duration)

        if verbose:
            print(f"Created {len(list1)} and {len(list2)} epochs of duration {DEFAULT_EPOCHS_DURATION} seconds")

        self.epos_add_step([list1[0], list2[0]], PREPROCESS_STEP_RAW)
        return self

    def prep_ica_fit(
            self,
            n_components: int,
            method: str = 'infomax',
            fit_params: dict = dict(extended=True),
            random_state: int = 42,
            **kwargs,
    ) -> 'EEGDyad':
        self.icas = ICA_fit(self.epos, n_components, method, fit_params, random_state, **kwargs)
        return self
    
    def prep_ica_apply(
            self,
            subject_idx: int,
            component_idx: int,
            label: str = 'blink',
            ch_type: str = 'eeg',
            threshold: float = 0.9,
            plot: bool = False,
    ) -> 'EEGDyad':
        self._assert_has_icas()
        self.epos_add_step(ICA_apply(self.icas, subject_idx, component_idx, self.epos, plot=plot, label=label, ch_type=ch_type, threshold=threshold), PREPROCESS_STEP_ICA_APPLY)
        self.icas_applied.append(label)
        return self
    
    def prep_autoreject_apply(self, strategy: str = None, threshold: float = None, show: bool = None, **kwargs) -> 'EEGDyad':
        # Forward arguments to underlying function
        if strategy is not None:
            kwargs['strategy'] = strategy
        if threshold is not None:
            kwargs['threshold'] = threshold
        if show is not None:
            # AR_local use the "verbose" argument to control plot display
            kwargs['verbose'] = show
        
        epos, dic_ar = AR_local(self.epos, **kwargs)
        self.epos_add_step(epos, PREPROCESS_STEP_AR)
        self.dic_ar = dic_ar
        return self

    def analyse_pow(
            self,
            freq_bands: FreqBands,
            n_fft: int = None,
            n_per_seg: int = None,
            epochs_average: bool = True,
            show: bool = False,
            **kwargs,
    ) -> 'EEGDyad':
        if n_fft is None:
            n_fft = int(self.sfreq)
        if n_per_seg is None:
            n_per_seg = int(self.sfreq)

        self.psds = []
        for freq_band in freq_bands:
            for epo in self.epos:
                freqs, psd = pow(epo, fmin=freq_band.fmin, fmax=freq_band.fmax, n_fft=n_fft, n_per_seg=n_per_seg, epochs_average=epochs_average, **kwargs)
                self.psds.append(SpectralPower(freqs, psd, epo.ch_names, freq_band.name))
        
        if show:
            self.plot_psds()

        return self
    

    def compute_complex_signal_freq_bands(self, freq_bands: FreqBands = None, **kwargs):
        if freq_bands is not None:
            kwargs['freq_bands'] = freq_bands

        self.complex_signal = ComplexSignal(self.epos, self.sfreq, **kwargs)
        return self


    def analyse_connectivity(self, mode: str, epochs_average: bool = True):
        assert self.complex_signal is not None
        matrix = compute_sync(self.complex_signal.data, mode = mode, epochs_average = epochs_average)
        self.connectivities_per_mode[mode] = Connectivities(mode, self.complex_signal.freq_bands, matrix, (self.epo1.ch_names, self.epo2.ch_names))
        return self

    def get_synchrony_time_series(self):
        raise NotImplementedError()

    def plot_icas_components(self) -> 'EEGDyad':
        for i, ica in enumerate(self.icas):
            print(f"Subject idx: {i}")
            ica.plot_components()
        return
    
    def plot_psds(self) -> 'EEGDyad':
        n_rows = len(self.psds) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 12), sharey=True)

        for i, power in enumerate(self.psds[::-1]):
            band_idx = i // 2
            subject_idx = i % 2
            subject_id = subject_idx + 1
            ax = axes[band_idx, subject_idx]
            power.plot(ax=ax, title=f"Subject {subject_id} / {power.band_display_name} ({power.freqs[0]}-{power.freqs[-1]}Hz)")

        fig.suptitle('Average power in EEG Frequency Bands')
        return 
    
    def __repr__(self):
        nl = "\n" # cannot have backslashes in multiline below
        return f"""EEGDyad
  label: {self.label}
  sfreq: {self.sfreq}
  n epochs initial: s1: {len(self.steps[0].epos[0])}, s2: {len(self.steps[0].epos[1])}
  n epochs remaining: s1: {len(self.steps[-1].epos[0])}, s2: {len(self.steps[-1].epos[1])}
  steps: 
{nl.join([f"  - {step.name}" for step in self.steps])}
  icas computed: {'yes' if self.is_icas_computed else 'no'}
  icas applied:
{nl.join([f"  - {label}" for label in self.icas_applied])}
  autoreject applied: {f"yes ({self.dic_ar['dyad']:.1f}% rejected)" if self.is_autoreject_applied else 'no'}
  psd computed: {'yes' if self.is_psds_computed else 'no'}
{nl.join([f"  - {psd.band_name}" for psd in self.psds1]) if self.is_psds_computed else ""}
  complex signal computed: {'yes' if self.is_complex_signal_computed else 'no'}
  connectivity computed: {'yes' if self.is_connectivity_computed else 'no'}
{nl.join([f"  - {mode}" for mode in self.connectivity_modes])}
"""
    # steps: list[EEGDyadStep]
    # sfreq: float
    # raws: list[mne.io.Raw] | None
    # icas: list[ICA] | None
    # dic_ar: DicAR | None
    # psds: list[PSD]
    # complex_signal: ComplexSignal | None
    #    self.icas = None
    #    self.dic_ar = None
    #    self.psds = None
    #    self.complex_signal = None
    #    self.connectivities = {}