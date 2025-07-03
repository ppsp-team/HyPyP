from typing import List, Tuple
import pickle

from ..wavelet.base_wavelet import BaseWavelet
from .preprocessor.base_preprocessor import BasePreprocessor
from ..wavelet.coherence_data_frame import CoherenceDataFrame
from ..profiling import TimeTracker
from .dyad import Dyad, PairChannelMatchType
from ..plots import (
    plot_coherence_matrix,
    plot_coherence_bars_per_task,
    plot_coherence_connectogram,
    plot_coherence_connectogram_split,
)

class Study():
    dyads: List[Dyad]
    dyads_shuffled: List[Dyad]|None

    def __init__(self, dyads: List[Dyad] = []):
        """
        The Study object encapsulates the logic of an hyperscanning experiment.

        Args:
            dyads (List[Dyad], optional): List of all the dyads in the study. Defaults to [].
        """
        self.dyads = dyads
        self.dyads_shuffled = None

    @property
    def is_wtc_computed(self) -> bool:
        """True if the WTCs have been computed on all dyads in the study object"""
        for dyad in self.dyads:
            if not dyad.is_wtc_computed:
                return False
        return True

    @property
    def is_wtc_shuffle_computed(self) -> bool:
        """True if the WTCs have been computed on all shuffled dyads in the study object"""
        if self.dyads_shuffled is None:
            return False
        for dyad in self.dyads_shuffled:
            if not dyad.is_wtc_computed:
                return False
        return True
    
    @property
    def df(self) -> CoherenceDataFrame:
        """The pandas dataframe object from computed WTCs"""
        return self.get_coherence_df()

    def preprocess(self, preprocessor: BasePreprocessor):
        """
        Run the preprocess steps on every recordings in the study

        Args:
            preprocessor (BasePreprocessor): Which preprocessor to use. If no preprocessing is necessary, use MnePreprocessorUpstream()

        Returns:
            self: the Study object itself. Useful for chaining operations
        """
        for dyad in self.dyads:
            if not dyad.is_preprocessed:
                dyad.preprocess(preprocessor)
        return self
    
    def compute_wtcs(
        self,
        # All the arguments are a copy from Dyad.compute_wtcs, but we need them explicitely for type hinting
        wavelet: BaseWavelet|None = None,
        ch_match: PairChannelMatchType|None = None,
        only_time_range: Tuple[float,float]|None = None,
        bin_seconds: float|None = None,
        period_cuts: List[float]|None = None,
        verbose: bool = False,
        with_intra: bool = True,
        downsample: int|None = None,
        keep_wtcs: bool = True,
        show_time_estimation: bool = True,
    ): 
        """
        Wraps the `compute_wtcs` of all the dyads. Arguments are directly passed to the dyads method

        Args:
            wavelet (BaseWavelet | None): the wavelet to use for the wavelet transform. Defaults to ComplexMorletWavelet() without arguments.
            ch_match (PairMatchType | None, optional): string, list or regex to match channel name.
                                                Can be a tuple of 2 items if subject1 and subject2 have different matches.
                                                Defaults to None, which means all channels.
            only_time_range (Tuple[float,float] | None, optional): compute only a portion of the signal, defined as time range. Defaults to None.
            bin_seconds (float | None, optional): split the resulting WTC in time bins for balancing weights. Defaults to None.
            period_cuts (List[float] | None, optional): split the resulting WTC in period/frequency bins for balancing weights and finer analysis. Defaults to None.
            verbose (bool, optional): verbose flag. Defaults to False.
            with_intra (bool, optional): compute intra-subject as well. Defaults to False.
            downsample (int | None, optional): downsample in time the resulting WTC. Useful to save memory space and faster display. Defaults to None.
            keep_wtcs (bool, optional): if False, all the WTCs will be removed from object after the coherence dataframe has been computed. Useful to save memory space. Defaults to True.

        Returns:
            self: the Study object itself. Useful for chaining operations
        """
        for i, dyad in enumerate(self.dyads):
            if i == 0:
                tracker = TimeTracker()
                tracker.start()

            dyad.compute_wtcs(
                wavelet,
                ch_match,
                only_time_range,
                bin_seconds,
                period_cuts,
                verbose,
                with_intra,
                downsample,
                keep_wtcs,
            )

            if i == 0:
                tracker.stop()
                if show_time_estimation:
                    self._print_time_estimation(tracker.duration, len(self.dyads))
                
        return self
    
    def _print_time_estimation(self, single_duration, count):
            print(f'Time for computing one dyad: {TimeTracker.human_readable_duration(single_duration)}')
            print(f'Expected time for {count} dyads: {TimeTracker.human_readable_duration(single_duration * count)}')
    
    def estimate_wtcs_run_time(self, *args, **kwargs):
        """
        Computes the WTC for one dyad and print the expected run time for the whole study

        Returns:
            self: the Study object itself. Useful for chaining operations
        """
        dyad = self.dyads[0]
        tracker = TimeTracker()
        tracker.start()
        dyad.compute_wtcs(*args, **kwargs)
        tracker.stop()
        self._print_time_estimation(tracker.duration, len(self.dyads))

        return self
    
    def reset(self):
        self._clear_dyads_shuffle()
        for dyad in self.dyads:
            dyad.reset()
        return self
    
    def _clear_dyads_shuffle(self):
        """
        Delete all the shuffle dyads that have been created

        Returns:
            self: the Study object itself. Useful for chaining operations
        """
        self.dyads_shuffled = None
        return self
    
    def get_dyads_shuffle(self) -> List[Dyad]:
        """
        Get a list of permutated recording pairs, useful for statistical analysis.

        Returns:
            List[Dyad]: permulated pairs
        """
        dyads_shuffle = []
        for i, dyad1 in enumerate(self.dyads):
            for j, dyad2 in enumerate(self.dyads):
                if i == j:
                    continue
                dyads_shuffle.append(Dyad(dyad1.s1, dyad2.s2, label=f'shuffle s1:{dyad1.label}-s2:{dyad2.label}', is_pseudo=True))
        return dyads_shuffle

    # TODO add as argument the number of shuffle dyads
    def compute_wtcs_shuffle(self, *args, **kwargs):
        """
        Wraps the `compute_wtcs` of all the dyads_shuffle. Arguments are directly passed to the dyads_shuffle method

        Returns:
            self: the Study object itself. Useful for chaining operations
        """
        self.dyads_shuffled = self.get_dyads_shuffle()
        for dyad_shuffle in self.dyads_shuffled:
            dyad_shuffle.compute_wtcs(*args, **kwargs, with_intra=False)
        return self
    
    def get_coherence_df(self) -> CoherenceDataFrame:
        """
        Loop over every dyad and concatenate all the pandas dataframes into one

        If dyads_shuffle are computed, they are also included

        Raises:
            RuntimeError: all the WTCs must have been computed in order to get the pandas dataframe

        Returns:
            CoherenceDataFrame: typed pandas dataframe object
        """
        dfs = []
        if not self.is_wtc_computed:
            raise RuntimeError('wtc not computed')

        for dyad in self.dyads:
            dfs.append(dyad.df)

        if self.is_wtc_shuffle_computed:
            for dyad_shuffle in self.dyads_shuffled:
                dfs.append(dyad_shuffle.df)

        return CoherenceDataFrame.concat(dfs)
    
    #
    # Plots
    #
    def plot_coherence_matrix(
            self,
            field1:str='channel1',
            field2:str='channel2',
            query:str | None=None,
            s1_label:str='Subject1',
            s2_label:str='Subject2',
            **kwargs):
        """
        Plot the computed coherence metric for pair of fields (channel or roi) in a matrix format

        Args:
            field1 (str): name of the field in dataframe for x axis. Defaults to 'channel1'.
            field2 (str): name of the field in dataframe for y axis. Defaults to 'channel2'.
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        df = self.df
        if query is not None:
            df = df.query(query)
            
        ch_names1 = []
        ch_names2 = []

        for dyad in self.dyads:
            for ch_name in dyad.s1.ordered_ch_names:
                if ch_name not in ch_names1:
                    ch_names1.append(ch_name)

            for ch_name in dyad.s2.ordered_ch_names:
                if ch_name not in ch_names2:
                    ch_names2.append(ch_name)

        ordered_names = ch_names1 + [name for name in ch_names2 if name not in ch_names1] 
        # TODO: missing ordered roi

        return plot_coherence_matrix(
            df,
            s1_label,
            s2_label,
            field1,
            field2,
            ordered_names,
            **kwargs)
        
    def plot_coherence_matrix_per_channel(self, query:str|None=None, **kwargs):
        """
        Wraps plot_coherence_matrix to plot per channel

        Args:
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        return self.plot_coherence_matrix(
            'channel1',
            'channel2',
            query,
            **kwargs)
        
    def plot_coherence_matrix_per_roi(self, query:str|None=None, **kwargs):
        """
        Wraps plot_coherence_matrix to plot per region of interest

        Args:
            query (str | None, optional): pandas query to filter the dataframe. Defaults to None.
        """
        return self.plot_coherence_matrix(
            'roi1',
            'roi2',
            query,
            **kwargs)
    
    def plot_coherence_matrix_per_channel_for_task(self, task:str, **kwargs):
        """
        Wraps plot_coherence_matrix_per_channel to plot for a specific task

        Args:
            task (str): task name
        """
        return self.plot_coherence_matrix(
            'channel1',
            'channel2',
            query=f'task=="{task}"',
            **kwargs)
        
    def plot_coherence_matrix_per_roi_for_task(self, task:str, **kwargs):
        """
        Wraps plot_coherence_matrix_per_roi to plot for a specific task

        Args:
            task (str): task name
        """
        return self.plot_coherence_matrix(
            'roi1',
            'roi2',
            query=f'task=="{task}"',
            **kwargs)
    
    def plot_coherence_bars_per_task(self, **kwargs):
        """
        Plot coherence metric per task for comparison
        """
        return plot_coherence_bars_per_task(
            self.df,
            **kwargs)
        
    def plot_coherence_connectogram(
            self,
            query:str|None=None,
            title:str|None=None,
            s1_label:str='Subject1',
            s2_label:str='Subject2',
            **kwargs):
        df = self.df.copy()
        selector = df['is_intra']==False
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        # rename to have them separated in the plot
        df_filtered.loc[:, 'roi1'] = s1_label + '_' + df_filtered['roi1'].astype(str)
        df_filtered.loc[:, 'roi2'] = s2_label + '_' + df_filtered['roi2'].astype(str)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')

        if title is None:
            title = f'{s1_label} / {s2_label}'

        return plot_coherence_connectogram_split(
            pivot,
            title=title,
            **kwargs)

    def plot_coherence_connectogram_intra(self, is_intra_of:int, query:str|None=None, **kwargs):
        df = self.df
        selector = (df['is_intra']==True) & (df['is_intra_of']==is_intra_of)
        df_filtered = df[selector]

        if query is not None:
            df_filtered = df_filtered.query(query)

        pivot = df_filtered.pivot_table(index='roi1', columns='roi2', values='coherence', aggfunc='mean')
        return plot_coherence_connectogram(pivot, **kwargs)

    def plot_coherence_connectogram_s1(self, query:str|None=None, **kwargs):
        return self.plot_coherence_connectogram_intra(1, query, **kwargs)

    def plot_coherence_connectogram_s2(self, query:str|None=None, **kwargs):
        return self.plot_coherence_connectogram_intra(2, query, **kwargs)

    #
    # Disk serialisation
    #
    @staticmethod
    def from_pickle(file_path: str):
        """
        Reload a Study object from a serialized file

        Args:
            file_path (str): previously stored study object

        Returns:
            Study: the Study object
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, file_path: str):
        """
        Serialize the object to a local file

        Args:
            file_path (str): disk path for the serialisation
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def save_feather(self, file_path: str):
        """
        Save the pandas dataframe of coherence data to a .feather file

        Args:
            file_path (str): disk path for the .feather file
        """
        CoherenceDataFrame.save_feather(self.df, file_path)

    def save_csv(self, file_path: str):
        """
        Save the pandas dataframe of coherence data to a .csv file

        Args:
            file_path (str): disk path for the .csv file
        """
        self.df.to_csv(file_path)
