from typing import List, Self
import pickle

from hypyp.fnirs.preprocessor.base_preprocessor import BasePreprocessor
from hypyp.wavelet.coherence_data_frame import CoherenceDataFrame


from .dyad import Dyad

class Cohort():
    dyads: List[Dyad]
    dyads_shuffle: List[Dyad]|None

    def __init__(self, dyads: List[Dyad] = []):
        """
        The Cohort object encapsulates the logic of an hyperscanning experiment.

        Args:
            dyads (List[Dyad], optional): List of all the dyads in the cohort. Defaults to [].
        """
        self.dyads = dyads
        self.dyads_shuffle = None

    @property
    def is_wtc_computed(self) -> bool:
        """True if the WTCs have been computed on all dyads in the cohort object"""
        for dyad in self.dyads:
            if not dyad.is_wtc_computed:
                return False
        return True

    @property
    def is_wtc_shuffle_computed(self) -> bool:
        """True if the WTCs have been computed on all shuffled dyads in the cohort object"""
        if self.dyads_shuffle is None:
            return False
        for dyad in self.dyads_shuffle:
            if not dyad.is_wtc_computed:
                return False
        return True
    
    @property
    def df(self) -> CoherenceDataFrame:
        """The pandas dataframe object from computed WTCs"""
        return self.get_coherence_df()

    def preprocess(self, preprocessor: BasePreprocessor) -> Self:
        """
        Run the preprocess steps on every subject in the cohort

        Args:
            preprocessor (BasePreprocessor): Which preprocessor to use. If no preprocessing is necessary, use MnePreprocessorUpstream()

        Returns:
            Self: the object itself. Useful for chaining operations
        """
        for dyad in self.dyads:
            if not dyad.is_preprocessed:
                dyad.preprocess(preprocessor)
        return self
    
    def compute_wtcs(self, *args, **kwargs) -> Self:
        """
        Wraps the `compute_wtcs` of all the dyads. Arguments are directly passed to the dyads method

        Returns:
            Self: the object itself. Useful for chaining operations
        """
        for dyad in self.dyads:
            dyad.compute_wtcs(*args, **kwargs)
        
        return self
    
    def clear_dyads_shuffle(self) -> Self:
        """
        Delete all the shuffle dyads that have been created

        Returns:
            Self: _description_
        """
        self.dyads_shuffle = None
    
    def get_dyads_shuffle(self) -> List[Dyad]:
        """
        Get a list of permutated subject pairs, useful for statistical analysis.

        Returns:
            List[Dyad]: permulated pairs
        """
        dyads_shuffle = []
        for i, dyad1 in enumerate(self.dyads):
            for j, dyad2 in enumerate(self.dyads):
                if i == j:
                    continue
                dyads_shuffle.append(Dyad(dyad1.s1, dyad2.s2, label=f'shuffle s1:{dyad1.label}-s2:{dyad2.label}', is_shuffle=True))
        return dyads_shuffle

    # TODO add as argument the number of shuffle dyads
    def compute_wtcs_shuffle(self, *args, **kwargs) -> Self:
        """
        Wraps the `compute_wtcs` of all the dyads_shuffle. Arguments are directly passed to the dyads_shuffle method

        Returns:
            Self: the object itself. Useful for chaining operations
        """
        self.dyads_shuffle = self.get_dyads_shuffle()
        for dyad_shuffle in self.dyads_shuffle:
            dyad_shuffle.compute_wtcs(*args, **kwargs)
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
            for dyad_shuffle in self.dyads_shuffle:
                dfs.append(dyad_shuffle.df)

        return CoherenceDataFrame.concat(dfs)
    
    #
    # Disk serialisation
    #
    @staticmethod
    def from_pickle(file_path: str)-> Self:
        """
        Reload a Cohort object from a serialized file

        Args:
            file_path (str): previously stored cohort object

        Returns:
            Cohort: the Cohort object
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
