#!/usr/bin/env python
# coding=utf-8

"""
Statistical functions

| Option | Description |
| ------ | ----------- |
| title           | stats.py |
| authors         | Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


from collections import namedtuple
from typing import List, Tuple
import numpy as np
import scipy
from scipy.stats import f as f_dist
import matplotlib.pylab as plt
import mne
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_test


def statsCond(data: np.ndarray, epochs: mne.Epochs, n_permutations: int, alpha: float) -> namedtuple:
    """
    Perform statistical t-test on participant measures (e.g., PSD) for a condition.
    
    This function tests whether the observed mean significantly deviates from 0
    using a permutation-based t-test with False Discovery Rate (FDR) correction
    for multiple comparisons.
    
    Parameters
    ----------
    data : np.ndarray
        Array of participant measures with shape (n_samples, n_tests, n_freq)
        where n_tests typically represents channels and n_freq the frequencies.
        Values will be averaged across frequencies for statistics.
        
    epochs : mne.Epochs
        Epochs object for a condition from a random participant, used only
        to access information like channel positions.
        
    n_permutations : int
        Number of permutations for the statistical test
        Should be at least 2*n_samples, e.g., 50000
        
    alpha : float
        Significance threshold for the test, e.g., 0.05
    
    Returns
    -------
    statsCondTuple : namedtuple
        A named tuple containing:
        - T_obs: T-statistic observed for all variables, array of shape (n_tests)
        - p_values: p-values for all tests, array of shape (n_tests)
        - H0: T-statistics obtained by permutations, array of shape (n_permutations)
        - adj_p: tuple with boolean assessment of significance and FDR-corrected p-values
        - T_obs_plot: statistical values for significant sensors, array of shape (n_tests)
    
    Notes
    -----
    This test calculates if the observed mean significantly deviates from 0;
    it doesn't compare two periods, but tests one period against the null hypothesis.
    
    Randomized data are generated with random sign flips, and the test is two-tailed
    by default (the alternative hypothesis is that the data mean is different from 0).
    
    To reduce false positives due to multiple comparisons, False Discovery Rate (FDR)
    correction is applied to the p-values.
    
    The frequency dimension is reduced to one for the test (average in the frequency
    band-of-interest). To take frequencies into account, use cluster statistics
    (see statscondCluster function).
    
    Examples
    --------
    >>> # Independent t-test between two groups
    >>> ind_ttest_results = statscluster(
    ...     [group1_data, group2_data],
    ...     test='ind ttest',
    ...     factor_levels=None,
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=0,  # two-tailed test
    ...     n_permutations=10000,
    ...     alpha=0.05
    ... )
    >>> 
    >>> # 2×2 repeated measures ANOVA (within-subjects design)
    >>> # Factor 1: Condition (2 levels), Factor 2: Time (2 levels)
    >>> anova_results = statscluster(
    ...     data_array_2x2,  # Shape: (4, n_subjects, n_features)
    ...     test='f multipleway',
    ...     factor_levels=[2, 2],
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=1,  # F-tests are one-tailed
    ...     n_permutations=10000,
    ...     alpha=0.05
    ... )
    """

    # checking whether data have the same size
    assert(len(data.shape) == 3), "PSD does not have the appropriate shape!"

    # averaging across frequencies (compute stats only in ch space)
    power = np.mean(data, axis=2)
    T_obs, p_values, H0 = mne.stats.permutation_t_test(power, n_permutations,
                                                       tail=0, n_jobs=1)
    adj_p = mne.stats.fdr_correction(p_values, alpha=alpha, method='indep')

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c in adj_p[1]:
        if c <= alpha:
            i = np.where(adj_p[1] == c)
            T_obs_plot[i] = T_obs[i]
    T_obs_plot = np.nan_to_num(T_obs_plot)

    # retrieving sensor position
    pos = np.array([[0, 0]])
    for i in range(0, len(epochs.info['ch_names'])):
        cor = np.array([epochs.info['chs'][i]['loc'][0:2]])
        pos = np.concatenate((pos, cor), axis=0)
    pos = pos[1:]

    statsCondTuple = namedtuple(
        'statsCond', ['T_obs', 'p_values', 'H0', 'adj_p', 'T_obs_plot'])

    return statsCondTuple(
        T_obs=T_obs,
        p_values=p_values,
        H0=H0,
        adj_p=adj_p,
        T_obs_plot=T_obs_plot)


def con_matrix(epochs: mne.Epochs, freqs_mean: List[float], draw: bool = False) -> namedtuple:
    """
    Compute a priori channel connectivity across space and frequencies.
    
    This function creates connectivity matrices that define which channels and
    frequencies should be considered neighbors for cluster-based statistics.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing channel information
        
    freqs_mean : List[float]
        List of frequencies in the frequency-band-of-interest used for
        power or coherence spectral density calculation
        
    draw : bool, optional
        Whether to plot the connectivity matrices (default=False)
    
    Returns
    -------
    con_matrixTuple : namedtuple
        A named tuple containing:
        - ch_con: Connectivity matrix between channels in space,
          scipy.sparse.csr_matrix of shape (n_channels, n_channels)
        - ch_con_freq: Connectivity matrix between channels across space and
          frequencies, scipy.sparse.csr_matrix of shape
          (n_channels*len(freqs_mean), n_channels*len(freqs_mean))
    
    Notes
    -----
    The channel connectivity matrix (ch_con) is based on the spatial adjacency
    of EEG electrodes - channels that are physically adjacent are considered
    connected.
    
    The frequency-space connectivity matrix (ch_con_freq) extends this spatial
    adjacency to include frequency adjacency - neighboring frequencies for the
    same channel are also considered connected.
    
    These connectivity matrices are used as inputs to cluster-based statistical
    functions to define the neighborhood structure for clustering.
    
    Examples
    --------
    >>> # Create connectivity matrices for alpha band frequencies
    >>> alpha_freqs = np.arange(8, 13)
    >>> conn = con_matrix(epochs, alpha_freqs, draw=True)
    >>> ch_con = conn.ch_con  # Channel spatial connectivity
    >>> ch_con_freq = conn.ch_con_freq  # Channel-frequency connectivity
    """

    # creating channel-to-channel connectivity matrix in space
    ch_con, ch_names_con = find_ch_adjacency(epochs.info,
                                                ch_type='eeg')

    ch_con_arr = ch_con.toarray()

    # duplicating the array 'freqs_mean' or 'freqs' times (PSD or CSD)
    # to take channel connectivity across neighboring frequencies into
    # account
    l_freq = len(freqs_mean)
    init = np.zeros((l_freq*len(ch_names_con),
                     l_freq*len(ch_names_con)))
    for i in range(0, l_freq*len(ch_names_con)):
        for p in range(0, l_freq*len(ch_names_con)):
            if (p//len(ch_names_con) == i//len(ch_names_con)) or (p//len(ch_names_con) == i//len(ch_names_con) + 1) or (p//len(ch_names_con) == i//len(ch_names_con) - 1):
                init[i][p] = 1

    ch_con_mult = np.tile(ch_con_arr, (l_freq, l_freq))
    ch_con_freq = np.multiply(init, ch_con_mult)

    if draw:
        plt.figure()
        # visualizing the matrix and transforming it into array
        plt.subplot(1, 2, 1)
        plt.spy(ch_con)
        plt.title("Connectivity matrix")
        plt.subplot(1, 2, 2)
        plt.spy(ch_con_freq)
        plt.title("Meta-connectivity matrix")

    con_matrixTuple = namedtuple('con_matrix', ['ch_con', 'ch_con_freq'])

    return con_matrixTuple(
        ch_con=ch_con,
        ch_con_freq=ch_con_freq)


def metaconn_matrix_2brains(electrodes: List[Tuple[int, int]], ch_con: scipy.sparse.csr_matrix, 
                           freqs_mean: List[float], plot: bool = False) -> namedtuple:
    """
    Compute a priori connectivity matrices for hyperscanning analyses.
    
    This function creates connectivity matrices for pairs of channels across
    two brains (participants), taking into account spatial adjacency within
    each brain but assuming no direct connectivity between brains.
    
    Parameters
    ----------
    electrodes : List[Tuple[int, int]]
        List of electrode pairs for which connectivity indices have been computed.
        Each tuple contains the indices of channels from participant 1 and participant 2.
        
    ch_con : scipy.sparse.csr_matrix
        Connectivity matrix between channels in space, typically from con_matrix()
        
    freqs_mean : List[float]
        List of frequencies in the frequency-band-of-interest
        
    plot : bool, optional
        Whether to plot the connectivity matrices (default=False)
    
    Returns
    -------
    metaconn_matrix_2brainsTuple : namedtuple
        A named tuple containing:
        - metaconn: Connectivity matrix between channel pairs,
          array of shape (len(electrodes), len(electrodes))
        - metaconn_freq: Connectivity matrix between channel pairs across
          space and frequencies, array of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean))
    
    Notes
    -----
    This function assumes there is no a priori connectivity between channels
    from different participants. It considers two channel pairs connected if:
    1. The respective channels within each participant are connected, or
    2. Some channels are identical across the pairs
    
    The resulting connectivity matrices define the neighborhood structure for
    cluster-based statistics on hyperscanning data.
    
    Examples
    --------
    >>> # Create metaconnectivity matrices for interbrain connectivity
    >>> electrode_pairs = indices_connectivity_interbrain(epochs_hyper)
    >>> metaconn = metaconn_matrix_2brains(
    ...     electrode_pairs, ch_con.ch_con, freqs_mean=[10], plot=True
    ... )
    """

    n = np.max(electrodes, axis=0)[0]+1
    # n = 62
    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11, e12) in enumerate(electrodes):
        for ne2, (e21, e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            # considering no a priori connectivity between the 2 brains
            metaconn[ne1, ne2] = (((ch_con[e11, e21]) and (ch_con[e12-n, e22-n])) or
                                  ((ch_con[e11, e21]) and (e12 == e22)) or
                                  ((ch_con[e12-n, e22-n]) and (e11 == e21)) or
                                  ((e12 == e22) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channel connectivity
    # across neighboring frequencies into account
    l_freq = len(freqs_mean)

    init = np.zeros((l_freq*len(electrodes),
                     l_freq*len(electrodes)))
    for i in range(0, l_freq*len(electrodes)):
        for p in range(0, l_freq*len(electrodes)):
            if (p//len(electrodes) == i//len(electrodes)) or (p//len(electrodes) == i//len(electrodes) + 1) or (p//len(electrodes) == i//len(electrodes) - 1):
                init[i][p] = 1

    metaconn_mult = np.tile(metaconn, (l_freq, l_freq))
    metaconn_freq = np.multiply(init, metaconn_mult)

    if plot:
        # vizualising the array
        plt.figure()
        plt.spy(metaconn_freq)
        plt.title("Meta-connectivity matrix")

    metaconn_matrix_2brainsTuple = namedtuple(
        'metaconn_matrix_2brains', ['metaconn', 'metaconn_freq'])

    return metaconn_matrix_2brainsTuple(
        metaconn=metaconn,
        metaconn_freq=metaconn_freq)


def metaconn_matrix(electrodes: List[Tuple[int, int]], ch_con: scipy.sparse.csr_matrix, 
                   freqs_mean: List[float]) -> namedtuple:
    """
    Compute a priori connectivity between pairs of sensors within one brain.
    
    This function creates connectivity matrices for pairs of channels within
    a single brain, taking into account spatial adjacency for cluster-based statistics.
    
    Parameters
    ----------
    electrodes : List[Tuple[int, int]]
        List of electrode pairs for which connectivity indices have been computed.
        Each tuple contains the indices of two channels from the same participant.
        
    ch_con : scipy.sparse.csr_matrix
        Connectivity matrix between channels in space, typically from con_matrix()
        
    freqs_mean : List[float]
        List of frequencies in the frequency-band-of-interest
    
    Returns
    -------
    metaconn_matrixTuple : namedtuple
        A named tuple containing:
        - metaconn: Connectivity matrix between channel pairs,
          array of shape (len(electrodes), len(electrodes))
        - metaconn_freq: Connectivity matrix between channel pairs across
          space and frequencies, array of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean))
    
    Notes
    -----
    This function determines whether two channel pairs are connected based on
    the spatial adjacency of their constituent channels. It considers various 
    combinations of adjacency between the channels.
    
    The resulting connectivity matrices define the neighborhood structure for
    cluster-based statistics on connectivity data within a single brain.
    
    Examples
    --------
    >>> # Create metaconnectivity matrices for intrabrain connectivity
    >>> electrode_pairs = indices_connectivity_intrabrain(epochs)
    >>> metaconn = metaconn_matrix(
    ...     electrode_pairs, ch_con.ch_con, freqs_mean=[10]
    ... )
    """

    metaconn = np.zeros((len(electrodes), len(electrodes)))
    for ne1, (e11, e12) in enumerate(electrodes):
        for ne2, (e21, e22) in enumerate(electrodes):
            # print(ne1,e11,e12,ne2,e21,e22)
            metaconn[ne1, ne2] = (((ch_con[e11, e21]) and (ch_con[e12, e22])) or
                                  ((ch_con[e11, e22]) and (ch_con[e12, e21])) or
                                  ((ch_con[e11, e21]) and (e12 == e22)) or
                                  ((ch_con[e11, e22]) and (e12 == e21)) or
                                  ((ch_con[e12, e21]) and (e11 == e22)) or
                                  ((ch_con[e12, e22]) and (e11 == e21)))

    # duplicating the array 'freqs_mean' times to take channels connectivity
    # across neighboring frequencies into account
    l_freq = len(freqs_mean)

    init = np.zeros((l_freq*len(electrodes),
                     l_freq*len(electrodes)))
    for i in range(0, l_freq*len(electrodes)):
        for p in range(0, l_freq*len(electrodes)):
            if (p//len(electrodes) == i//len(electrodes)) or (p//len(electrodes) == i//len(electrodes) + 1) or (p//len(electrodes) == i//len(electrodes) - 1):
                init[i][p] = 1

    metaconn_mult = np.tile(metaconn, (l_freq, l_freq))
    metaconn_freq = np.multiply(init, metaconn_mult)

    # TODO: option with verbose
    # vizualising the array
    plt.spy(metaconn_freq)

    metaconn_matrixTuple = namedtuple(
        'metaconn_matrix', ['metaconn', 'metaconn_freq'])

    return metaconn_matrixTuple(
        metaconn=metaconn,
        metaconn_freq=metaconn_freq)


def statscondCluster(data: list, freqs_mean: list, ch_con_freq: scipy.sparse.csr_matrix, 
                    tail: int, n_permutations: int, alpha: float = 0.05) -> namedtuple:
    """
    Perform cluster-level statistical permutation test on neurophysiological data.
    
    This function applies a cluster-based permutation test to identify significant
    differences between conditions, correcting for multiple comparisons by taking
    into account connectivity across space and frequencies.
    
    Parameters
    ----------
    data : list
        List of arrays containing values from different conditions or groups
        to compare. Each array has shape (n_observations, n_features)
        
    freqs_mean : list
        Frequencies in the frequency-band-of-interest
        
    ch_con_freq : scipy.sparse.csr_matrix
        Connectivity or metaconnectivity matrix defining adjacency across
        space and frequencies, typically from con_matrix() or metaconn_matrix()
        
    tail : int
        Direction of the test:
        - 0: two-tailed test
        - 1: one-tailed test (greater)
        - -1: one-tailed test (less)
        
    n_permutations : int
        Number of permutations for the statistical test, e.g., 50000
        
    alpha : float, optional
        Significance threshold for clusters (default=0.05)
    
    Returns
    -------
    statscondClusterTuple : namedtuple
        A named tuple containing:
        - F_obs: Observed F-statistic for all variables, array of shape (n_features)
        - clusters: Boolean array indicating locations in significant clusters
        - cluster_p_values: p-value for each identified cluster
        - H0: Max cluster-level statistics under permutation, array of shape (n_permutations)
        - F_obs_plot: F-values for significant sensors, array of shape (n_features)
    
    Notes
    -----
    This function uses MNE's permutation_cluster_test to perform the analysis.
    
    With t_power=1 (default), each location is weighted by its statistical score
    within a cluster, which gives more weight to stronger effects.
    
    The function automatically calculates appropriate thresholds for cluster
    formation based on the F-distribution for two-tailed tests.
    
    Examples
    --------
    >>> # Compare alpha power between two conditions
    >>> cluster_stats = statscondCluster(
    ...     [condition1_data, condition2_data],
    ...     freqs_mean=np.arange(8, 13),
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=0,  # two-tailed test
    ...     n_permutations=10000,
    ...     alpha=0.05
    ... )
    >>> # Get significant clusters
    >>> significant_clusters = [i for i, p in enumerate(cluster_stats.cluster_p_values) if p <= 0.05]
    >>> print(f"Found {len(significant_clusters)} significant clusters")
    """

    # Compute F-threshold for two-tailed test if needed
    dfn = len(data) - 1  # Numerator degrees of freedom
    dfd = np.sum([len(d) for d in data]) - len(data)  # Denominator degrees of freedom

    if tail == 0:
        threshold = f_dist.ppf(1 - alpha / 2, dfn, dfd)  # 2-tailed F-test
    else:
        threshold = None  # One-tailed test uses MNE's default
    
    # computing the cluster permutation t test
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(data,
                                                                     threshold=threshold,
                                                                     n_permutations=n_permutations,
                                                                     tail=tail, adjacency=ch_con_freq,
                                                                     t_power=1, out_type='mask')    
    # t_power = 1 weighs each location by its statistical score,
    # when set to 0 it gives a count of locations in each cluster

    # getting F values for sensors belonging to a significant cluster
    F_obs_plot = np.zeros(F_obs.shape)
    for cluster_p in cluster_p_values:
        if cluster_p <= alpha:
            sensors_plot = clusters[np.where(cluster_p_values == cluster_p)[
                0][0]].astype('uint8')
            F_values = sensors_plot*F_obs
            # taking maximum F value if a sensor belongs to many clusters
            F_obs_plot = np.maximum(F_obs_plot, F_values)

    statscondClusterTuple = namedtuple('statscondCluster', [
                                       'F_obs', 'clusters', 'cluster_p_values', 'H0', 'F_obs_plot'])

    return statscondClusterTuple(
        F_obs=F_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        F_obs_plot=F_obs_plot)


def statscluster(data: list, test: str, factor_levels: List[int], ch_con_freq: scipy.sparse.csr_matrix, 
                tail: int, n_permutations: int, alpha: float = 0.05) -> namedtuple:
    """
    Perform cluster-based statistical tests with various test statistics.
    
    This function provides a flexible interface to cluster-based permutation
    tests, supporting different statistical tests including t-tests, one-way ANOVA,
    and multiple-way ANOVA for complex experimental designs.
    
    Parameters
    ----------
    data : list or np.ndarray
        For t-tests and one-way ANOVA: list of arrays containing values from
        different groups or conditions to compare
        For multiple-way ANOVA: numpy array organized by factors
        
    test : str
        Type of statistical test to use:
        - 'ind ttest': Independent samples t-test
        - 'rel ttest': Related (paired) samples t-test
        - 'f oneway': One-way ANOVA
        - 'f multipleway': Multiple-way ANOVA
        
    factor_levels : List[int] or None
        For multiple-way ANOVA, list specifying the number of levels for each factor
        (e.g., [2, 3] for a 2×3 design with 2 levels of factor 1 and 3 levels of factor 2)
        Set to None for other tests
        
    ch_con_freq : scipy.sparse.csr_matrix
        Connectivity or metaconnectivity matrix defining adjacency across
        space and frequencies
        
    tail : int
        Direction of the test:
        - 0: two-tailed test (must be used for f oneway)
        - 1: one-tailed test (greater) (must be used for f multipleway)
        - -1: one-tailed test (less)
        
    n_permutations : int
        Number of permutations for the statistical test, e.g., 50000
        
    alpha : float, optional
        Significance threshold for clusters (default=0.05)
    
    Returns
    -------
    statscondClusterTuple : namedtuple
        A named tuple containing:
        - Stat_obs: Observed statistic (T or F) for all variables
        - clusters: Boolean array indicating locations in significant clusters
        - cluster_p_values: p-value for each identified cluster
        - H0: Max cluster-level statistics under permutation
        - Stat_obs_plot: Statistical values for significant sensors
    
    Notes
    -----
    For multiple-way ANOVA with connectivity values, the last dimensions may
    need to be flattened from shape (n_sensors, n_sensors) to a vector using np.reshape.
    
    The function applies different thresholding approaches based on the test type:
    - For t-tests: Uses the alpha value directly
    - For one-way ANOVA: Uses the alpha value directly
    - For multiple-way ANOVA: Calculates appropriate F-thresholds based on factor levels
    
    With t_power=1, each location is weighted by its statistical score within a cluster.
    
    Examples
    --------
    >>> # Independent t-test between two groups
    >>> ind_ttest_results = statscluster(
    ...     [group1_data, group2_data],
    ...     test='ind ttest',
    ...     factor_levels=None,
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=0,  # two-tailed test
    ...     n_permutations=10000,
    ...     alpha=0.05
    ... )
    >>> 
    >>> # 2×2 repeated measures ANOVA (within-subjects design)
    >>> # Factor 1: Condition (2 levels), Factor 2: Time (2 levels)
    >>> anova_results = statscluster(
    ...     data_array_2x2,  # Shape: (4, n_subjects, n_features)
    ...     test='f multipleway',
    ...     factor_levels=[2, 2],
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=1,  # F-tests are one-tailed
    ...     n_permutations=10000,
    ...     alpha=0.05
    ... )
    """

    # type of test
    if test == 'ind ttest':
        def stat_fun(*arg):
            return(scipy.stats.ttest_ind(arg[0], arg[1], equal_var=False)[0])
        threshold = alpha
    elif test == 'rel ttest':
        def stat_fun(*arg):
            return(scipy.stats.ttest_rel(arg[0], arg[1])[0])
        threshold = alpha
    elif test == 'f oneway':
        def stat_fun(*arg):
            return(scipy.stats.f_oneway(arg[0], arg[1])[0])
        threshold = alpha
    elif test == 'f multipleway':
        if max(factor_levels) > 2:
            correction = True
        else:
            correction = False
        def stat_fun(*arg):
            return(mne.stats.f_mway_rm(np.swapaxes(args, 1, 0),
                                       factor_levels,
                                       effects='all',
                                       correction=correction,
                                       return_pvals=False)[0])
        threshold = mne.stats.f_threshold_mway_rm(n_subjects=data.shape[1],
                                                  factor_levels=factor_levels,
                                                  effects='all',
                                                  pvalue=0.05)

    # computing the cluster permutation t test
    Stat_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(data,
                                                                        stat_fun=stat_fun,
                                                                        threshold=threshold,
                                                                        tail=tail,
                                                                        n_permutations=n_permutations,
                                                                        adjacency=ch_con_freq,
                                                                        t_power=1,
                                                                        out_type='mask')
    # getting F values for sensors belonging to a significant cluster
    Stat_obs_plot = np.zeros(Stat_obs.shape)
    for cluster_p in cluster_p_values:
        if cluster_p <= alpha:
            sensors_plot = clusters[np.where(cluster_p_values == cluster_p)[
                0][0]].astype('uint8')
            Stat_values = sensors_plot*Stat_obs
            # taking maximum statistical value if a sensor is in many clusters
            Stat_obs_plot = np.maximum(Stat_obs_plot, Stat_values)

    statscondClusterTuple = namedtuple('statscondCluster', [
                                       'Stat_obs', 'clusters', 'cluster_p_values', 'H0', 'Stat_obs_plot'])

    return statscondClusterTuple(
        Stat_obs=Stat_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        Stat_obs_plot=Stat_obs_plot)
