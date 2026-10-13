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
import warnings
import numpy as np
import scipy
from scipy.stats import f as f_dist
import matplotlib.pylab as plt
import mne
from mne.channels import find_ch_adjacency
from mne.stats import (
    combine_adjacency,
    permutation_cluster_1samp_test,
    permutation_cluster_test,
)


def statsCond(
    data: np.ndarray,
    epochs: mne.Epochs,
    n_permutations: int | str,
    alpha: float,
    seed: int | None = None,
) -> namedtuple:
    """
    One-sample permutation t-test of a contrast against zero, per channel.

    For every channel, the test asks whether the mean of `data` across
    participants differs from zero, by flipping the sign of each participant's
    value (MNE's permutation_t_test). The p-values are corrected for the number
    of channels by the max-T method, which controls the family-wise error rate.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_tests, n_freq): one contrast per
        participant (the difference between two conditions, or a log-ratio
        against a baseline), per channel and frequency. Values are averaged
        across frequencies before the test. Raw power, which is positive
        everywhere, would come out "significant" on every channel: pass a
        contrast.

    epochs : mne.Epochs
        Unused; kept so that existing calls keep working.

    n_permutations : int or "all"
        Number of sign flips drawn, e.g. 5000, or "all" for the exact test
        over the 2**n_samples flips (feasible up to about twenty participants).

    alpha : float
        Significance threshold on the corrected p-values, e.g. 0.05

    seed : int or None, optional
        Seed of the permutation, forwarded to MNE, so that a run can be
        reproduced (default=None)

    Returns
    -------
    statsCondTuple : namedtuple
        A named tuple containing:
        - T_obs: T-statistic observed for all variables, array of shape (n_tests)
        - p_values: max-T corrected p-values, array of shape (n_tests)
        - H0: maximum absolute T under permutation, array of shape (n_permutations)
        - adj_p: tuple (p_values <= alpha, p_values), kept for compatibility;
          the p-values are already corrected and no second correction is applied
        - T_obs_plot: T_obs on the significant channels, zero elsewhere

    Notes
    -----
    This test calculates if the observed mean significantly deviates from 0;
    it doesn't compare two periods, but tests one period against the null hypothesis.

    Randomized data are generated with random sign flips, and the test is two-tailed
    (the alternative hypothesis is that the data mean is different from 0).

    The frequency dimension is reduced to one for the test (average in the frequency
    band-of-interest). To take frequencies into account, use cluster statistics
    (see statscondCluster function).

    Examples
    --------
    >>> # Alpha power in a condition against a baseline, one contrast per participant
    >>> contrast = np.log(power_condition / power_baseline)  # (n_participants, n_channels, n_freq)
    >>> result = statsCond(contrast, epochs, n_permutations=5000, alpha=0.05, seed=0)
    >>> significant_channels = np.array(epochs.ch_names)[result.adj_p[0]]
    """

    # checking whether data have the same size
    assert len(data.shape) == 3, "PSD does not have the appropriate shape!"

    # averaging across frequencies (compute stats only in ch space)
    power = np.mean(data, axis=2)
    if np.all(power > 0) or np.all(power < 0):
        warnings.warn(
            "every value has the same sign: statsCond tests the mean against zero, "
            "so pass a contrast (a difference between conditions or a log-ratio), "
            "not raw power"
        )
    T_obs, p_values, H0 = mne.stats.permutation_t_test(
        power, n_permutations, tail=0, n_jobs=1, seed=seed
    )
    # the p-values are already corrected for the number of channels (max-T)
    significant = p_values <= alpha
    adj_p = (significant, p_values)

    T_obs_plot = np.where(significant, T_obs, 0)

    statsCondTuple = namedtuple(
        "statsCond", ["T_obs", "p_values", "H0", "adj_p", "T_obs_plot"]
    )

    return statsCondTuple(
        T_obs=T_obs, p_values=p_values, H0=H0, adj_p=adj_p, T_obs_plot=T_obs_plot
    )


def _across_frequencies(adjacency, n_freq: int) -> scipy.sparse.csr_matrix:
    """Adjacency of (feature, frequency) in feature-major order, row index
    `feature * n_freq + frequency`, which is how MNE flattens data of shape
    (n_observations, n_features, n_frequencies): neighbouring features at the
    same frequency, and the same feature at neighbouring frequencies."""
    # a copy, because combine_adjacency clears the diagonal of its input in place
    base = scipy.sparse.csr_matrix(adjacency, copy=True)
    return scipy.sparse.csr_matrix(combine_adjacency(base, n_freq))


def _check_adjacency(adjacency, data) -> scipy.sparse.csr_matrix | None:
    """The adjacency as a sparse matrix with one row per feature of the data."""
    if adjacency is None:
        return None
    if not scipy.sparse.issparse(adjacency):
        warnings.warn(
            "the adjacency was given as a dense array; converting it to the sparse "
            "matrix that MNE's cluster functions require"
        )
        adjacency = scipy.sparse.csr_matrix(adjacency)
    n_features = int(np.prod(np.shape(data[0])[1:]))
    if adjacency.shape[0] != n_features:
        raise ValueError(
            f"the adjacency has {adjacency.shape[0]} rows for {n_features} features "
            "per observation. Data of shape (n_observations, n_channels, "
            "n_frequencies) take con_matrix(...).ch_con_freq, whose rows are ordered "
            "channel-major (channel * n_frequencies + frequency) like the flattened "
            "data; connectivity values take one vector per observation in the order "
            "of the `electrodes` list given to metaconn_matrix or "
            "metaconn_matrix_2brains, with its `metaconn_freq`"
        )
    return adjacency


def con_matrix(
    epochs: mne.Epochs, freqs_mean: List[float], draw: bool = False
) -> namedtuple:
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
    same channel are also considered connected. Its rows are ordered
    channel-major, row index `channel * len(freqs_mean) + frequency`, which is
    the order in which MNE flattens data of shape (n_observations, n_channels,
    n_frequencies), so it can be passed to statscondCluster or statscluster
    with such data as is.

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
    ch_con, ch_names_con = find_ch_adjacency(epochs.info, ch_type="eeg")

    # the same adjacency at every frequency, plus each channel's neighbouring
    # frequencies, in the order MNE flattens (n_channels, n_frequencies) data
    ch_con_freq = _across_frequencies(ch_con, len(freqs_mean))

    if draw:
        plt.figure()
        # visualizing the matrix and transforming it into array
        plt.subplot(1, 2, 1)
        plt.spy(ch_con)
        plt.title("Connectivity matrix")
        plt.subplot(1, 2, 2)
        plt.spy(ch_con_freq)
        plt.title("Meta-connectivity matrix")

    con_matrixTuple = namedtuple("con_matrix", ["ch_con", "ch_con_freq"])

    return con_matrixTuple(ch_con=ch_con, ch_con_freq=ch_con_freq)


def metaconn_matrix_2brains(
    electrodes: List[Tuple[int, int]],
    ch_con: scipy.sparse.csr_matrix,
    freqs_mean: List[float],
    plot: bool = False,
) -> namedtuple:
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
          space and frequencies, scipy.sparse.csr_matrix of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean)),
          row index `pair * len(freqs_mean) + frequency`

    Notes
    -----
    This function assumes there is no a priori connectivity between channels
    from different participants. It considers two channel pairs connected if:
    1. The respective channels within each participant are connected, or
    2. Some channels are identical across the pairs

    The channels of the second participant are numbered from
    `ch_con.shape[0]`, as indices_connectivity_interbrain numbers them.

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

    n = ch_con.shape[0]
    adjacency = scipy.sparse.csr_matrix(ch_con).toarray().astype(bool)
    first, second = np.asarray(electrodes).T
    # considering no a priori connectivity between the 2 brains: two pairs are
    # neighbours when, in each brain, their members are adjacent or identical
    # (second-brain indices below n wrap around, as some callers rely on)
    own = adjacency[first[:, None], first[None, :]]
    partner = adjacency[second[:, None] - n, second[None, :] - n]
    same_own = first[:, None] == first[None, :]
    same_partner = second[:, None] == second[None, :]
    metaconn = (
        (own & partner)
        | (own & same_partner)
        | (partner & same_own)
        | (same_own & same_partner)
    ).astype(float)

    metaconn_freq = _across_frequencies(metaconn, len(freqs_mean))

    if plot:
        # vizualising the array
        plt.figure()
        plt.spy(metaconn_freq)
        plt.title("Meta-connectivity matrix")

    metaconn_matrix_2brainsTuple = namedtuple(
        "metaconn_matrix_2brains", ["metaconn", "metaconn_freq"]
    )

    return metaconn_matrix_2brainsTuple(metaconn=metaconn, metaconn_freq=metaconn_freq)


def metaconn_matrix(
    electrodes: List[Tuple[int, int]],
    ch_con: scipy.sparse.csr_matrix,
    freqs_mean: List[float],
    plot: bool = True,
) -> namedtuple:
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

    plot : bool, optional
        Whether to draw the meta-connectivity matrix in the current figure
        (default=True, the historical behaviour)

    Returns
    -------
    metaconn_matrixTuple : namedtuple
        A named tuple containing:
        - metaconn: Connectivity matrix between channel pairs,
          array of shape (len(electrodes), len(electrodes))
        - metaconn_freq: Connectivity matrix between channel pairs across
          space and frequencies, scipy.sparse.csr_matrix of shape
          (len(electrodes)*len(freqs_mean), len(electrodes)*len(freqs_mean)),
          row index `pair * len(freqs_mean) + frequency`

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

    adjacency = scipy.sparse.csr_matrix(ch_con).toarray().astype(bool)
    first, second = np.asarray(electrodes).T

    def adjacent(x, y):
        return adjacency[x[:, None], y[None, :]]

    def same(x, y):
        return x[:, None] == y[None, :]

    # two pairs are neighbours when their members are adjacent or identical,
    # member by member or swapped
    metaconn = (
        (adjacent(first, first) & adjacent(second, second))
        | (adjacent(first, second) & adjacent(second, first))
        | (adjacent(first, first) & same(second, second))
        | (adjacent(first, second) & same(second, first))
        | (adjacent(second, first) & same(first, second))
        | (adjacent(second, second) & same(first, first))
    ).astype(float)

    metaconn_freq = _across_frequencies(metaconn, len(freqs_mean))

    if plot:
        # vizualising the array
        plt.spy(metaconn_freq)

    metaconn_matrixTuple = namedtuple("metaconn_matrix", ["metaconn", "metaconn_freq"])

    return metaconn_matrixTuple(metaconn=metaconn, metaconn_freq=metaconn_freq)


def _t_threshold(alpha: float, df: int, tail: int) -> float:
    """Cluster-forming t threshold at alpha; negative for a lower-tail test."""
    p = alpha / 2 if tail == 0 else alpha
    threshold = scipy.stats.t.ppf(1 - p, df)
    return -threshold if tail == -1 else threshold


def _significant_values(
    stat_obs: np.ndarray, clusters: list, cluster_p_values: np.ndarray, alpha: float
) -> np.ndarray:
    """The statistic on the features of the significant clusters, zero elsewhere.

    MNE's clusters are disjoint, so every feature takes the statistic of the
    one cluster it belongs to, with its sign.
    """
    values = np.zeros(stat_obs.shape)
    for mask, p in zip(clusters, cluster_p_values):
        if p <= alpha:
            values[mask] = stat_obs[mask]
    return values


def statscondCluster(
    data: list,
    freqs_mean: list,
    ch_con_freq: scipy.sparse.csr_matrix,
    tail: int,
    n_permutations: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> namedtuple:
    """
    Perform cluster-level statistical permutation test on neurophysiological data.

    This function applies a cluster-based permutation test to identify significant
    differences between conditions, correcting for multiple comparisons by taking
    into account connectivity across space and frequencies.

    Parameters
    ----------
    data : list
        List of arrays containing values from different conditions or groups
        to compare. Each array has shape (n_observations, n_channels,
        n_frequencies), which MNE flattens channel-major, the order of
        con_matrix(...).ch_con_freq; or (n_observations, n_pairs) with the
        connectivity values in the order of the `electrodes` list given to
        metaconn_matrix or metaconn_matrix_2brains, with its `metaconn_freq`

    freqs_mean : list
        Frequencies in the frequency-band-of-interest

    ch_con_freq : scipy.sparse.csr_matrix
        Connectivity or metaconnectivity matrix defining adjacency across
        space and frequencies, typically from con_matrix() or metaconn_matrix().
        It must have one row per feature of an observation; a dense array is
        converted with a warning

    tail : int
        Direction of the test. The F statistic is non-negative, so use 1
        (0 gives the same result); -1 is rejected.

    n_permutations : int
        Number of permutations for the statistical test, e.g., 50000

    alpha : float, optional
        Used twice: as the p-value of the F distribution that sets the
        cluster-forming threshold, and as the cluster-level significance
        that selects the clusters shown in F_obs_plot (default=0.05)

    seed : int or None, optional
        Seed of the permutation, forwarded to MNE, so that a run can be
        reproduced (default=None)

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

    The cluster-forming threshold is the F value at `alpha` for the degrees of
    freedom of the design, whatever the tail.

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

    if tail == -1:
        raise ValueError("The F statistic is non-negative: use tail=1, not -1.")
    ch_con_freq = _check_adjacency(ch_con_freq, data)

    # cluster-forming threshold from the F distribution at alpha, for every tail
    dfn = len(data) - 1  # Numerator degrees of freedom
    dfd = np.sum([len(d) for d in data]) - len(data)  # Denominator degrees of freedom
    threshold = f_dist.ppf(1 - alpha, dfn, dfd)

    # computing the cluster permutation t test
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        data,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=ch_con_freq,
        t_power=1,
        out_type="mask",
        seed=seed,
    )
    # t_power = 1 weighs each location by its statistical score,
    # when set to 0 it gives a count of locations in each cluster

    F_obs_plot = _significant_values(F_obs, clusters, cluster_p_values, alpha)

    statscondClusterTuple = namedtuple(
        "statscondCluster",
        ["F_obs", "clusters", "cluster_p_values", "H0", "F_obs_plot"],
    )

    return statscondClusterTuple(
        F_obs=F_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        F_obs_plot=F_obs_plot,
    )


def statscluster(
    data: list,
    test: str,
    factor_levels: List[int],
    ch_con_freq: scipy.sparse.csr_matrix,
    tail: int,
    n_permutations: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> namedtuple:
    """
    Perform cluster-based statistical tests with various test statistics.

    This function provides a flexible interface to cluster-based permutation
    tests, supporting different statistical tests including t-tests, one-way ANOVA,
    and multiple-way ANOVA for complex experimental designs.

    Parameters
    ----------
    data : list
        List of arrays containing values from different groups or conditions
        to compare, each of shape (n_observations, ...) with the layouts
        described in statscondCluster. For 'rel ttest': two arrays with the
        same number of observations, observation k of each being the same
        pair (the same participant, or the same dyad, in two conditions)

    test : str
        Type of statistical test to use:
        - 'ind ttest': Independent samples t-test (Welch)
        - 'rel ttest': Related (paired) samples t-test; the null flips the sign
          of each pair's difference, which is the exchangeability a paired
          design has (MNE's permutation_cluster_1samp_test on the differences)
        - 'f oneway': One-way ANOVA

    factor_levels : None
        Ignored. It served 'f multipleway', which was removed, and is kept so
        that existing calls keep working

    ch_con_freq : scipy.sparse.csr_matrix
        Connectivity or metaconnectivity matrix defining adjacency across
        space and frequencies, with one row per feature of an observation (see
        statscondCluster for the two layouts); a dense array is converted with
        a warning

    tail : int
        Direction of the test:
        - 0: two-tailed test
        - 1: one-tailed test (greater). The F statistic is non-negative, so
          use 1 for 'f oneway' (0 gives the same result)
        - -1: one-tailed test (less); t-tests only

    n_permutations : int
        Number of permutations for the statistical test, e.g., 50000

    alpha : float, optional
        Used twice: as the p-value of the t or F distribution that sets the
        cluster-forming threshold, and as the cluster-level significance that
        selects the clusters shown in Stat_obs_plot (default=0.05)

    seed : int or None, optional
        Seed of the permutation, forwarded to MNE, so that a run can be
        reproduced (default=None)

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
    Connectivity values go in as one vector per observation in the order of
    the `electrodes` list given to metaconn_matrix or metaconn_matrix_2brains,
    with its `metaconn_freq` as the adjacency; not as the full (n_sensors,
    n_sensors) matrix flattened, which counts every pair twice and includes
    the diagonal.

    The cluster-forming threshold is the t or F value at `alpha` for the
    degrees of freedom of the design.

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
    >>> # Paired t-test, the same dyads in two conditions
    >>> rel_ttest_results = statscluster(
    ...     [condition1_data, condition2_data],  # observation k: the same dyad
    ...     test='rel ttest',
    ...     factor_levels=None,
    ...     ch_con_freq=connectivity.ch_con_freq,
    ...     tail=0,
    ...     n_permutations=10000,
    ...     alpha=0.05,
    ...     seed=0,
    ... )
    """

    if test == "f oneway" and tail == -1:
        raise ValueError("The F statistic is non-negative: use tail=1, not -1.")
    if test == "f multipleway":
        raise ValueError(
            "statscluster(test='f multipleway') was removed: it permuted observations "
            "between subjects under a repeated-measures F, which is not a valid null, "
            "and could not run with two factors. See the changelog."
        )
    ch_con_freq = _check_adjacency(ch_con_freq, data)

    # type of test
    if test == "ind ttest":

        def stat_fun(*arg):
            return scipy.stats.ttest_ind(arg[0], arg[1], equal_var=False)[0]

        df = len(data[0]) + len(data[1]) - 2
        threshold = _t_threshold(alpha, df, tail)
    elif test == "rel ttest":
        if len(data) != 2 or len(data[0]) != len(data[1]):
            raise ValueError(
                "'rel ttest' needs two groups with the same number of observations, "
                "observation k of each being the same pair."
            )
        stat_fun = None  # the sign-flip test below computes the paired t itself
        threshold = _t_threshold(alpha, len(data[0]) - 1, tail)
    elif test == "f oneway":

        def stat_fun(*arg):
            return scipy.stats.f_oneway(*arg)[0]

        dfn = len(data) - 1
        dfd = sum(len(d) for d in data) - len(data)
        threshold = f_dist.ppf(1 - alpha, dfn, dfd)
    else:
        raise ValueError(
            f"Unknown test '{test}'. Use 'ind ttest', 'rel ttest' or 'f oneway'."
        )

    if test == "rel ttest":
        # the exchangeability of a paired design: under the null, each pair's
        # two labels could be swapped, which flips the sign of its difference
        Stat_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            np.asarray(data[0]) - np.asarray(data[1]),
            threshold=threshold,
            tail=tail,
            n_permutations=n_permutations,
            adjacency=ch_con_freq,
            t_power=1,
            out_type="mask",
            seed=seed,
        )
    else:
        # computing the cluster permutation test
        Stat_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            data,
            stat_fun=stat_fun,
            threshold=threshold,
            tail=tail,
            n_permutations=n_permutations,
            adjacency=ch_con_freq,
            t_power=1,
            out_type="mask",
            seed=seed,
        )
    Stat_obs_plot = _significant_values(Stat_obs, clusters, cluster_p_values, alpha)

    statscondClusterTuple = namedtuple(
        "statscondCluster",
        ["Stat_obs", "clusters", "cluster_p_values", "H0", "Stat_obs_plot"],
    )

    return statscondClusterTuple(
        Stat_obs=Stat_obs,
        clusters=clusters,
        cluster_p_values=cluster_p_values,
        H0=H0,
        Stat_obs_plot=Stat_obs_plot,
    )
