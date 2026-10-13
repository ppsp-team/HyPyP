"""Cluster-level permutation tests on planted synthetic effects.

Twelve features on a chain adjacency, groups of standard normal observations,
an effect planted on a few contiguous features of the last group. Every
expected value comes from scipy or from the construction of the data, not
from HyPyP. The statistic reported by ``statscluster`` is scipy's with the
groups in the order given, so a positive lift on the second group gives a
negative t.
"""

import numpy as np
import pytest
import scipy.sparse
import scipy.stats
from numpy.testing import assert_allclose, assert_array_equal

from hypyp import stats

N_FEATURES = 12
ADJACENCY = scipy.sparse.csr_matrix(
    np.eye(N_FEATURES) + np.eye(N_FEATURES, k=1) + np.eye(N_FEATURES, k=-1)
)
N_PERMUTATIONS = 500
ALPHA = 0.05


def groups(n_obs, lift, features, seed, n_groups=2):
    """Standard normal groups; the last one is lifted on the given features."""
    rng = np.random.default_rng(seed)
    data = [rng.standard_normal((n_obs, N_FEATURES)) for _ in range(n_groups)]
    data[-1][:, features] += lift
    return data


def significant_masks(result):
    return [
        mask for mask, p in zip(result.clusters, result.cluster_p_values) if p <= ALPHA
    ]


def test_f_oneway_statistic_uses_every_group():
    data = groups(15, 2.0, [3, 4, 5], seed=0, n_groups=3)
    result = stats.statscluster(
        data, "f oneway", None, ADJACENCY, tail=1, n_permutations=N_PERMUTATIONS, seed=0
    )
    assert_allclose(result.Stat_obs, scipy.stats.f_oneway(*data)[0])
    masks = significant_masks(result)
    assert len(masks) == 1
    assert {3, 4, 5} <= set(np.flatnonzero(masks[0]))


def test_ind_ttest_statistic_matches_scipy_welch():
    data = groups(20, 3.0, [3, 4, 5], seed=1)
    result = stats.statscluster(
        data,
        "ind ttest",
        None,
        ADJACENCY,
        tail=0,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )
    expected = scipy.stats.ttest_ind(data[0], data[1], equal_var=False)[0]
    assert_allclose(result.Stat_obs, expected)


def test_negative_cluster_is_kept_in_plot_array():
    data = groups(20, 3.0, [3, 4, 5], seed=1)
    result = stats.statscluster(
        data,
        "ind ttest",
        None,
        ADJACENCY,
        tail=0,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )
    assert np.all(result.Stat_obs[3:6] < 0)
    assert_array_equal(result.Stat_obs_plot[3:6], result.Stat_obs[3:6])
    assert np.count_nonzero(result.Stat_obs_plot) == 3


def test_clusters_with_equal_p_values_are_all_kept_in_plot_array():
    data = groups(20, 6.0, [1, 2, 8, 9], seed=2)
    result = stats.statscondCluster(
        data, [10], ADJACENCY, tail=1, n_permutations=N_PERMUTATIONS, seed=0
    )
    assert len(significant_masks(result)) == 2
    assert len(set(result.cluster_p_values)) == 1
    assert_array_equal(np.flatnonzero(result.F_obs_plot), [1, 2, 8, 9])
    assert_array_equal(result.F_obs_plot[[1, 2, 8, 9]], result.F_obs[[1, 2, 8, 9]])


@pytest.mark.parametrize("test", ["ind ttest", "rel ttest"])
def test_tail_minus_one_finds_the_negative_cluster(test):
    data = groups(20, 3.0, [3, 4, 5], seed=3)
    result = stats.statscluster(
        data, test, None, ADJACENCY, tail=-1, n_permutations=N_PERMUTATIONS, seed=0
    )
    masks = significant_masks(result)
    assert len(masks) == 1
    assert_array_equal(np.flatnonzero(masks[0]), [3, 4, 5])
    assert np.all(result.Stat_obs_plot[3:6] < 0)


def test_f_tests_reject_tail_minus_one():
    data = groups(10, 0.0, [], seed=4)
    with pytest.raises(ValueError, match="tail"):
        stats.statscondCluster(data, [10], ADJACENCY, tail=-1, n_permutations=50)
    with pytest.raises(ValueError, match="tail"):
        stats.statscluster(
            data, "f oneway", None, ADJACENCY, tail=-1, n_permutations=50
        )


def test_alpha_sets_the_cluster_forming_threshold_for_tail_one():
    data = groups(20, 2.0, [3, 4, 5], seed=8)
    loose = stats.statscondCluster(
        data, [10], ADJACENCY, tail=1, n_permutations=N_PERMUTATIONS, alpha=0.05, seed=0
    )
    strict = stats.statscondCluster(
        data, [10], ADJACENCY, tail=1, n_permutations=N_PERMUTATIONS, alpha=1e-6, seed=0
    )
    extent = lambda result: sum(int(mask.sum()) for mask in result.clusters)  # noqa: E731
    assert extent(loose) >= 3
    assert extent(strict) < extent(loose)


def test_seed_makes_the_permutation_reproducible():
    data = groups(20, 2.0, [3, 4, 5], seed=9)
    first = stats.statscondCluster(data, [10], ADJACENCY, 1, N_PERMUTATIONS, seed=0)
    second = stats.statscondCluster(data, [10], ADJACENCY, 1, N_PERMUTATIONS, seed=0)
    other = stats.statscondCluster(data, [10], ADJACENCY, 1, N_PERMUTATIONS, seed=1)
    assert_array_equal(first.H0, second.H0)
    assert_array_equal(first.cluster_p_values, second.cluster_p_values)
    assert not np.array_equal(first.H0, other.H0)
