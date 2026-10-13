"""The paired cluster test is the sign-flip test on the within-pair differences.

Twelve features on a chain, twenty pairs of observations that share a large
per-pair offset across every feature (the EEG situation: every subject has a
topography), a small lift on three contiguous features. The reference is
MNE's permutation_cluster_1samp_test on the differences with the same seed
and threshold, which is the exchangeability a paired design has.
"""

import numpy as np
import pytest
import scipy.sparse
import scipy.stats
from mne.stats import permutation_cluster_1samp_test
from numpy.testing import assert_allclose, assert_array_equal

from hypyp import stats

N_FEATURES = 12
ADJACENCY = scipy.sparse.csr_matrix(
    np.eye(N_FEATURES) + np.eye(N_FEATURES, k=1) + np.eye(N_FEATURES, k=-1)
)
N_PERMUTATIONS = 500


def paired_groups(n_pairs, lift, seed):
    rng = np.random.default_rng(seed)
    offset = rng.standard_normal((n_pairs, 1)) * 3.0
    first = offset + rng.standard_normal((n_pairs, N_FEATURES)) * 0.5
    second = offset + rng.standard_normal((n_pairs, N_FEATURES)) * 0.5
    first[:, 3:6] += lift
    return [first, second]


@pytest.mark.parametrize("seed", range(3))
def test_rel_ttest_equals_the_sign_flip_test_on_the_differences(seed):
    data = paired_groups(20, lift=0.5, seed=seed)
    result = stats.statscluster(
        data,
        "rel ttest",
        None,
        ADJACENCY,
        tail=0,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )
    threshold = scipy.stats.t.ppf(1 - 0.025, len(data[0]) - 1)
    t_obs, clusters, p_values, h0 = permutation_cluster_1samp_test(
        data[0] - data[1],
        threshold=threshold,
        tail=0,
        adjacency=ADJACENCY,
        n_permutations=N_PERMUTATIONS,
        t_power=1,
        out_type="mask",
        seed=0,
        verbose=False,
    )
    assert_allclose(result.Stat_obs, t_obs)
    assert_allclose(result.Stat_obs, scipy.stats.ttest_rel(data[0], data[1])[0])
    assert_array_equal(result.cluster_p_values, p_values)
    assert_array_equal(result.H0, h0)
    assert all(np.array_equal(a, b) for a, b in zip(result.clusters, clusters))


def test_rel_ttest_finds_a_small_lift_under_a_shared_topography():
    data = paired_groups(40, lift=0.5, seed=7)
    result = stats.statscluster(
        data,
        "rel ttest",
        None,
        ADJACENCY,
        tail=0,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )
    significant = [
        mask for mask, p in zip(result.clusters, result.cluster_p_values) if p <= 0.05
    ]
    assert len(significant) == 1
    assert {3, 4, 5} <= set(np.flatnonzero(significant[0]))


def test_rel_ttest_needs_the_same_number_of_observations_in_both_groups():
    data = paired_groups(20, lift=0.0, seed=0)
    with pytest.raises(ValueError, match="pair"):
        stats.statscluster([data[0], data[1][:-1]], "rel ttest", None, ADJACENCY, 0, 50)


def test_f_multipleway_is_rejected_with_a_pointer():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 12, N_FEATURES))
    with pytest.raises(ValueError, match="removed"):
        stats.statscluster(X, "f multipleway", [2, 2], ADJACENCY, 1, 50)
