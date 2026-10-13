"""Adjacency builders and the feature order they imply.

Four sensors in a known layout: A and B are neighbours, A and C are
neighbours, A and D are not. Every expected value follows from the
stated rules, which are written out independently of HyPyP: spatial
neighbours at the same frequency, the same channel at neighbouring
frequencies, pairs of channels whose members are neighbours.
"""

import mne
import numpy as np
import pytest
import scipy.sparse
from numpy.testing import assert_array_equal

from hypyp import stats

NAMES = ["A", "B", "C", "D"]
POSITIONS = {"A": [0, 0, 0], "B": [1, 0, 0], "C": [0, 1, 0], "D": [2.5, 2.5, 0]}
N_CHANNELS = len(NAMES)


@pytest.fixture
def epochs():
    info = mne.create_info(NAMES, 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(ch_pos=POSITIONS, coord_frame="head")
    )
    return mne.EpochsArray(np.zeros((2, N_CHANNELS, 10)), info, verbose=False)


def neighbours(ch_con):
    adjacency = ch_con.toarray().astype(bool)
    np.fill_diagonal(adjacency, False)
    return adjacency


def expected_across_frequencies(adjacency, n_freq):
    """Linked: same channel at neighbouring frequencies, or spatial neighbours at
    the same frequency. Feature index is channel * n_freq + frequency."""
    n = adjacency.shape[0]
    out = np.zeros((n * n_freq, n * n_freq), dtype=bool)
    for a in range(n):
        for f in range(n_freq):
            for b in range(n):
                for g in range(n_freq):
                    same_channel_next_freq = a == b and abs(f - g) == 1
                    neighbour_same_freq = adjacency[a, b] and f == g
                    out[a * n_freq + f, b * n_freq + g] = (
                        same_channel_next_freq or neighbour_same_freq
                    )
    return out


def off_diagonal(matrix):
    dense = np.asarray(matrix.toarray() if scipy.sparse.issparse(matrix) else matrix)
    dense = dense.astype(bool).copy()
    np.fill_diagonal(dense, False)
    return dense


def test_the_montage_is_what_the_rules_assume(epochs):
    adjacency = neighbours(stats.con_matrix(epochs, [10]).ch_con)
    assert adjacency[0, 1] and adjacency[0, 2]
    assert not adjacency[0, 3]


def test_con_matrix_links_channel_major_features(epochs):
    result = stats.con_matrix(epochs, [10, 11, 12])
    assert scipy.sparse.issparse(result.ch_con_freq)
    expected = expected_across_frequencies(neighbours(result.ch_con), 3)
    assert_array_equal(off_diagonal(result.ch_con_freq), expected)


def test_con_matrix_with_one_frequency_is_the_channel_adjacency(epochs):
    result = stats.con_matrix(epochs, [10])
    assert_array_equal(off_diagonal(result.ch_con_freq), neighbours(result.ch_con))


def test_clusters_follow_the_data_layout(epochs):
    """An effect on channel A at frequencies 0 and 2 (not 1) gives two separate
    clusters once the adjacency is in the order MNE flattens the data."""
    n_freq = 3
    result = stats.con_matrix(epochs, [10, 11, 12])
    rng = np.random.default_rng(3)
    groups = [rng.standard_normal((20, N_CHANNELS, n_freq)) for _ in range(2)]
    groups[1][:, 0, 0] += 4.0
    groups[1][:, 0, 2] += 4.0
    test = stats.statscondCluster(
        groups, [10, 11, 12], result.ch_con_freq, tail=1, n_permutations=300, seed=0
    )
    significant = [
        set(map(tuple, np.argwhere(mask)))
        for mask, p in zip(test.clusters, test.cluster_p_values)
        if p <= 0.05
    ]
    assert sorted(significant, key=sorted) == [{(0, 0)}, {(0, 2)}]


def expected_pair_adjacency(pairs, adjacency, n_first_brain=None):
    """Two pairs are neighbours when their members are neighbours or identical,
    member by member; within one brain the members may also be swapped."""
    A = adjacency | np.eye(adjacency.shape[0], dtype=bool)
    out = np.zeros((len(pairs), len(pairs)), dtype=bool)
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            if n_first_brain is None:
                out[i, j] = (A[a, c] and A[b, d]) or (A[a, d] and A[b, c])
            else:
                shift = n_first_brain
                out[i, j] = A[a, c] and A[b - shift, d - shift]
    return out


def test_metaconn_matrix_matches_the_pair_rule(epochs):
    ch_con = stats.con_matrix(epochs, [10]).ch_con
    pairs = [(i, j) for i in range(N_CHANNELS) for j in range(i + 1, N_CHANNELS)]
    result = stats.metaconn_matrix(pairs, ch_con, [10], plot=False)
    assert_array_equal(
        result.metaconn.astype(bool), expected_pair_adjacency(pairs, neighbours(ch_con))
    )


def test_metaconn_matrix_2brains_matches_the_pair_rule(epochs):
    ch_con = stats.con_matrix(epochs, [10]).ch_con
    pairs = [
        (i, j) for i in range(N_CHANNELS) for j in range(N_CHANNELS, 2 * N_CHANNELS)
    ]
    result = stats.metaconn_matrix_2brains(pairs, ch_con, [10], plot=False)
    expected = expected_pair_adjacency(
        pairs, neighbours(ch_con), n_first_brain=N_CHANNELS
    )
    assert_array_equal(result.metaconn.astype(bool), expected)
    index = {pair: k for k, pair in enumerate(pairs)}
    assert result.metaconn[index[(0, 7)], index[(3, 4)]] == 0  # no link across brains
    assert result.metaconn[index[(0, 4)], index[(1, 4)]] == 1  # A~B, same partner


def test_metaconn_matrix_2brains_takes_the_channel_count_from_the_adjacency(epochs):
    ch_con = stats.con_matrix(epochs, [10]).ch_con
    pairs = [
        (i, j) for i in range(3) for j in range(N_CHANNELS, 2 * N_CHANNELS)
    ]  # no D
    result = stats.metaconn_matrix_2brains(pairs, ch_con, [10], plot=False)
    expected = expected_pair_adjacency(
        pairs, neighbours(ch_con), n_first_brain=N_CHANNELS
    )
    assert_array_equal(result.metaconn.astype(bool), expected)


def test_metaconn_matrix_2brains_accepts_second_brain_indices_from_zero(epochs):
    ch_con = stats.con_matrix(epochs, [10]).ch_con
    shifted = [
        (i, j) for i in range(N_CHANNELS) for j in range(N_CHANNELS, 2 * N_CHANNELS)
    ]
    from_zero = [(i, j - N_CHANNELS) for i, j in shifted]
    assert_array_equal(
        stats.metaconn_matrix_2brains(from_zero, ch_con, [10], plot=False).metaconn,
        stats.metaconn_matrix_2brains(shifted, ch_con, [10], plot=False).metaconn,
    )


def test_metaconn_freq_is_pair_major(epochs):
    ch_con = stats.con_matrix(epochs, [10]).ch_con
    pairs = [(i, j) for i in range(N_CHANNELS) for j in range(i + 1, N_CHANNELS)]
    result = stats.metaconn_matrix(pairs, ch_con, [10, 11], plot=False)
    assert scipy.sparse.issparse(result.metaconn_freq)
    expected = expected_across_frequencies(off_diagonal(result.metaconn), 2)
    assert_array_equal(off_diagonal(result.metaconn_freq), expected)


def test_cluster_functions_reject_an_adjacency_of_the_wrong_size(epochs):
    ch_con = stats.con_matrix(epochs, [10, 11, 12]).ch_con
    groups = [np.zeros((5, N_CHANNELS, 3)) for _ in range(2)]
    with pytest.raises(ValueError, match="features"):
        stats.statscondCluster(groups, [10, 11, 12], ch_con, 1, 50)
    with pytest.raises(ValueError, match="features"):
        stats.statscluster(groups, "ind ttest", None, ch_con, 0, 50)


def test_cluster_functions_accept_a_dense_adjacency_with_a_warning(epochs):
    ch_con_freq = stats.con_matrix(epochs, [10, 11]).ch_con_freq.toarray()
    rng = np.random.default_rng(0)
    groups = [rng.standard_normal((8, N_CHANNELS, 2)) for _ in range(2)]
    with pytest.warns(UserWarning, match="sparse"):
        stats.statscondCluster(groups, [10, 11], ch_con_freq, 1, 50, seed=0)
