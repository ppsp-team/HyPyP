"""Adjacency construction checks on a tiny 4-sensor montage."""

import time

import mne
import numpy as np
import scipy.sparse
from mne.stats import combine_adjacency, permutation_cluster_test

from hypyp import stats

np.set_printoptions(linewidth=200)

# 4 sensors on a line: A(0,0) B(1,0) C(2,0) D(3,0) -> Delaunay in 2D degenerate; use a square+offset
names = ["A", "B", "C", "D"]
pos = {"A": [0, 0, 0], "B": [1, 0, 0], "C": [0, 1, 0], "D": [2.5, 2.5, 0]}
info = mne.create_info(names, 100.0, "eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head"))
epochs = mne.EpochsArray(np.zeros((2, 4, 10)), info, verbose=False)

print("=== con_matrix on 4 sensors, 2 freqs")
cm = stats.con_matrix(epochs, [10, 11])
print("ch_con type:", type(cm.ch_con), "\n", cm.ch_con.toarray().astype(int))
print("ch_con_freq type:", type(cm.ch_con_freq), "shape", cm.ch_con_freq.shape)
print(cm.ch_con_freq.astype(int))
ref = combine_adjacency(2, cm.ch_con).toarray()
print("equals combine_adjacency(n_freq, ch_con) [freq-major index f*n_ch+ch]:",
      np.array_equal(cm.ch_con_freq, ref))
print("equals combine_adjacency(ch_con, n_freq) [ch-major index ch*n_freq+f]:",
      np.array_equal(cm.ch_con_freq, combine_adjacency(cm.ch_con, 2).toarray()))

print("\n=== dense ch_con_freq straight into permutation_cluster_test")
X = [np.random.default_rng(0).standard_normal((5, 8)) for _ in range(2)]
try:
    permutation_cluster_test(X, adjacency=cm.ch_con_freq, n_permutations=10, verbose=False)
except Exception as e:
    print("ERROR:", type(e).__name__, e)

print("\n=== feature-order mismatch: 3-D data (n_obs, n_ch, n_freq) vs freq-major ch_con_freq")
# ch-major flatten used by MNE: feature = ch*n_freq + f ; ch_con_freq is f*n_ch + ch
n_ch, n_freq = 4, 2
adj_hypyp = cm.ch_con_freq
adj_right = combine_adjacency(cm.ch_con, n_freq).toarray()  # matches C-order flatten of (n_ch, n_freq)
print("hypyp adjacency == correct adjacency for (n_obs,n_ch,n_freq) data:",
      np.array_equal(adj_hypyp, adj_right))
# concrete consequence: planted effect on channel A (idx0) at both freqs.
# In ch-major flatten these are features 0,1 which should be adjacent (same channel, neighbouring freq).
# In hypyp's freq-major adjacency features 0,1 are channels A and B at freq 0 (adjacent only if A~B).
# Pick channels C(2) and D(3): C and D are not neighbours.
print("A~B spatially:", bool(cm.ch_con[0, 1]), "; C~D spatially:", bool(cm.ch_con[2, 3]))
rng = np.random.default_rng(1)
g1 = rng.standard_normal((15, n_ch, n_freq))
g2 = rng.standard_normal((15, n_ch, n_freq))
g2[:, 3, :] += 3.0  # planted effect on D at both freqs
for label, adj in [("hypyp freq-major", adj_hypyp), ("correct ch-major", adj_right)]:
    F, cl, pv, H0 = permutation_cluster_test(
        [g1, g2], adjacency=scipy.sparse.csr_matrix(adj), n_permutations=500,
        threshold=4.2, tail=1, seed=0, verbose=False, out_type="mask")
    print(f"{label}: n clusters={len(cl)}, sizes={[int(c.sum()) for c in cl]}, p={np.round(pv, 3)}")

print("\n=== metaconn_matrix on 4 sensors")
ch_con = cm.ch_con
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
mc = stats.metaconn_matrix(pairs, ch_con, [10], plot=False)
print("pairs:", pairs)
print(mc.metaconn.astype(int))


def ref_pair_adj(pairs, A, two_brains=False, n=None):
    """Reference: pairs (a,b),(c,d) neighbours iff a~c and b~d (A has the diagonal), or swapped
    for intra-brain."""
    A = A.toarray().astype(bool)
    out = np.zeros((len(pairs), len(pairs)), int)
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            if two_brains:
                out[i, j] = A[a, c] and A[b - n, d - n]
            else:
                out[i, j] = (A[a, c] and A[b, d]) or (A[a, d] and A[b, c])
    return out


print("matches reference (a~c & b~d) | (a~d & b~c):",
      np.array_equal(mc.metaconn.astype(int), ref_pair_adj(pairs, ch_con)))

print("\n=== metaconn_matrix_2brains on 4+4 sensors")
pairs2 = [(i, j) for i in range(4) for j in range(4, 8)]
mc2 = stats.metaconn_matrix_2brains(pairs2, ch_con, [10], plot=False)
ref2 = ref_pair_adj(pairs2, ch_con, two_brains=True, n=4)
print("matches reference (a~c & b~d within own brain):", np.array_equal(mc2.metaconn.astype(int), ref2))
# any cross-brain adjacency? pair (A,A') vs (B',?) impossible by construction since first index always P1
# check: pairs sharing nothing but with P1 chan adjacent to P2 chan index-wise
idx = {p: k for k, p in enumerate(pairs2)}
print("(A,D') ~ (D,A') ?", int(mc2.metaconn[idx[(0, 7)], idx[(3, 4)]]), "(expected 0)")
print("(A,A') ~ (B,A') ?", int(mc2.metaconn[idx[(0, 4)], idx[(1, 4)]]), "(expected 1 since A~B)")
print("(A,A') ~ (A,C') ?", int(mc2.metaconn[idx[(0, 4)], idx[(0, 6)]]), "(expected 1 since A~C)")
print("(A,A') ~ (A,D') ?", int(mc2.metaconn[idx[(0, 4)], idx[(0, 7)]]), "(expected 0 since A!~D)")

print("\n=== n inference in metaconn_matrix_2brains with a subset of P1 channels")
sub = [(i, j) for i in range(3) for j in range(4, 8)]  # P1 channel D omitted -> n inferred as 3
try:
    mc3 = stats.metaconn_matrix_2brains(sub, ch_con, [10], plot=False)
    ref3 = ref_pair_adj(sub, ch_con, two_brains=True, n=4)
    print("matches reference with true n=4:", np.array_equal(mc3.metaconn.astype(int), ref3))
except Exception as e:
    print("ERROR:", type(e).__name__, e)

print("\n=== timing of metaconn_matrix_2brains (pure-python O(P^2 F^2))")
for nch in (8, 16, 24):
    A = scipy.sparse.csr_matrix(np.eye(nch) + np.eye(nch, k=1) + np.eye(nch, k=-1))
    pp = [(i, j) for i in range(nch) for j in range(nch, 2 * nch)]
    t0 = time.perf_counter()
    stats.metaconn_matrix_2brains(pp, A, [10, 11, 12], plot=False)
    print(f"n_ch={nch}: {len(pp)} pairs x 3 freqs -> {time.perf_counter() - t0:.1f}s")
