import warnings, numpy as np, tracemalloc
from hypyp import analyses
warnings.simplefilter("ignore")
rng = np.random.default_rng(12)
cs = rng.standard_normal((2, 4, 32, 1, 1000)) + 1j*rng.standard_normal((2, 4, 32, 1, 1000))
print("input MB:", cs.nbytes/1e6)
for mode in ["plv", "pli", "wpli"]:
    tracemalloc.start(); analyses.compute_sync(cs, mode); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"{mode:5s} peak MB {peak/1e6:8.1f}   (n_ep*n_f*(2n_ch)^2*n_t*16B = {4*1*64*64*1000*16/1e6:.0f} MB)")
