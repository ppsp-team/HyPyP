import warnings, numpy as np
from hypyp import analyses
warnings.simplefilter("ignore")
rng = np.random.default_rng(11)
cs = rng.standard_normal((2, 1, 1, 1, 600)) + 0j
kw = dict(mvar_params={"mvar_order": 2, "fitting_method": "default", "delta": 0},
          ica_params={"method": "infomax_extended", "random_state": 0},
          measure_params={"name": "mvar_tf", "n_fft": 8})
a = analyses.compute_conn_mvar(cs, check_stability=False, **kw)
b = analyses.compute_conn_mvar(cs, check_stability=True, **kw)
print("check_stability=False dtype", a.dtype, "max |imag|", np.abs(a.imag).max())
print("check_stability=True  dtype", b.dtype, "max |imag|", np.abs(b.imag).max())
print("max |a-b|:", np.abs(a-b).max())
