"""Peak memory of the numpy PLI / wPLI path versus the input size."""
import tracemalloc
import warnings

import numpy as np

from hypyp.sync import get_metric

warnings.simplefilter("ignore")
AX = (0, 1, 3, 2)
rng = np.random.default_rng(0)

for E, F, C, T in [(2, 1, 32, 500), (2, 1, 64, 500), (4, 1, 64, 500)]:
    sig = rng.standard_normal((E, F, C, T)) + 1j * rng.standard_normal((E, F, C, T))
    for mode in ["pli", "wpli", "plv"]:
        tracemalloc.start()
        get_metric(mode).compute(sig, T, AX)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        five_d = E * F * C * C * T * 8
        print(
            f"E={E} F={F} C={C} T={T} input={sig.nbytes/1e6:7.1f} MB  {mode:5s} "
            f"peak={peak/1e6:8.1f} MB  = {peak/five_d:4.1f} x one (E,F,C,C,T) float64"
        )

E, F, C, T = 30, 2, 128, 1000  # 64 ch/subject, 30 epochs, 2 bands, 4 s @ 250 Hz
print(
    f"\nextrapolation E={E} F={F} C={C} T={T}: one (E,F,C,C,T) float64 = "
    f"{E*F*C*C*T*8/1e9:.1f} GB; numpy pli peak ~ 6x that"
)
