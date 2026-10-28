"""Compare two outputs.npz files key by key and print a markdown table.

Usage: python compare.py master.npz branch.npz [label]
"""

import sys

import numpy as np

reference = np.load(sys.argv[1], allow_pickle=False)
candidate = np.load(sys.argv[2], allow_pickle=False)
label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]

keys = sorted(set(reference.files) | set(candidate.files))
identical, changed, rows = 0, [], []
for key in keys:
    if key not in reference.files:
        rows.append((key, "new on branch"))
        changed.append(key)
        continue
    if key not in candidate.files:
        rows.append((key, "missing on branch"))
        changed.append(key)
        continue
    a, b = reference[key], candidate[key]
    if a.shape != b.shape:
        rows.append((key, f"shape {a.shape} -> {b.shape}"))
        changed.append(key)
    elif a.dtype.kind in "US" or b.dtype.kind in "US":
        same = np.array_equal(a, b)
        rows.append((key, "identical" if same else f"{a} -> {b}"))
        identical += same
        if not same:
            changed.append(key)
    elif np.array_equal(a, b, equal_nan=True):
        identical += 1
        rows.append((key, "identical"))
    else:
        diff = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
        rows.append((key, f"max |diff| {diff:.3g}"))
        changed.append(key)

print(f"## {label}: {identical} of {len(keys)} arrays bit-identical to master")
if changed:
    print()
    print("| array | result |")
    print("| --- | --- |")
    for key, result in rows:
        if result != "identical":
            print(f"| `{key}` | {result} |")
