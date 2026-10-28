"""Function lengths, duplicated blocks and the nansum question."""
import ast
import hashlib
import warnings
from pathlib import Path

import numpy as np

ROOT = Path("/home/gdumas/Github/HyPyP")
files = sorted((ROOT / "hypyp/sync").rglob("*.py")) + [
    ROOT / "tests/test_sync.py",
    ROOT / "tests/accorr_reference.py",
]
print("== functions/methods over 100 lines (incl. docstring) ==")
total = 0
for f in files:
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = node.end_lineno - node.lineno + 1
            total += 1
            if n > 100:
                print(f"  {f.relative_to(ROOT)}:{node.lineno} {node.name} ({n} lines)")
print(f"({total} functions scanned)")

print("\n== identical 8-line windows shared across different files (hypyp/sync only) ==")
seen = {}
for f in files[:-2]:
    lines = [l.strip() for l in f.read_text().splitlines()]
    for i in range(len(lines) - 8):
        win = lines[i : i + 8]
        if sum(len(l) for l in win) < 120 or any(l.startswith('"""') for l in win):
            continue
        h = hashlib.md5("\n".join(win).encode()).hexdigest()
        seen.setdefault(h, []).append((str(f.relative_to(ROOT)), i + 1))
dups = {h: v for h, v in seen.items() if len({p for p, _ in v}) > 1}
groups = {}
for h, v in dups.items():
    key = tuple(sorted({p for p, _ in v}))
    groups.setdefault(key, []).append(v[0][1])
for key, starts in sorted(groups.items()):
    print(f"  {' <-> '.join(Path(k).name for k in key)}: {len(starts)} windows, e.g. line {min(starts)}")

print("\n== Coh numpy: does np.nansum in the denominator rescue a NaN sample? ==")
from hypyp.sync import get_metric

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
sig = rng.standard_normal((1, 1, 2, 100)) + 1j * rng.standard_normal((1, 1, 2, 100))
sig[0, 0, 1, 5] = np.nan
print("coh with one NaN sample:", get_metric("coh").compute(sig, 100, (0, 1, 3, 2))[0, 0])
