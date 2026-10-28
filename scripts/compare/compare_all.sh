#!/usr/bin/env bash
# Compare every branch's outputs with master's and write out/summary.md.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/gdumas/Github/HyPyP/.venv/bin/python
OUT=${1:-$HERE/out}
: > "$OUT/summary.md"
for name in b01 b02 b03 b04 b05 b06 b07 b08 b09 b10 b11; do
    "$PY" "$HERE/compare.py" "$OUT/master.npz" "$OUT/$name.npz" "$name" >> "$OUT/summary.md"
    echo >> "$OUT/summary.md"
done
cat "$OUT/summary.md"
