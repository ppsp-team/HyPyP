#!/usr/bin/env bash
# Compute the fixed output set on master and on every fix branch snapshot.
# Usage: bash run_all.sh <snapshots dir> <out dir>
set -u
SNAPS=$1
OUT=$2
PY=/home/gdumas/Github/HyPyP/.venv/bin/python
RUNNER=/home/gdumas/.claude/jobs/9fe7d438/tmp/run_in_cwd.py
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
: > "$OUT/run.log"
for name in master b01 b02 b03 b04 b05 b06 b07 b08 b09 b10 b11; do
    cd "$SNAPS/$name" || continue
    "$PY" "$RUNNER" "$HERE/outputs.py" "$OUT/$name.npz" >> "$OUT/run.log" 2>&1
    echo "=== $name: $(tail -1 "$OUT/run.log")" >> "$OUT/run.log"
done
grep "=== " "$OUT/run.log"
