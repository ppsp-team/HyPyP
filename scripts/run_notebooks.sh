#!/usr/bin/env bash
# run_notebooks.sh — Execute all tutorial notebooks sequentially and log results.
# Usage: bash scripts/run_notebooks.sh [notebook_name]
#   (no argument = run all; pass a name to run a single one)

set -euo pipefail

REPO=/Users/remyramadour/Workspace/PPSP/HypypDev/Updates/20260329/HyPyP
TUTORIAL_DIR="$REPO/tutorial"
LOG_DIR="$REPO/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

NOTEBOOKS=(
    "simulations.ipynb"
    "import_from_eeglab.ipynb"
    "manage_eeglab_montage.ipynb"
    "import_from_xdf.ipynb"
    "getting_started.ipynb"
    "fnirs_recording_inspection.ipynb"
    "fnirs_getting_started.ipynb"
)

# If a single notebook is requested, run only that one
if [[ $# -gt 0 ]]; then
    NOTEBOOKS=("$1")
fi

PASSED=()
FAILED=()

run_notebook() {
    local nb="$1"
    local log="$LOG_DIR/nb_${nb%.ipynb}_${TIMESTAMP}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶  $nb"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv run jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.kernel_name=hypyp-dev \
        --ExecutePreprocessor.timeout=600 \
        "$TUTORIAL_DIR/$nb" \
        2>&1 | tee "$log"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "  ✅  PASSED: $nb"
        PASSED+=("$nb")
    else
        echo "  ❌  FAILED: $nb  (see $log)"
        FAILED+=("$nb")
    fi
}

cd "$REPO"

for nb in "${NOTEBOOKS[@]}"; do
    run_notebook "$nb"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Passed: ${#PASSED[@]}"
for nb in "${PASSED[@]}"; do echo "       $nb"; done
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  ❌  Failed: ${#FAILED[@]}"
    for nb in "${FAILED[@]}"; do echo "       $nb"; done
    exit 1
fi
