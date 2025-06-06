#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ..

export HYPYP_NIRS_DATA_PATH="/media/patrice/My Passport/"

shiny run hypyp/app/shiny_dashboard.py --reload --host 0.0.0.0 --port 8000

