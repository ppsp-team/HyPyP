#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ..

shiny run hypyp/shiny/wavelet_explorer.py --reload --host 0.0.0.0 --port 8000

