#!/bin/bash
set -e

PYTHONPATH="$PYTHONPATH:." 

shiny run hypyp/app/dashboard.py --reload --host 0.0.0.0

