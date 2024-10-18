#!/bin/bash
set -e

export PYTHONPATH=".:$PYTHONPATH" 
export PYTHONPATH="/home/patrice/work/ppsp/pywt:$PYTHONPATH"
export PYTHONPATH="/home/patrice/work/ppsp/pycwt:$PYTHONPATH"

echo $PYTHONPATH

shiny run hypyp/app/dashboard.py --reload --host 0.0.0.0

