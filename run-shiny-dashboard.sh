#!/bin/bash
set -e

shiny run hypyp/app/shiny-dashboard.py --reload --host 0.0.0.0 --port 8000

