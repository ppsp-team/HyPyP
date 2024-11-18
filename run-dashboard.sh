#!/bin/bash
set -e

shiny run hypyp/app/dashboard.py --reload --host 0.0.0.0 --port 8000

