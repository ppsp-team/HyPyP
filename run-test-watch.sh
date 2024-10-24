#!/bin/bash
set -e

# Exclude slow tests from pytest-watch
ptw $(ls tests/test_* | grep -Ev '(test_stats.py|test_prep.py)') hypyp -- -s

