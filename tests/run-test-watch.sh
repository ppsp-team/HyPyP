#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ..

# Exclude slow tests from pytest-watch
# Add all the extra arguments to the end
# Example to run a single test
#   poetry run ./run-test-watch.sh -k test_cohort_is_shuffle_no_duplicate

ptw $(ls tests/test_* | grep -Ev '(test_stats.py|test_prep.py)') hypyp -- -s $@

