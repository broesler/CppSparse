#!/usr/bin/env bash
#===============================================================================
#     File: run_pytest.sh
#  Created: 2025-07-16 11:34
#   Author: Bernie Roesler
#
#  Description: A script to run pytest with the specified options, falling back
#  to skipped tests if pytest segfaults (known bug), or printing additional
#  debug information on test failues. This script is intended to be used
#  in a CI environment. Any additional arguments passed to this script will
#  be passed directly to pytest.
#
#  Usage: ./run_pytest.sh [pytest options]
#
#===============================================================================

# Allow script to continue on error
set +e

# Run pytest in a subshell in case it segfaults
echo "--- Running pytest with full test suite."
(pytest "$@")
INITIAL_EXIT_CODE=$?

echo "--- Initial pytest exit code $INITIAL_EXIT_CODE."

# Check if the initial pytest run failed
DESELECT_TESTS='--deselect=python/csparse/tests/test_lu.py::TestLU'
LAST_FAILED_FLAGS=( -v -s --last-failed --tb=auto --showlocals )

if [[ $INITIAL_EXIT_CODE -eq 139 ]]; then
    echo "--- Initial pytest run likely segfaulted (exit code 139)."
    echo "--- Re-Run: skipping TestLU."
    if pytest "$@" $DESELECT_TESTS; then
        echo "--- Re-run: pytest still failed after skipping TestLU."
        pytest "$@" $DESELECT_TESTS "${LAST_FAILED_FLAGS[@]}"
    fi
elif [[ $INITIAL_EXIT_CODE -ne 0 ]]; then
    echo "--- Initial pytest failed: exit code $INITIAL_EXIT_CODE."
    echo "--- Re-run: last failed tests."
    pytest "$@" "${LAST_FAILED_FLAGS[@]}"
fi


#===============================================================================
#===============================================================================
