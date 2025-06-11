#!/usr/bin/env python3
# =============================================================================
#     File: conftest.py
#  Created: 2025-06-11 11:58
#   Author: Bernie Roesler
#
"""
Configuration file for pytest to set up the testing environment.
"""
# =============================================================================

def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--make-figures",
        action="store_true",
        default=False,
        help="Make and save figures for tests that generate plots."
    )

# =============================================================================
# =============================================================================
