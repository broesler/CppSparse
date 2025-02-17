#!/usr/bin/env python3
# =============================================================================
#     File: __init__.py
#  Created: 2025-02-14 08:59
#   Author: Bernie Roesler
#
"""
csparse: A Python wrapper for the CSparse++ library.

This module provides bindings for sparse matrix operations using pybind11.

Example usage:
    import csparse
    rows = [0, 1, 2, 0, 1, 2, 0, 2]
    cols = [0, 0, 0, 1, 1, 1, 2, 2]
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    A = csparse.CSparse(rows, cols, vals, 3, 3)
    print(A)

Author: Bernie Roesler
Date: 2025-02-14
Version: 0.1
"""
# =============================================================================

# Import all bindings from the csparse module
from .csparse import *
from .qr_utils import *
from .utils import *

__all__ = [x for x in dir() if not x.startswith('_')]

# =============================================================================
# =============================================================================
