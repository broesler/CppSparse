#!/usr/bin/env python3
# =============================================================================
#     File: ex7.6_large_diags.py
#  Created: 2025-07-17 16:23
#   Author: Bernie Roesler
#
"""
Solution to Davis, Exercise 7.6: Heuristics for placing large entries on the
diagonal of a matrix.

Try the following method:
1. Scale a copy of *A* so that the largest entry in each column is 1.
2. Remove small entries from the matrix.
3. Use `cs_maxtrans` to find a zero-free diagonal.
4. If too many entries were dropped, decrease the drop tolerance and repeat.
5. Use the matching as a pre-ordering *Q*, and order :math:`AQ + (AQ)^T` with
   minimum degree.
6. Use a small pivot tolerance in `cs_lu` and determine how many off-diagonal
   pivots are found.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from scipy import sparse
from scipy.sparse import linalg as spla

import csparse

A = csparse.davis_example_amd()

# Create a random matrix with the same pattern as A
rng = np.random.default_rng(565656)
A.data = np.full(A.nnz, rng.uniform(size=A.nnz))

# Scale the matrix so that the largest entry in each column is 1
# NOTE sparse "A / max_elems" does not work ("dimension mismatch")
max_elems = A.max(axis=0).toarray()
A = sparse.csc_array(A.toarray() / max_elems)

# Remove small entries from the matrix
drop_tol = 0.5
A = A * (A > drop_tol)

if A.nnz < A.shape[0]:
    raise ValueError(f"Too many entries dropped with {drop_tol=:.2e}. "
                     "Try decreasing the drop tolerance.")
    # TODO retry in a loop with decreasing drop tolerance
    # print("Too many entries dropped, decreasing drop tolerance and retrying...")
    # drop_tol *= 0.5
    # A = A * (A > drop_tol)

# Find a zero-free diagonal using cs_maxtrans
jmatch, imatch = csparse.maxtrans(A, seed=0)

# Use the matching as a pre-ordering Q
AQ = A[:, jmatch]

# Compute the LU factorization with minimum degree ordering of AQ + (AQ)^T
L, U, p, q = csparse.lu(AQ, order='APlusAT', tol=1e-3)

np.testing.assert_allclose((L @ U).toarray(), AQ[p][:, q].toarray(), atol=1e-15)

# Determine how many off-diagonal pivots are found
off_diag_pivots = np.sum(np.abs(U.diagonal()) < 1.0)

# =============================================================================
# =============================================================================
