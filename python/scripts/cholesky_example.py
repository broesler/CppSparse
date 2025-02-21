#!/usr/bin/env python3
# =============================================================================
#     File: cholesky_example.py
#  Created: 2025-02-20 19:56
#   Author: Bernie Roesler
#
"""
Example of computing the Cholesky decomposition of a matrix using scipy and
csparse.
"""
# =============================================================================

import numpy as np
import scipy.linalg as la

from scipy import sparse

import csparse

# Define the example matrix from Davis, Figure 4.2, p. 39
Ac = csparse.davis_example_chol()
A = Ac.toarray()  # scipy Cholesky is only implemented for dense matrices

# R = la.cholesky(A, lower=True)
R = csparse.chol_up(A, lower=True)

# NOTE etree is not implemented in scipy!
# Get the elimination tree and post order it
parent = csparse.etree(Ac)
p = csparse.post(parent)
Rp = la.cholesky(A[p][:, p], lower=True)

# post-ordering does not change nnz
assert (sparse.csc_array(R).nnz == sparse.csc_array(Rp).nnz)

# Compute the row counts of the post-ordered Cholesky factor
row_counts = np.sum(Rp != 0, axis=1)
col_counts = np.sum(Rp != 0, axis=0)

# Count the nonzeros of the Cholesky factor of A^T A
ATA = A.T @ A
L_ATA = la.cholesky(ATA, lower=True)
nnz_cols = np.diff(sparse.csc_matrix(L_ATA).indptr)


# =============================================================================
# =============================================================================
