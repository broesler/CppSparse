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
N = 11

# Only off-diagonal elements
rows = np.r_[5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]
cols = np.r_[0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]
vals = np.ones((rows.size,))

# Values for the lower triangle
L = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))

# Create the symmetric matrix A
A = L + L.T

# Set the diagonal to ensure positive definiteness
A.setdiag(np.arange(10, 21))

# scipy Cholesky is only implemented for dense matrices
A = A.toarray()

# R = la.cholesky(A, lower=True)
R = csparse.chol_up(A, lower=True)

# NOTE etree is not implemented in scipy!
# Get the elimination tree
# [parent, post] = csparse.etree(A)
# Rp = la.cholesky(A(post, post), lower=True)
# assert (R.nnz == Rp.nnz)  # post-ordering does not change nnz

# Compute the row counts of the post-ordered Cholesky factor
row_counts = np.sum(R != 0, axis=1)
col_counts = np.sum(R != 0, axis=0)

# Count the nonzeros of the Cholesky factor of A^T A
ATA = A.T @ A
L_ATA = la.cholesky(ATA, lower=True)
nnz_cols = np.diff(sparse.csc_matrix(L_ATA).indptr)


# =============================================================================
# =============================================================================
