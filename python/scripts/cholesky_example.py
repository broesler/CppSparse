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
# scipy Cholesky is only implemented for dense matrices
A = csparse.davis_example_chol(format='ndarray')

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
