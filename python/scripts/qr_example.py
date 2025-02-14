#!/usr/bin/env python3
# =============================================================================
#     File: qr_example.py
#  Created: 2025-02-13 12:26
#   Author: Bernie Roesler
#
"""
Test the column counts of A^T A for QR decomposition.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as sla

import csparse

# Matrix from Davis Figure 5.1, p 74.
N = 8
rows = np.r_[0, 3, 1, 6, 1, 2, 6, 0, 2, 3, 4, 5, 7, 4, 5, 7, 0, 1, 3, 6, 7, 5, 6];
cols = np.r_[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7];
vals = np.ones(rows.size);

A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
A.setdiag(np.r_[np.arange(1, N), 0])

# Compute A^T A
ATA = A.T @ A

# Compute the Cholesky factor of A^T A
L = la.cholesky(ATA.toarray(), lower=True)

# Count the number of nonzeros in each column of the Cholesky factor
nnz_cols = np.diff(sparse.csc_matrix(L).indptr)

print("A = ")
print(A.toarray())
print("A.T @ A = ")
print(ATA.toarray())
print("nnz cols of A.T @ A = ")
print(nnz_cols)

# Compute the QR decomposition of A
(Qraw, tau), Rraw = la.qr(A.toarray(), mode='raw')
Q, R = la.qr(A.toarray())

# Create an arbitrary M > N matrix
# M = 10
# N = 8
# A = sparse.rand(M, N, density=0.5, format='csc')
# A.setdiag(np.arange(1, N+1))
# A = A.toarray()

# (Qraw, tau), Rraw = la.qr(A, mode='raw')
# # Qraw.shape = (M, N)
# # Rraw.shape = (N, N)

# Q, R = la.qr(A)
# # Q.shape = (M, M)
# # R.shape = (M, N)

# Compute QR decomposition with csparse
Ac = csparse.COOMatrix(vals, rows, cols, N, N).tocsc()

S = csparse.sqr(Ac)
# VbR = csparse.qr(Ac, S)
# V, beta, Rraw = VbR.V, VbR.beta, VbR.R
V, beta, Rraw = csparse.qr(Ac, S)

print(V.shape)

# FIXME V needs to have slicing or conversion to csc_array, beta needs to be
# an ndarray. Rraw is not actually used here.
# Get the actual Q matrix
# Q = csparse.qright(sparse.eye(N), V, beta)


# =============================================================================
# =============================================================================
