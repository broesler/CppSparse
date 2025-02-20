#!/usr/bin/env python3
# =============================================================================
#     File: qr_example.py
#  Created: 2025-02-13 12:26
#   Author: Bernie Roesler
#
"""
Compute the QR decomposition of a matrix using scipy and csparse.

Use for testing our csparse implementation.
"""
# =============================================================================

# TODO convert this entire script to a pytest unit testing framework so that we
# can run through multiple tests like the identity matrix, the Davis example,
# the Strang example, etc.

import numpy as np

from scipy import sparse
from scipy import linalg as la

import csparse

# TODO move cholesky.py and qr.py functions into csparse.linalg module

atol = 1e-14


# ---------- Matrix from Davis Figure 5.1, p 74.
# N = 8
# rows = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
#              3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6]
# cols = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
#              0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7]
# vals = np.r_[np.arange(1, N), 0, np.ones(rows.size - N)]
# A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
# Ac = csparse.COOMatrix(vals, rows, cols, (N, N)).tocsc()

# ---------- Davis 4x4 example
# Ac = csparse.davis_example()
# A = csparse.to_scipy_sparse(Ac)
# N = A.shape[0]

# ---------- Create a random matrix
# N = 7
# rng = np.random.default_rng(565656)
# A = sparse.random_array((N, N), density=0.5, rng=rng, format='csc')
# A.setdiag(10*np.arange(1, N+1))  # ensure structural full rank

# ---------- Identity matrix
# Q and R should be the same as A
# N = 7
# A = sparse.eye_array(N).tocsc()

# ---------- Diagonal matrix
N = 7

# A = sparse.diags(np.arange(1, N+1)).tocsc()  # positive diagonal
# A only has a positive main diagonal
# R matches R_ entirely

# A = -sparse.diags(np.arange(1, N+1)).tocsc()  # negative diagonal
# A only has a negative main diagonal
# Diagonal of R is negated

A = sparse.diags([np.ones(N-1), np.arange(1, N+1)], [-1, 0]).tocsc()
# A has all positive diagonals
# R matches off-diagonals and R[-1, -1]
# Diagonal of R (except R[-1, -1]) is negated

# A = sparse.diags([np.ones(N-1), np.arange(1, N+1), np.ones(N-1)], [-1, 0, 1]).tocsc()
# A has all positive diagonals
# R matches off-diagonals and R[-1, -1]
# Diagonal of R (except R[-1, -1]) is negated

# A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tocsc()
# A has a negative main diagonal, with positive off-diagonals
# R matches on all but R[-1, -1], which is negated

# A = -sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tocsc()
# A has a positive main diagonal, with negative off-diagonals
# R matches off-diagonals and R[-1, -1]
# Diagonal of R (except R[-1, -1]) is negated


Ac = csparse.from_scipy_sparse(A, format='csc')

A_dense = A.toarray()
print(A_dense)

# TODO try permuting the rows of A_dense here with S.p_inv to see if we get the
# same V and beta as the csparse.qr function?

# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
(Qraw, tau), Rraw = la.qr(A_dense, mode='raw')
Q_, R_ = la.qr(A_dense)

V_ = np.tril(Qraw, -1) + np.eye(N)
Qr_ = csparse.qright(V_, tau)
Ql_ = csparse.qleft(V_, tau)

np.testing.assert_allclose(Qr_, Ql_.T, atol=atol)
np.testing.assert_allclose(Qr_, Q_, atol=atol)

# TODO try using csparse.qr *within* qr_right (instead of la.qr) to see if we
# get the same Q and R as la.qr?

# -----------------------------------------------------------------------------
#         Compute QR decomposition with csparse
# -----------------------------------------------------------------------------
# TODO the row and column permutations are stored in S, but the MATLAB cs_qr
# function returns p and q, and only uses S internally. We should implement
# this functionality in our csparse implementation.
S = csparse.sqr(Ac)
QRres = csparse.qr(Ac, S)

V, beta, R = QRres.V, QRres.beta, QRres.R

# Convert for easier debugging
V = V.toarray()
beta = np.r_[beta]
R = R.toarray()

# Get the actual Q matrix, don't forget the row permutation!
p = csparse.inv_permute(S.p_inv)
Q = csparse.qright(V, beta, p)
Ql = csparse.qleft(V, beta, p)

np.testing.assert_allclose(Q, Ql.T, atol=atol)

# Compare the Householder reflectors
# NOT true when we have a permutation of the rows of A
if np.all(p == np.arange(N)):
    np.testing.assert_allclose(V, V_, atol=atol)
    np.testing.assert_allclose(beta, tau, atol=atol)

# NOTE We are getting the correct values in Q and R, up to a sign change. The
# sign difference occurs when x[0] is positive, and the sign of Hx[0] is
# negative.
np.testing.assert_allclose(np.abs(Q), np.abs(Q_), atol=atol)
np.testing.assert_allclose(np.abs(R), np.abs(R_), atol=atol)

# Q == Q_ for all tested diagonal matrices, regardless of sign
# Q != Q_ in general. Columns are off by a sign change.
np.testing.assert_allclose(Q, Q_, atol=atol)
np.testing.assert_allclose(R, R_, atol=atol)

# NOTE this is the "unit test" for csparse.qr since we do not have a C++
# implementation of qright or qleft to compare against.
# Reproduce A = QR
np.testing.assert_allclose(Q @ R, A_dense, atol=atol)


# =============================================================================
# =============================================================================
