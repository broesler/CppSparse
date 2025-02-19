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
import sys
sys.path.append('./')

from qr import qr_right

atol = 1e-14


# ---------- Matrix from Davis Figure 5.1, p 74.
N = 8
rows = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
             3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6]
cols = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
             0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7]
vals = np.r_[np.arange(1, N), 0, np.ones(rows.size - N)]

A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))

Ac = csparse.COOMatrix(vals, rows, cols, (N, N)).tocsc()

# ---------- Davis 4x4 example
# Ac = csparse.davis_example()
# A = csparse.to_scipy_sparse(Ac)
# N = A.shape[0]

# ---------- Identity matrix
# Q and R should be the same as A
# A = sparse.eye_array(N).tocsc()
# Ac = csparse.from_scipy_sparse(A, format='csc')

A_dense = A.toarray()

# TODO try permuting the rows of A_dense here with S.p_inv to see if we get the
# same V and beta as the csparse.qr function.

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

# Not sure what to glean from this comparison. Mostly the same H matrices up to
# a permutation and sign change.
Hs_ = [build_H(v, t) for v, t in zip(V_.T, tau)]
Hs = [build_H(v, t) for v, t in zip(V.T, beta)]

# Sum the number of nonzeros in the columns of V
# V_cols_nnz = np.sum(V != 0, axis=0)  # size of non-zero part of each H matrix

# print("V_ = ")
# print(V_)
# print("V = ")
# print(V)

# print("tau = ")
# print(tau)
# print("beta = ")
# print(beta)

# print("R_ = ")
# print(R_)
# print("R = ")
# print(R)

# Get the actual Q matrix, don't forget the row permutation!
p = csparse.inv_permute(S.p_inv)
Q = csparse.qright(V, beta, p)
Ql = csparse.qleft(V, beta, p)

np.testing.assert_allclose(Q, Ql.T, atol=atol)

# NOTE csparse Q, R with Davis book code is the negative of scipy Q, R
# np.testing.assert_allclose(R, -R_, atol=atol)
# np.testing.assert_allclose(Q, -Q_, atol=atol)

# With house LAPACK code
# csparse Q differs from scipy Q_ by a sign in the last column only!
np.testing.assert_allclose(Q[:, :-1], Q_[:, :-1], atol=atol)
np.testing.assert_allclose(Q[:, -1], -Q_[:, -1], atol=atol)
# csparse R differs from scipy R_ by a sign on the diagonal.
np.testing.assert_allclose(np.diag(R), -np.diag(R_), atol=atol)
np.testing.assert_allclose(np.triu(R, 1), np.triu(R_, 1), atol=atol)
np.testing.assert_allclose(np.tril(R, -1), np.tril(R_, -1), atol=atol)

# NOTE this is the "unit test" for csparse.qr since we do not have a C++
# implementation of qright or qleft to compare against.
# Reproduce A = QR
np.testing.assert_allclose(Q @ R, A_dense, atol=atol)


# =============================================================================
# =============================================================================
