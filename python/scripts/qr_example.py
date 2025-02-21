#!/usr/bin/env python3
# =============================================================================
#     File: qr_example.py
#  Created: 2025-02-13 12:26
#   Author: Bernie Roesler
#
"""
Compute the QR decomposition of a matrix using scipy and csparse.

Use for testing usage of the csparse module API interactively.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy import linalg as la

import csparse

atol = 1e-12


# ---------- Matrix from Davis Figure 5.1, p 74.
A = csparse.davis_example_qr(format='csc')

# ---------- Davis 4x4 example
# A = csparse.davis_small_example(format='csc')

N = A.shape[0]
Ac = csparse.from_scipy_sparse(A, format='csc')

A_dense = A.toarray()
print("A = ")
print(A_dense)

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

# TODO can we turn off the row permutation in sqr/vcount?
# Get the actual Q matrix, don't forget the row permutation!
p = csparse.inv_permute(S.p_inv)
Q = csparse.apply_qright(V, beta, p)

# Permute the rows of A_dense here with S.p_inv to get the
# same V and beta as the csparse.qr function.
Ap = A_dense[p]

# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
(Qraw, tau), Rraw = la.qr(Ap, mode='raw')
Q_, R_ = la.qr(Ap)
V_ = np.tril(Qraw, -1) + np.eye(N)
Qr_ = csparse.apply_qright(V_, tau, p)

np.testing.assert_allclose(Q_ @ R_, Ap, atol=atol)
np.testing.assert_allclose(Q, Q_[S.p_inv], atol=atol)

# Reproduce A = QR
np.testing.assert_allclose(Q @ R, A_dense, atol=atol)
print("Q @ R = ")
print(Q @ R)


# =============================================================================
# =============================================================================
