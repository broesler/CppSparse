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
A = csparse.davis_example_qr()

# ---------- Davis 4x4 example
# A = csparse.davis_small_example()

N = A.shape[0]

A_dense = A.toarray()
# print("A = ")
# print(A_dense)

# -----------------------------------------------------------------------------
#         Compute QR decomposition with csparse
# -----------------------------------------------------------------------------
# ---------- Compute using Householder reflections
order = 'ATA'  # 'Natural' or 'ATA'

V, beta, R, p_inv, q = csparse.qr(A, order=order)

# Convert for easier debugging
V = V.toarray()
R = R.toarray()

# Get the actual Q matrix, don't forget the row permutation!
p = csparse.inv_permute(p_inv)
Q = csparse.apply_qright(V, beta, p)


# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
# Permute the rows of A_dense here with QRres.p_inv to get the
# same V and beta as the csparse.qr function.
Apq = A_dense[p][:, q]

(Qraw, tau), Rraw = la.qr(Apq, mode='raw')
Q_, R_ = la.qr(Apq)
V_ = np.tril(Qraw, -1) + np.eye(N)
Qr_ = csparse.apply_qright(V_, tau, p)

# Now we get the same Householder vectors and weights
np.testing.assert_allclose(V, V_, atol=atol)
np.testing.assert_allclose(beta, tau, atol=atol)
np.testing.assert_allclose(R, R_, atol=atol)

np.testing.assert_allclose(Q, Qr_, atol=atol)
np.testing.assert_allclose(Q, Q_[p_inv], atol=atol)
np.testing.assert_allclose(Q[p], Q_, atol=atol)

# Reproduce A = QR
np.testing.assert_allclose(Q_ @ R_, Apq, atol=atol)
np.testing.assert_allclose(Q_[p_inv] @ R_, A_dense[:, q], atol=atol)
np.testing.assert_allclose(Q @ R, A_dense[:, q], atol=atol)
# print("Q @ R = ")
# print(Q @ R)


# -----------------------------------------------------------------------------
#         QR with a M != N matrix
# -----------------------------------------------------------------------------
# M, N = 8, 5
M, N = 5, 8

Ar = A[:M, :N]

Ar_dense = Ar.toarray()

print("Ar = ")
print(Ar_dense)

Vr, beta_r, Rr, p_inv, q  = csparse.qr(Ar, order=order)

Vr = Vr.toarray()
Rr = Rr.toarray()

# Get the actual Q matrix
pr = csparse.inv_permute(p_inv)

Qr = csparse.apply_qright(Vr, beta_r, pr)  # (M, M)
Qrt = csparse.apply_qtleft(Vr, beta_r, pr).T  # (M, M)

# Get the scipy version
Arp = Ar_dense[pr][:, q]
# Arp = Ar_dense[:, q]
Qr_, Rr_ = la.qr(Arp)

(Qraw_r, tau_r), _ = la.qr(Arp, mode='raw')
Vr_ = np.tril(Qraw_r, -1)[:, :M] + np.eye(M, min(M, N))

print("Qr_ = ")
print(Qr_)
print("Rr_ = ")
print(Rr_)

# Qr_r = csparse.apply_qright(Vr_, tau_r)

np.testing.assert_allclose(Qr, Qr_[p_inv], atol=atol)
np.testing.assert_allclose(Rr, Rr_, atol=atol)
np.testing.assert_allclose(Qr @ Rr, Ar_dense[:, q], atol=atol)

print("Q @ R = ")
print(Qr @ Rr)

# -----------------------------------------------------------------------------
#         Try a random sparse A matrix
# -----------------------------------------------------------------------------
rng = np.random.default_rng(56565)
A = sparse.random(M, N, density=0.5, format='csc', dtype=np.float64, rng=rng)

# Set the diagonal to 1..M to ensure full rank
A.setdiag(np.arange(1, M + 1))

A = A.toarray()
print("A = ")
print(A)

Q, R = la.qr(A)
(Qraw, tau), _ = la.qr(A, mode='raw')
V = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))

# Run our own python QR
V_l, beta_l, R_l = csparse.qr_left(A)
Q_l = csparse.apply_qtleft(V_l, beta_l).T

np.testing.assert_allclose(V, V_l, atol=atol)
np.testing.assert_allclose(tau, beta_l, atol=atol)
np.testing.assert_allclose(Q, Q_l, atol=atol)
np.testing.assert_allclose(R, R_l, atol=atol)
np.testing.assert_allclose(Q_l @ R_l, A, atol=atol)

# Run our own python QR
A = A[:, :M]  # (5, 5)
A[2, :] = 0   # make it rank deficient
V, beta, R = csparse.qr_left(A)
Q = csparse.apply_qtleft(V, beta).T

# -----------------------------------------------------------------------------
#         Compute using Givens rotations
# -----------------------------------------------------------------------------
x = np.r_[3, 4]
G = csparse.givens(x)

# NOTE not sure what to do with these yet... there is no V or Q computed, and
# the two algorithms do not give equal results.
Rf = csparse.qr_givens_full(A_dense)
Rg = csparse.qr_givens(A_dense)

# print(G @ x)
np.testing.assert_allclose(np.abs(G @ x), np.r_[la.norm(x), 0], atol=atol)
# TODO work out what the sign of the rotation should be and test for various
# vectors.

# =============================================================================
# =============================================================================
