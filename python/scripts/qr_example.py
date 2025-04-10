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
# print("A = ")
# print(A_dense)

# -----------------------------------------------------------------------------
#         Compute QR decomposition with csparse
# -----------------------------------------------------------------------------
# ---------- Compute using Householder reflections
QRres = csparse.qr(Ac)

V, beta, R = QRres.V, QRres.beta, QRres.R

# Convert for easier debugging
V = V.toarray()
R = R.toarray()

# TODO can we turn off the row permutation in sqr/vcount?
# Get the actual Q matrix, don't forget the row permutation!
p = csparse.inv_permute(QRres.p_inv)
Q = csparse.apply_qright(V, beta, p)

# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
# Permute the rows of A_dense here with QRres.p_inv to get the
# same V and beta as the csparse.qr function.
Ap = A_dense[p]

(Qraw, tau), Rraw = la.qr(Ap, mode='raw')
Q_, R_ = la.qr(Ap)
V_ = np.tril(Qraw, -1) + np.eye(N)
Qr_ = csparse.apply_qright(V_, tau, p)

np.testing.assert_allclose(Q_ @ R_, Ap, atol=atol)
np.testing.assert_allclose(Q, Q_[QRres.p_inv], atol=atol)

# Reproduce A = QR
np.testing.assert_allclose(Q @ R, A_dense, atol=atol)
# print("Q @ R = ")
# print(Q @ R)

# -----------------------------------------------------------------------------
#         QR with a M != N matrix and/or column pivoting
# -----------------------------------------------------------------------------
M, N = 8, 5
# M, N = 5, 8

Ar = A[:M, :N]

Ar_dense = Ar.toarray()
M, N = Ar.shape

print("Ar = ")
print(Ar_dense)

Arc = csparse.from_scipy_sparse(Ar)
QRr_res = csparse.qr(Arc)

Vr, beta_r, Rr, p_inv = QRr_res.V, QRr_res.beta, QRr_res.R, QRr_res.p_inv
Vr = Vr.toarray()
Rr = Rr.toarray()

# Get the actual Q matrix
pr = csparse.inv_permute(p_inv)
Qr = csparse.apply_qright(Vr, beta_r, pr)  # (M, M)

# Get the scipy version
# NOTE that scipy pivots columns to generate a non-increasing R diagonal (aka
# the norm of each vector)
Arp = Ar_dense[pr]
if pivoting:
    Qr_, Rr_, q_ = la.qr(Arp, pivoting=True)
    (Qraw_r, tau_r), _, _ = la.qr(Arp, mode='raw', pivoting=True)
else:
    Qr_, Rr_, = la.qr(Arp)
    (Qraw_r, tau_r), _ = la.qr(Arp, mode='raw')

Vr_ = np.tril(Qraw_r, -1)[:, :M] + np.eye(M, min(M, N))

print("Qr_ = ")
print(Qr_)
print("Vr_ = ")
print(Vr_)
print("Rr_ = ")
print(Rr_)

Qr_r = csparse.apply_qright(Vr_, tau_r)

if pivoting:
    np.testing.assert_allclose(Qr_ @ Rr_, Arp[:, q_], atol=atol)
    np.testing.assert_allclose((Qr_ @ Rr_)[:, q_], Arp, atol=atol)
    np.testing.assert_allclose(Qr @ Rr, Ar_dense[:, q], atol=atol)
else:
    np.testing.assert_allclose(Qr, Qr_[p_inv], atol=atol)
    np.testing.assert_allclose(Rr, Rr_, atol=atol)
    np.testing.assert_allclose(Qr @ Rr, Ar_dense, atol=atol)

print("Q @ R = ")
print(Qr @ Rr)


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
