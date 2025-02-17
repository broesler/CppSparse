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

import numpy as np

from scipy import sparse
from scipy import linalg as la

import csparse


Ac = csparse.davis_example()
A = csparse.to_scipy_sparse(Ac)
N = A.shape[0]

A_dense = A.toarray()

# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
(Qraw, tau), Rraw = la.qr(A_dense, mode='raw')
Q_, R_ = la.qr(A_dense)

atol = 1e-15
np.testing.assert_allclose(Q_ @ R_, A.toarray(), atol=atol)

V_ = np.tril(Qraw, -1) + np.eye(N)
Qr_ = csparse.qright(V_, tau)
Ql_ = csparse.qleft(V_, tau)

np.testing.assert_allclose(Qr_, Ql_.T, atol=atol)
np.testing.assert_allclose(Qr_, Q_, atol=atol)

# -----------------------------------------------------------------------------
#         Compute QR decomposition with csparse
# -----------------------------------------------------------------------------
S = csparse.sqr(Ac)
VbR = csparse.qr(Ac, S)

V, beta, R = VbR.V, VbR.beta, VbR.R

R = csparse.to_scipy_sparse(R)

np.testing.assert_allclose(V.toarray(), csparse.to_ndarray(V), atol=atol)
np.testing.assert_allclose(V.toarray(order='F'), csparse.to_ndarray(V), atol=atol)
np.testing.assert_allclose(V.toarray(order='C'), csparse.to_ndarray(V), atol=atol)
np.testing.assert_allclose(V.tocoo().toarray(), csparse.to_ndarray(V), atol=atol)

print("V.toarray() = ")
print(V.toarray())
print("beta = ")
print(np.r_[beta])
print("R = ")
print(R.toarray())

# Get the actual Q matrix
Qr = csparse.qright(V, beta)
Ql = csparse.qleft(V, beta)

np.testing.assert_allclose(Qr.toarray(), Ql.T.toarray(), atol=atol)

# FIXME Reproduce A = QR
# np.testing.assert_allclose((Qr @ R).toarray(), A_dense, atol=atol)
# np.testing.assert_allclose((Ql.T @ R).toarray(), A_dense, atol=atol)


# =============================================================================
# =============================================================================
