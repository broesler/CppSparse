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

# Compute the QR decomposition of A with scipy
(Qraw, tau), Rraw = la.qr(A.toarray(), mode='raw')
Q_, R_ = la.qr(A.toarray())

atol = 1e-15
np.testing.assert_allclose(Q_ @ R_, A.toarray(), atol=atol)

# Compute QR decomposition with csparse
S = csparse.sqr(Ac)
VbR = csparse.qr(Ac, S)

V, beta, Rraw = VbR.V, VbR.beta, VbR.R

print("Rraw = ")
print(csparse.to_ndarray(Rraw))

# Get the actual Q matrix
Qr = csparse.qright(V, beta)
print("Qr = ")
print(Qr.toarray())

Ql = csparse.qleft(V, beta)
print("Ql = ")
print(Ql.toarray())

np.testing.assert_allclose(Qr.toarray(), Ql.T.toarray(), atol=atol)


# =============================================================================
# =============================================================================
