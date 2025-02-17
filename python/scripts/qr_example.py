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
# Ac = csparse.COOMatrix(vals, rows, cols, N, N).tocsc()

S = csparse.sqr(Ac)
VbR = csparse.qr(Ac, S)
V, beta, Rraw = VbR.V, VbR.beta, VbR.R

# Get the actual Q matrix
Q = csparse.qright(sparse.eye(N), V, beta)
print("Q = ")
print(Q.toarray())


# =============================================================================
# =============================================================================
