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


# See Davis pp 7-8, Eqn (2.1)
N = 4
rows = np.r_[2,    1,    3,    0,    1,    3,    3,    1,    0,    2]
cols = np.r_[2,    0,    3,    2,    1,    0,    1,    3,    0,    1]
vals = np.r_[3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7]

A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))

# Compute the QR decomposition of A with scipy
(Qraw, tau), Rraw = la.qr(A.toarray(), mode='raw')
Q_, R_ = la.qr(A.toarray())

atol = 1e-15
np.testing.assert_allclose(Q_ @ R_, A.toarray(), atol=atol)

# Compute QR decomposition with csparse
Ac = csparse.COOMatrix(vals, rows, cols, N, N).tocsc()

S = csparse.sqr(Ac)
VbR = csparse.qr(Ac, S)
V, beta, Rraw = VbR.V, VbR.beta, VbR.R

# Get the actual Q matrix
Q = csparse.qright(sparse.eye(N), V, beta)
print("Q = ")
print(Q.toarray())


# =============================================================================
# =============================================================================
