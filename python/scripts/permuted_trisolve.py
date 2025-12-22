#!/usr/bin/env python3
# =============================================================================
#     File: permuted_trisolve.py
#  Created: 2025-01-21 09:48
#   Author: Bernie Roesler
#
"""Test code to solve permuted triangular systems."""
# =============================================================================

import numpy as np
import scipy.linalg as la

# Create the matrices
# A = np.tile(np.c_[np.arange(1, 7)], 6)
A = np.c_[np.arange(10, 70, 10)] + np.arange(1, 7)
L = np.tril(A)
U = np.triu(A)

# Create the permutation vectors/matrices
p = np.r_[5, 3, 0, 1, 4, 2]
q = np.r_[1, 4, 0, 2, 5, 3]

P = np.eye(len(p))[p]
Q = np.eye(q.size)[:, q]

PLQ = P @ L @ Q
PUQ = P @ U @ Q

x = np.arange(1, 7)

# Create the RHS
bL = PLQ @ x
bU = PUQ @ x

assert np.allclose(la.solve(PLQ, bL), x)
assert np.allclose(la.solve(PUQ, bU), x)

print("L = ")
print(L)
print("U = ")
print(U)

print("bL = ")
print(bL)
print("bU = ")
print(bU)

# =============================================================================
# =============================================================================
