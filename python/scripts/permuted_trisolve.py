#!/usr/bin/env python3
# =============================================================================
#     File: permuted_trisolve.py
#  Created: 2025-01-21 09:48
#   Author: Bernie Roesler
#
"""
Description:
"""
# =============================================================================

import numpy as np
import scipy.linalg as la

from scipy import sparse as sps

# Create the matrices
A = np.tile(np.c_[np.arange(1, 7)], 6)
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
bL = L @ x
bU = U @ x

PbU = PUQ @ x
PbL = PLQ @ x

assert np.allclose(la.solve(PLQ, PbL), x)
assert np.allclose(la.solve(PUQ, PbU), x)

print("L = ")
print(L)
print("U = ")
print(U)

print("bL = ")
print(bL)
print("bU = ")
print(bU)

print("PLQ = ")
print(PLQ)
print("PUQ = ")
print(PUQ)

print("PbL = ")
print(PbL)
print("PbU = ")
print(PbU)

# =============================================================================
# =============================================================================
