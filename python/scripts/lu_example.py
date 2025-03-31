#!/usr/bin/env python3
# =============================================================================
#     File: lu_example.py
#  Created: 2025-03-31 12:13
#   Author: Bernie Roesler
#
"""
Example of computing the LU decomposition of a matrix using scipy and csparse.
"""
# =============================================================================

import numpy as np
import scipy.linalg as la

from scipy import sparse

import csparse


def allclose(a, b, atol=1e-15):
    return np.testing.assert_allclose(a, b, atol=atol)


# Define the example matrix from Davis, Figure 4.2, p. 39
Ac = csparse.davis_example_qr()

N = Ac.shape[0]

# Create a numerically rank-deficient matrix
for i in range(N):
    # Ac[i, 3] = Ac[i, 2]  # duplicate columns
    Ac[i, 3] = 0.0  # zero column

rank = np.linalg.matrix_rank(Ac.toarray())
print("Size of A:", Ac.shape)
print("Rank of A:", rank)

# Convert to dense and sparse formats
A = Ac.toarray()
As = sparse.csc_matrix(A)

print("A:")
print(A)

# Compute the LU decomposition of A
# Scipy dense
#   -- works with a 0 column
#   -- fails on duplicate columns (L, U computed, but L @ U != A)
pd, Ld, Ud = la.lu(A, p_indices=True)

# allclose(Ld @ Ud, A[pd])

# C++Sparse -- fails if singular
try:
    lu_res = csparse.lu(Ac)
    p_inv, L, U = lu_res.p_inv, lu_res.L, lu_res.U
    P = sparse.eye_array(A.shape[0]).tocsc()[p_inv]

    allclose((L @ U).toarray(), (P @ As).toarray())
    np.testing.assert_allclose(p_inv, pd)

except Exception as e:
    if "singular" in str(e):
        print("C++Sparse: Matrix is singular!")
    else:
        raise e

# Scipy sparse -- fails if singular
try:
    lu = sparse.linalg.splu(As, permc_spec='NATURAL')  # no column reordering
    p_, L_, U_ = lu.perm_r, lu.L, lu.U

    np.testing.assert_allclose(p_, pd)
    allclose((L_ @ U_).toarray(), As[p_].toarray())
    allclose(Ld, L_.toarray())
    allclose(Ud, U_.toarray())

except Exception as e:
    if "singular" in str(e):
        print("Scipy: Matrix is singular!")
    else: 
        raise e

# =============================================================================
# =============================================================================
