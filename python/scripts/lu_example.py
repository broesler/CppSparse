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

M, N = Ac.shape

# Add a random perturbation to the diagonal
for i in range(M):
    # Ac[i, i] += np.random.rand()
    Ac[i, i] += 10

# ---------- Permute the matrix rows arbitrarily
p = np.r_[5, 1, 7, 0, 2, 6, 4, 3]
Ac = Ac.permute_rows(p);

# ---------- Create a numerically rank-deficient matrix
# for i in range(N):
#     # Numerical rank deficiency (linearly dependent rows/columns)
#     # Ac[i, 3] = 2 * Ac[i, 5]  # 2 linearly dependent column WORKS
#     # Ac[i, 2] = 2 * Ac[i, 4]  # 2 *sets* of linearly dependent columns WORKS

#     # Ac[3, i] = 2 * Ac[4, i]  # 2 linearly dependent rows WORKS
#     # Ac[2, i] = 2 * Ac[5, i]  # 2 *sets* of linearly dependent rows WORKS

#     # Numerical rank deficiency (zero rows/columns)
#     # Ac[3, i] = 0.0  # zero row WORKS

#     for j in [2, 3, 5]:
#         Ac[j, i] = 0.0  # multiple zero rows WORKS (but not for scipy.sparse)

#     # Ac[i, 3] = 0.0  # WORKS single zero column
#     # for j in [2, 3, 5]:
#     #     Ac[i, j] = 0.0  # multiple zero columns WORKS


# ---------- Structural rank deficiency: remove zero rows and columns
# Ac = Ac.dropzeros()

# ---------- Create a rectangular matrix
# r = 3

# ----- M < N
# Ac = Ac.slice(0, M - r, 0, N)  # (M-r, N)

# L -> (6, 6) == (M-r, M-r)
# U -> (6, 8) == (M-r, N)

# ----- M > N
# Ac = Ac.slice(0, M, 0, N - r)  # (M, N-r)

# L -> (8, 8) == (M, M)
# U -> (8, 6) == (M, N-x)


# -----------------------------------------------------------------------------
#         Run the tests
# -----------------------------------------------------------------------------
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
pd, Ld, Ud = la.lu(A, p_indices=True)

print("pd:")
print(pd)
print("Ld:")
print(Ld)
print("Ud:")
print(Ud)

allclose(Ld[pd] @ Ud, A)

# C++Sparse
try:
    lu_res = csparse.lu(Ac)
    p_inv, L, U = lu_res.p_inv, lu_res.L, lu_res.U

    allclose((L[p_inv] @ U).toarray(), A)
    # np.testing.assert_allclose(p, pd)  # not necessarily identical!

except Exception as e:
    if "singular" in str(e):
        print("C++Sparse: Matrix is singular!")
    elif "square" in str(e):
        print("C++Sparse: Matrix is not square!")
    else:
        raise e

# Scipy sparse -- fails if singular or non-square
try:
    lu = sparse.linalg.splu(As, permc_spec='NATURAL')  # no column reordering
    p_, L_, U_ = lu.perm_r, lu.L, lu.U

    np.testing.assert_allclose(p_, p_inv)  # not necessarily identical!
    allclose((L_[p_] @ U_).toarray(), As.toarray())
    allclose(Ld, L_.toarray())
    allclose(Ud, U_.toarray())

except Exception as e:
    if "singular" in str(e):
        print("scipy.sparse: Matrix is singular!")
    elif "square" in str(e):
        print("scipy.sparse: Matrix is not square!")
    elif "failed" in str(e):
        print("scipy.sparse: Failed to factorize matrix!")
    else: 
        raise e

# Try lu_rightprv
pv, Lv, Uv = csparse.lu_rightprv(A)

np.testing.assert_equal(pv, p_inv)

Pv = np.eye(N)[:, pv]

allclose(Pv @ A, A[p])
allclose(Lv, Ld)
allclose(Uv, Ud)
allclose(Lv @ Uv, Pv @ A)

# =============================================================================
# =============================================================================
