#!/usr/bin/env python3
# =============================================================================
#     File: lu_example.py
#  Created: 2025-03-31 12:13
#   Author: Bernie Roesler
#
"""Compute the LU decomposition of a matrix using scipy and csparse."""
# =============================================================================

import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as spla

import csparse


def allclose(a, b, atol=1e-15):
    """Check if two arrays are close to each other."""
    return np.testing.assert_allclose(a, b, atol=atol)


# Define the example matrix from Davis, Figure 4.2, p. 39
A = csparse.davis_example_qr(10)  # scipy.sparse.csc_array
Ac = csparse.CSCMatrix(A.data, A.indices, A.indptr, A.shape)

M, N = Ac.shape

# ---------- Permute the matrix rows arbitrarily
p_input = np.r_[5, 1, 7, 0, 2, 6, 4, 3]
p_input_inv = csparse.inv_permute(p_input)  # [3, 1, 4, 7, 6, 0, 5, 2]
Ac = Ac.permute_rows(p_input_inv)

# ---------- Create a numerically rank-deficient matrix
# for i in range(N):
    # Numerical rank deficiency (linearly dependent rows/columns)
    # This set of linearly dependent columns leaves row 3 as a zero row
    # Ac[i, 3] = 2.0 * Ac[i, 5]  # 2 linearly dependent column WORKS
    # Ac[i, 2] = 3.0 * Ac[i, 4]  # 2 *sets* of linearly dependent columns

    # Two sets of linearly dependent columns, with *NO* zero rows
    # Ac[i, 2] = 2.0 * Ac[i, 6]  # 2 linearly dependent column WORKS
    # Ac[i, 4] = 2.0 * Ac[i, 5]  # 2 *sets* of linearly dependent columns

    # Two sets of linearly dependent rows, with *NO* zero columns
    # Ac[3, i] = 2.0 * Ac[4, i]  # 2 linearly dependent rows WORKS
    # Ac[2, i] = 2.0 * Ac[5, i]  # 2 *sets* of linearly dependent rows WORKS

    # Numerical rank deficiency (zero rows/columns)
    # Ac[3, i] = 0.0  # zero row WORKS

    # for j in [2, 3, 5]:
    #     Ac[j, i] = 0.0  # multiple zero rows WORKS (but not for scipy.sparse)

    # Ac[i, 3] = 0.0  # WORKS single zero column

    # for j in [2, 3, 5]:
    #     Ac[i, j] = 0.0  # multiple zero columns WORKS


# ---------- Structural rank deficiency: remove zero rows and columns
# Ac = Ac.dropzeros()

# ---------- Create a rectangular matrix
# r = 3

# ----- M < N
# FIXME csparse.lu fails when *permuted* and M < N. L not lower tri!!
# NOTE not actually tested in C++Sparse. Only unpermuted case is tested.
# Permuted case should still work, however, since numpy can do it.

# Ac = Ac.slice(0, M - r, 0, N)  # (M-r, N)
# Ac = Ac[:M - r, :]  # (M-r, N)

# L -> (6, 6) == (M-r, M-r)
# U -> (6, 8) == (M-r, N)

# ----- M > N
# Ac = Ac.slice(0, M, 0, N - r)  # (M, N-r)
# Ac = Ac[:, :N - r]

# L -> (8, 8) == (M, M)
# U -> (8, 6) == (M, N-x)


# -----------------------------------------------------------------------------
#         Run the tests
# -----------------------------------------------------------------------------
# Convert to dense and sparse formats
As = csparse.scipy_from_csc(Ac)  # retain non-canonical format
A = As.toarray()

rank = np.linalg.matrix_rank(A)
# print("Size of A:", Ac.shape)
# print("Rank of A:", rank)

# print("A:")
# print(A)

# Compute the LU decomposition of A
# Scipy dense
pd, Ld, Ud = la.lu(A, p_indices=True)

# print("pd:")
# print(pd)
# print("Ld:")
# print(Ld)
# print("Ud:")
# print(Ud)

allclose(Ld[pd] @ Ud, A)

# C++Sparse
try:
    L, U, p, q = csparse.lu(As, order='Natural')

    allclose((L @ U).toarray(), A[p[:, np.newaxis], q])
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
    lu = spla.splu(As, permc_spec='NATURAL')  # no column reordering
    p_, L_, U_, q_ = lu.perm_r, lu.L, lu.U, lu.perm_c

    np.testing.assert_allclose(p_, csparse.inv_permute(p))  # not always true!
    allclose((L_ @ U_)[p_[:, np.newaxis], q_].toarray(), As.toarray())
    allclose(Ld, L_.toarray())  # not necessarily identical!
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


# -----------------------------------------------------------------------------
#         Test Incomplete LU decomposition
# -----------------------------------------------------------------------------
# NOTE spilu does not drop L entries that are < drop_tol?
#  * In SuperLU, 0 <= tol <= 1, because the drop tolerance is a fraction of the
#    maximum entry in each column.
# drop_tol = 0.08
drop_tol = 1.0  # drop everything off-diagonal -> FIXME does nothing?
# drop_tol = np.inf  # drop everything off-diagonal -> FIXME does nothing?
# drop_tol = 0.0  # keep everything

# Scipy.sparse
ilu = spla.spilu(As, drop_tol=drop_tol, permc_spec='NATURAL')

p_, L_, U_ = ilu.perm_r, ilu.L, ilu.U

if drop_tol == 0:
    allclose(L_.toarray(), Ld)
    allclose(U_.toarray(), Ud)
    allclose((L_[p_] @ U_).toarray(), A)
# elif drop_tol >= 1:  # only diagonals
#     allclose(L_.toarray(), np.eye(L_.shape[0]))  # FIXME both fail!
#     allclose(U_.diagonal(), A.diagonal())

# print(f"---------- ilu ({drop_tol=}):")
# print("p_:", p_)
# print("L_:")
# print(L_.toarray())
# print("U_:")
# print(U_.toarray())


# =============================================================================
# =============================================================================
