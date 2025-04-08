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
from scipy.sparse import linalg as spla

import csparse


def allclose(a, b, atol=1e-15):
    return np.testing.assert_allclose(a, b, atol=atol)


# Define the example matrix from Davis, Figure 4.2, p. 39
Ac = csparse.davis_example_qr()

M, N = Ac.shape

# Add a random perturbation to the diagonal to enforce expected pivoting
for i in range(M):
    # Ac[i, i] += np.random.rand()
    Ac[i, i] += 10

# ---------- Permute the matrix rows arbitrarily
p = np.r_[5, 1, 7, 0, 2, 6, 4, 3]
p_inv = csparse.inv_permute(p)
Ac = Ac.permute_rows(p_inv)

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
    lu = spla.splu(As, permc_spec='NATURAL')  # no column reordering
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


# -----------------------------------------------------------------------------
#         Test Incomplete LU decomposition
# -----------------------------------------------------------------------------
# NOTE spilu does not drop L entries that are < drop_tol?
#  * In SuperLU, 0 <= tol <= 1, because the drop tolerance is a fraction of the
#    maximum entry in each column.
# drop_tol = 0.08
drop_tol = 1.0  # drop everything off-diagonal -> FIXME does nothing?
# drop_tol = 0.0  # keep everything

# Scipy.sparse
ilu = spla.spilu(As, drop_tol=drop_tol, permc_spec='NATURAL')

p_, L_, U_ = ilu.perm_r, ilu.L, ilu.U

if drop_tol == 0:
    allclose(L_.toarray(), Ld)
    allclose(U_.toarray(), Ud)
    allclose((L_[p_] @ U_).toarray(), A)
# elif drop_tol == 1:  # only diagonals
#     allclose(L_.toarray(), np.eye(L_.shape[0]))  # FIXME both fail!
#     allclose(U_.diagonal(), A.diagonal())

# print(f"---------- ilu ({drop_tol=}):")
# print("p_:", p_)
# print("L_:")
# print(L_.toarray())
# print("U_:")
# print(U_.toarray())


# -----------------------------------------------------------------------------
#         Test 1-norm Estimate
# -----------------------------------------------------------------------------
normd = la.norm(A, 1)
norms = spla.norm(As, 1)
norm_est = spla.onenormest(A)

allclose(normd, norms)
allclose(normd, norm_est)

# Test out condition number estimate
# CSparse version:
#
# >> [L, U, P, Q] = lu(A);
# >> norm1est(L, U, P, Q)  % CSparse 1-norm estimate
# ans = 0.1153750055167834

# >> cond1est(A)  % CSparse 1-norm condition number estimate
# ans = 2.422875115852452

# >> condest(A)  % built-in 1-norm condition number estimate
# ans = 2.422875115852452

# C++Sparse version:
normc = csparse.norm1est(As)  # == 0.11537500551678347
κ = csparse.cond1est(As)      # == 2.422875115852453

# allclose(norm_est, κ)  # FIXME

print("---------- 1-norm estimate:")
print("   normd:", normd)
print("   norms:", norms)
print("norm_est:", norm_est)
print("   normc:", normc)
print("       κ:", κ)


# -----------------------------------------------------------------------------
#         Solve Ax = b
# -----------------------------------------------------------------------------
x = np.arange(1, N + 1)
b = A @ x

print("   solve(A, b):", la.solve(A, b))
print("spsolve(As, b):", spla.spsolve(As, b))

# LU solve
lu = spla.splu(As, permc_spec='NATURAL')  # no column reordering
L, U, p_, q = lu.L, lu.U, lu.perm_r, lu.perm_c

p_inv = csparse.inv_permute(p_)

Pb = b[p_inv]
y = spla.spsolve(L, Pb)
QTx = spla.spsolve(U, y)
x = QTx[q]

print('      LU solve:', x)

# =============================================================================
# =============================================================================
