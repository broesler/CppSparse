#!/usr/bin/env python3
# =============================================================================
#     File: qr_vcount.py
#  Created: 2025-02-17 13:18
#   Author: Bernie Roesler
#
"""
Compute the Cholesky factor of A^T A for C++ testing of `vcount` function.
"""
# =============================================================================

# Matrix from Davis Figure 5.1, p 74.
N = 8
rows = np.r_[0, 3, 1, 6, 1, 2, 6, 0, 2, 3, 4, 5, 7, 4, 5, 7, 0, 1, 3, 6, 7, 5, 6]
cols = np.r_[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7]
vals = np.ones(rows.size)

A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
A.setdiag(np.r_[np.arange(1, N), 0])

# Compute A^T A
ATA = A.T @ A

# Compute the Cholesky factor of A^T A
L = la.cholesky(ATA.toarray(), lower=True)

# Count the number of nonzeros in each column of the Cholesky factor
nnz_cols = np.diff(sparse.csc_matrix(L).indptr)

print("A = ")
print(A.toarray())
print("A.T @ A = ")
print(ATA.toarray())
print("nnz cols of A.T @ A = ")
print(nnz_cols)


# =============================================================================
# =============================================================================
