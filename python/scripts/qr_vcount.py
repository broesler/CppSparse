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

import numpy as np
from scipy import linalg as la, sparse

import csparse

# Matrix from Davis Figure 5.1, p 74.
A = csparse.davis_example_qr(format='csc')

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
