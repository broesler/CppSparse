#!/usr/bin/env python3
# =============================================================================
#     File: cholesky_example.py
#  Created: 2025-02-20 19:56
#   Author: Bernie Roesler
#
"""
Example of computing the Cholesky decomposition of a matrix using scipy and
csparse.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy import sparse

import csparse

# Define the example matrix from Davis, Figure 4.2, p. 39
Ac = csparse.davis_example_chol()
A = Ac.toarray()  # scipy Cholesky is only implemented for dense matrices

M, N = A.shape

# R = la.cholesky(A, lower=True)
R = csparse.chol_up(A, lower=True)

# NOTE etree is not implemented in scipy!
# Get the elimination tree and post order it
parent = csparse.etree(Ac)
p = csparse.post(parent)
Rp = la.cholesky(A[p][:, p], lower=True)

# post-ordering does not change nnz
assert (sparse.csc_array(R).nnz == sparse.csc_array(Rp).nnz)

# Compute the row counts of the post-ordered Cholesky factor
row_counts = np.sum(Rp != 0, axis=1)
col_counts = np.sum(Rp != 0, axis=0)

# Count the nonzeros of the Cholesky factor of A^T A
ATA = A.T @ A
L_ATA = la.cholesky(ATA, lower=True)
nnz_cols = np.diff(sparse.csc_matrix(L_ATA).indptr)

# Compute the Cholesky factor using csparse C++ algorithms
order = 'APlusAT'
L, p = csparse.chol(Ac, order=order)
Ls = csparse.symbolic_cholesky(Ac, order=order).L
Ll = csparse.leftchol(Ac, order=order).L
Lr = csparse.rechol(Ac, order=order).L

np.testing.assert_allclose(L.indptr, Ls.indptr)
np.testing.assert_allclose(L.indices, Ls.indices)
np.testing.assert_allclose(L.toarray(), Ll.toarray())
np.testing.assert_allclose(Ll.toarray(), Lr.toarray())

np.testing.assert_allclose((L @ L.T).toarray(), A[p][:, p], atol=1e-14)

# Try to solve Ax = b to test numpy array of float input
b = np.ones(M) + np.arange(M) / M
x = csparse.lsolve(sparse.csc_array(R), b)

# Plot the matrix and Cholesky factor
fig, axs = plt.subplots(num=1, nrows=1, ncols=2, clear=True)
fig.set_size_inches((10, 5), forward=True)
ax = axs[0]
csparse.cspy(A, ax=ax)

ax = axs[1]
csparse.cspy(L + L.T - sparse.eye_array(N), ax=ax)

plt.show()

# =============================================================================
# =============================================================================
