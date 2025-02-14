#!/usr/bin/env python3
# =============================================================================
#     File: householder_experiment.py
#  Created: 2025-02-12 16:06
#   Author: Bernie Roesler
#
"""
Experiment on p 79 of Davis. Test full Q vs Householder vectors.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg as la, sparse
from scipy.sparse import linalg as spla

from pathlib import Path

filename = Path('../data/west0479.csv')

dtypes = np.dtype([
    ('rows', np.int32),
    ('cols', np.int32),
    ('vals', np.float64)
])

data = np.genfromtxt(filename, delimiter=',', dtype=dtypes)

A = sparse.coo_matrix(
    (data['vals'],
     (data['rows'] - 1, data['cols'] - 1))
).tocsc()

# Use LU to get the permutation vector
# (inefficient! should have a dedicated colamd interface)
lu = spla.splu(A, permc_spec='COLAMD')
q = lu.perm_c
Aq = A[:, q].toarray()

# Get the QR decomposition
Q, R = la.qr(Aq, mode='full')

# Get the Householder vectors
(Qraw, tau), _ = la.qr(Aq, mode='raw')
V = np.tril(Qraw, -1) + np.eye(Aq.shape[0])
R_ = np.triu(Qraw)

np.testing.assert_allclose(R, R_)

# TODO check that these are correct? Book states "Q 38,070 vs V 3,906"
print("Q.nnz:", sparse.csc_matrix(Q).nnz)
print("V.nnz:", sparse.csc_matrix(V).nnz)

# -----------------------------------------------------------------------------
#         Plots
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=3, clear=True)
fig.set_size_inches(12, 4.8, forward=True)
# fig.suptitle('Householder vs Full Q')

ax = axs[0]
ax.spy(A, markersize=1)
ax.set_title('A')

ax = axs[1]
ax.spy(Q, markersize=1)
ax.set_title('Q')

ax = axs[2]
ax.spy(V, markersize=1)
ax.set_title('V')

plt.show()

# =============================================================================
# =============================================================================
