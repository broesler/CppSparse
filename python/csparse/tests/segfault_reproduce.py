#!/usr/bin/env python3
# =============================================================================
#     File: segfault_reproduce.py
#  Created: 2025-07-15 14:04
#   Author: Bernie Roesler
#
"""
A segfault often occurs in pytest when `scipy.sparse.linalg.splu` is called
with the "Pajek/GD99_c" matrix from the SuiteSparse collection.

This script is designed to reproduce the segfault in a controlled environment.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import suitesparseget as ssg

from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse import linalg as spla


# Load the Pajek/GD99_c matrix
df = ssg.get_index()
problem = ssg.get_problem(index=df, group="Pajek", name="GD99_c")
A = problem.A

print("Pre-conversion:")
print(f"{type(A) = }")

# Convert A to a sparse matrix
A = sparse.csr_array(A)

print("Post-conversion:")
print(f"{type(A) = }")

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title("Pajek/GD99_c matrix")

# Factor the matrix using dense LU
P, L, U = la.lu(A.toarray())

assert_allclose(P @ L @ U, A.toarray())

ax = axs[0, 1]
ax.spy(L + U, markersize=1)
ax.set_title("Dense L + U")

# Factor the matrix using sparse LU
permc_specs = [
    "NATURAL",        # segfaults or "Factor is exactly singular"
    "MMD_ATA",        # "failed to factorize matrix"
    "MMD_AT_PLUS_A",
    "COLAMD"
]

# NOTE
# The "NATURAL" permutation spec causes a segfault,
# or gives "Factor is exactly singular". The segfault is non-deterministic.
#
# The other permutations cause splu to fail with this message:
#   Failed to factorize matrix: failed to factorize matrix at line 110 in file
#   ../scipy/sparse/linalg/_dsolve/SuperLU/SRC/dsnode_bmod.c
#
# The offending lines are:
# 107 #if SCIPY_FIX
# 108        if (nsupr < nsupc) {
# 109            /* Fail early rather than passing in invalid parameters to TRSV. */
# 110            ABORT("failed to factorize matrix");
# 111        }
# 112 #endif
#

for permc_spec in permc_specs:
    print(f"Using permc_spec: {permc_spec}")
    try:
        lu = spla.splu(A, permc_spec=permc_spec)

        assert_allclose((lu.L @ lu.U)[lu.perm_r][:, lu.perm_c].toarray(), A.toarray())

        ax = axs[1, 1]
        ax.spy(lu.L + lu.U, markersize=1)
        ax.set_title("Sparse L + U")

    except RuntimeError as e:
        if "Factor is exactly singular" in str(e):
            print(f"Matrix is singular: {e}")
        elif "failed to factorize matrix" in str(e):
            print(f"Failed to factorize matrix: {e}")
        else:
            raise e

plt.show()

# =============================================================================
# =============================================================================
