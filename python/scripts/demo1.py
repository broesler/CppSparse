#!/usr/bin/env python3
# =============================================================================
#     File: demo1.py
#  Created: 2025-05-15 15:16
#   Author: Bernie Roesler
#
"""
Description: Python version of the C++Sparse/demo/demo1.cpp program.

Uses both C++Sparse function and Python functions and compares the results.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from scipy import sparse
from scipy.sparse import linalg as spla

import csparse


# Read the name of the matrix from the command line
if len(sys.argv) != 2:
    print("Usage: python demo1.py <filename>")
    sys.exit(1)

filename = Path(sys.argv[1])

# For testing:
# filename = Path('../../data/t1')

# Load the matrix from the file
data = np.genfromtxt(
    filename,
    delimiter=' ',
    dtype=[
        ('rows', int),
        ('cols', int),
        ('vals', float)
    ]
)

# Build the matrix two ways
A = sparse.coo_array((data['vals'], (data['rows'], data['cols']))).tocsc()
Ac = csparse.COOMatrix(data['vals'], data['rows'], data['cols']).tocsc()

print(f"A  difference: {spla.norm(A - Ac.toscipy(), ord=1)}")

# Plot the matrix
fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
ax = axs[0, 0]
ax.set_title(r'$A$')
csparse.cspy(A, ax=ax)

# Transpose the matrix two ways
AT = A.T
ATc = Ac.transpose()

print(f"AT difference: {spla.norm(AT - ATc.toscipy(), ord=1)}")

ax = axs[0, 1]
ax.set_title(r'$A^T$')
csparse.cspy(AT, ax=ax)

# Multiply A @ A.T
C = A @ AT
Cc = Ac @ ATc  # == Ac.dot(ATc)

print(f"C  difference: {spla.norm(C - Cc.toscipy(), ord=1)}")

ax = axs[1, 0]
ax.set_title(r'$C = A A^T$')
csparse.cspy(C, ax=ax)

N = A.shape[1]
I = sparse.eye_array(N)
cnorm = spla.norm(C, ord=1)
D = C + I * cnorm
Dc = Cc + csparse.csc_from_scipy(I * cnorm)

print(f"D  difference: {spla.norm(D - Dc.toscipy(), ord=1)}")

ax = axs[1, 1]
ax.set_title(r'$D = C + |C| I$')
csparse.cspy(D, ax=ax)

plt.show()

# =============================================================================
# =============================================================================
