#!/usr/bin/env python3
# =============================================================================
#     File: davis_example.py
#  Created: 2025-02-05 11:45
#   Author: Bernie Roesler
#
"""
Description: Davis book example from Equation (2.1) on pages 7--8.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla

import csparse as cs

# See Davis pp 7-8, Eqn (2.1)
i = np.r_[2,    1,    3,    0,    1,    3,    3,    1,    0,    2]
j = np.r_[2,    0,    3,    2,    1,    0,    1,    3,    0,    1]
v = np.r_[3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7]

Ac = sparse.coo_matrix((v, (i, j)), shape=(4, 4)).tocsc()

# print(Ac)

A = cs.COOMatrix(v, i, j, 4, 4)

# R = cs.COOMatrix.random(10, 10, density=0.1)
# print(np.r_[R.data()])

# with open('../data/t1', 'r') as fp:
#     C = cs.COOMatrix(fp)



# =============================================================================
# =============================================================================
