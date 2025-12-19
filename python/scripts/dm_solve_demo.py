#!/usr/bin/env python3
# =============================================================================
#     File: dmsolve_demo.py
#  Created: 2025-12-18 15:15
#   Author: Bernie Roesler
# =============================================================================

"""Demo of dmsolve function and look at suitesparseget database for examples."""

import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse
import suitesparseget as ssg

import csparse


df = ssg.get_index()

cols = ['id', 'group', 'name', 'nrows', 'ncols', 'nblocks', 'ncc', 'sprank']

tf = df[cols].copy()
tf['min_size'] = tf[['nrows','ncols']].min(axis=1)

rf = tf.loc[
    (tf['nrows'] == tf['ncols'])      # square
    & (tf['nblocks'] > 1)             # many blocks
    # & (tf['sprank'] < tf['min_size'])  # rank deficient
    & (tf['sprank'] == tf['min_size'])  # full rank
    & (tf['nblocks'] < tf['sprank'])  # not all singleton blocks
].sort_values(by='nrows')


# 14 x 14, 3 blocks, no under/overdetermined blocks
# prob = ssg.get_problem(group='Oberwolfach', name='LFAT5')

# 85 x 85, 15 blocks, rank deficient (exactly singular, fails)
# prob = ssg.get_problem(group='JGD_Margulies', name='cat_ears_2_1')

# 100 x 100,  48 blocks, full rank, many off-diagonal blocks
prob = ssg.get_problem(group='Morandini', name='rotor1')

# Set up the problem
A = prob.A
M, N = A.shape
expect_x = np.arange(1, N + 1)
b = A @ expect_x

x = csparse.dm_solve(A, b)

p, q, r, s, cc, rr = csparse.dmperm(A)

fig, ax = plt.subplots(num=1, clear=True)
csparse.dmspy(A, norm='symlog', cmap='coolwarm', ax=ax)

# np.testing.assert_allclose(A @ x, b, atol=1e-15)
np.testing.assert_allclose(x, expect_x, atol=1e-15)



# =============================================================================
# =============================================================================
