# LU Decomposition for Singular Matrices

# Exercise 6.5: LU for square, singular matrices
## Case: Two pairs of linearly dependent columns

```python
Ac[:, 3] = 2.0 * Ac[:, 5]  # 2 linearly dependent column WORKS
Ac[:, 2] = 3.0 * Ac[:, 4]  # 2 *sets* of linearly dependent columns
```

If we run the command

```python
Ac = Ac.dropzeros()  # or Ac =  Ac.to_canonical()
```

The test fails, because `csparse.lu` creates an invalid permutation vector:

```
p_inv: [0, 1, 7, 7, 2, 3, 6, 4]  # 7 repeated for A.to_canonical()
p_inv: [0, 1, 5, 7, 2, 3, 6, 4]  # fine with explicit zeros
```

What is happening? The lower triangular solve, `spsolve`, does not care about
the actual values of the matrix, but only about the structure. Thus, when we
have explicit zeros, we end up finding a "pivot" that is exactly zero 
(see `k: 5` below).

Since we're allowing singular matrices, instead of throwing an error when 
`a = 0`, we set the flag and continue. `ipiv` is *not* equal to -1, however, so
we choose a pivot (which is identically zero) and set `p_inv[ipiv] = k`.

p_inv: [0, 1, -1, 7, 2, 3, 6, 4]


```bash
% ./test_csparse '[ex6.5]' -c 'Two pairs of linearly dependent columns'
Filters: [ex6.5]
Randomness seeded to: 4058665505
k: 0, ipiv: 0, a: 11, pivot: 11
     p_inv: [0, -1, -1, -1, -1, -1, -1, -1]
k: 1, ipiv: 1, a: 12, pivot: 12
     p_inv: [0, 1, -1, -1, -1, -1, -1, -1]
k: 2, ipiv: 4, a: 45, pivot: 45
     p_inv: [0, 1, -1, -1, 2, -1, -1, -1]
k: 3, ipiv: 5, a: 31.8667, pivot: 31.8667
     p_inv: [0, 1, -1, -1, 2, 3, -1, -1]
k: 4, ipiv: 7, a: 1.30649e-17, pivot: 1.30649e-17
     p_inv: [0, 1, -1, -1, 2, 3, -1, 4]
k: 5, ipiv: -1, a: -1
k: 6, ipiv: 6, a: 16.9167, pivot: 16.9167
     p_inv: [0, 1, -1, -1, 2, 3, 6, 4]
k: 7, ipiv: 3, a: 0.0537394, pivot: -0.0537394
     p_inv: [0, 1, -1, 7, 2, 3, 6, 4]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_csparse is a Catch2 v3.8.1 host application.
Run with -? for options

-------------------------------------------------------------------------------
Exercise 6.5: LU for square, singular matrices
-------------------------------------------------------------------------------
test/test_lu.cpp:424
...............................................................................

test/test_lu.cpp:51: FAILED:
  CHECK_THAT( res.p_inv, UnorderedEquals(row_perm) )
with expansion:
  { 0, 1, 7, 7, 2, 3, 6, 4 } UnorderedEquals: { 0, 1, 2, 3, 4, 5, 6, 7 }
with message:
  Two pairs of linearly dependent columns

===============================================================================
test cases:  1 |  0 passed | 1 failed
assertions: 32 | 31 passed | 1 failed

(dev311) (main *>) C++Sparse % make debug
clang++ -Wall -pedantic -std=c++20 -DDEBUG -glldb -O0   -fno-inline -fno-omit-frame-pointer  -Iinclude -Itest -I/opt/homebrew/include -c test/test_lu.cpp -o test/test_lu.o
clang++ -Wall -pedantic -std=c++20 -DDEBUG -glldb -O0   -fno-inline -fno-omit-frame-pointer  -Iinclude -Itest -I/opt/homebrew/include -o test_csparse test/test_cholesky.o test/test_coomatrix.o test/test_cscmatrix.o test/test_csparse.o test/test_fillreducing.o test/test_gaxpy.o test/test_helpers.o test/test_lu.o test/test_qr.o test/test_solve.o test/test_trisolve.o test/test_utils.o src/cholesky.o src/coo.o src/csc.o src/example_matrices.o src/fillreducing.o src/lu.o src/qr.o src/solve.o src/sparse_matrix.o src/utils.o -L/opt/homebrew/lib -lcatch2 -lCatch2Main
(dev311) (main *>) C++Sparse % ./test_csparse '[ex6.5]' -c 'Two pairs of linearly dependent columns'
Filters: [ex6.5]
Randomness seeded to: 2647046455
k: 0, ipiv: 0, a: 11, pivot: 11
     p_inv: [0, -1, -1, -1, -1, -1, -1, -1]
k: 1, ipiv: 1, a: 12, pivot: 12
     p_inv: [0, 1, -1, -1, -1, -1, -1, -1]
k: 2, ipiv: 4, a: 45, pivot: 45
     p_inv: [0, 1, -1, -1, 2, -1, -1, -1]
k: 3, ipiv: 5, a: 31.8667, pivot: 31.8667
     p_inv: [0, 1, -1, -1, 2, 3, -1, -1]
k: 4, ipiv: 7, a: 1.30649e-17, pivot: 1.30649e-17
     p_inv: [0, 1, -1, -1, 2, 3, -1, 4]
k: 5, ipiv: 2, a: 0, pivot: 0
     p_inv: [0, 1, 5, -1, 2, 3, -1, 4]
k: 6, ipiv: 6, a: 16.9167, pivot: 16.9167
     p_inv: [0, 1, 5, -1, 2, 3, 6, 4]
k: 7, ipiv: 3, a: 0.0537394, pivot: -0.0537394
     p_inv: [0, 1, 5, 7, 2, 3, 6, 4]
===============================================================================
All tests passed (32 assertions in 1 test case)
```

