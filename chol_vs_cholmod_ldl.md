# Davis (2006) Exercise 4.2
## Compare `cs_chol` with `cholmod_rowfac` and `LDL`

### LDL factorization
See [Davis (2005), Algorithm 849](davis849) and/or [the url](davis849url) for the description of the
`LDL` symbolic and numeric factorization.

The LDL factorization is a variant of the Cholesky factorization that solves:

$$
LDL^T = A
$$

where $L$ is a unit lower triangular matrix and $D$ is a diagonal matrix. The LDL package uses an
up-looking algorithm, just like in `cs_chol`.

Similarities:
* Both `cs_chol` and `LDL` are up-looking algorithms.
* Both `cs_chol` and `LDL` compute the factorization one row of $L$ at a time. This order guarantees
  that the *columns* of $L$ are in sorted order, and that there are no duplicate entries.
* Both algorithms use a stack to maintain the non-zero pattern of x in postorder while traversing
  the elimination tree.

Differences:
* `cs_chol` uses an *nearly* $O(|A|)$-time algorithm to compute the symbolic factorization
  (elimination tree and column counts of $L$), while `LDL` uses an $O(|L|)$-time algorithm.
  * The `LDL` algorithm is a simple traversal of each row subtree of the partially-constructed
    elimination tree. It computes the row and column counts of $L$ as it goes.
  * The `cs_chol` algorithm is more complex. It maintains an ancestor tree via a dijoint-set-union
    data structure to compute the column counts of $L$ in *nearly* $O(|A|)$ time.


### `cholmod_rowfac`
The `cholmod_rowfac` function computes the LDL or Cholesky factorization of a sparse matrix. It uses
the same algorithm as `LDL`, but it is more general. It can handle both symmetric and unsymmetric
matrices.


[davis849]:'/Users/bernardroesler/Library/Mobile Documents/com~apple~CloudDocs/Papers/Numerical Computation/Davis 2005 - Algorithm 849 - A Concise Sparse Cholesky Factorization Package.pdf'
[davis849url]: https://tinyurl.com/5eazdda6
