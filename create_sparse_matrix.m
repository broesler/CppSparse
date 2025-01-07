%===============================================================================
%     File: create_sparse_matrix.m
%  Created: 2025-01-07 15:32
%   Author: Bernie Roesler
%
%  Description: Create a random sparse matrix.
%
%===============================================================================

function A = create_sparse_matrix(M, N, density)
    nnz = ceil(density * M * N);
    i = randi(M, 1, nnz);
    j = randi(N, 1, nnz);
    v = rand(nnz, 1);

    A = sparse(i, j, v, M, N);
end

%===============================================================================
%===============================================================================
